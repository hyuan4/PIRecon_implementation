import argparse
import csv
import os
from pathlib import Path
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusers import AutoencoderKL

# ============================================================
# Paths: change these if needed
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

IMG_ROOT = "../datasets/HAM10000/train"
VAL_IMG_ROOT = None
VQ_CKPT_PATH = "vqgan_finetuned_ham10000/vqgan_320_gan_final.ckpt"
VQ_CONFIG_PATH = "./checkpoints/vqgan_imagenet_f16_16384.yaml"
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# VAE-encoder-style student settings
FREEZE_VAE_BACKBONE = True
TRAIN_LAST_N_DOWNBLOCKS = 0
TRAIN_MID_BLOCK = False
HEAD_PROJ_CH = 256

OUT_CKPT_DIR = "checkpoints_softvq_distill_vaeenc"
OUT_VIS_DIR = "softvq_visuals_vaeenc"

# Distill preset switch
#   - "ham10000_512"
#   - "celebahq9k_320"
#   - "celebahq9k_512"
DISTILL_PRESET = "celebahq9k_256"

# taming-transformers path
TAMING_ROOT = "taming-transformers"
if TAMING_ROOT not in sys.path:
    sys.path.insert(0, TAMING_ROOT)

from taming.models.vqgan import VQModel


# ============================================================
# Utils
# ============================================================

def safe_torch_load(path, map_location="cpu"):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using `torch.load` with `weights_only=False`",
            category=FutureWarning,
        )
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)


CSV_IMAGE_COLUMNS = ("PATIENT_ID", "image", "path", "filepath", "file", "img", "img_path")


def resolve_repo_local_path(path_like, repo_root: Path = SCRIPT_DIR) -> Path:
    raw = str(path_like).strip()
    if len(raw) == 0:
        return Path(raw)

    src = Path(raw).expanduser()
    candidates = []
    if src.is_absolute():
        candidates.append(src)
        parts = src.parts
        if len(parts) >= 2 and parts[1] == "data":
            candidates.append(repo_root / raw.lstrip("/"))
        if repo_root.name in parts:
            anchor = parts.index(repo_root.name)
            rel = Path(*parts[anchor + 1 :])
            candidates.append(repo_root / rel)
    else:
        candidates.append(repo_root / src)
        candidates.append(src)

    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand.resolve()
    return candidates[0]


def resolve_repo_output_path(path_like, repo_root: Path = SCRIPT_DIR) -> Path:
    raw = str(path_like).strip()
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return repo_root / p


def resolve_distill_preset_name(name: str) -> str:
    x = str(name).strip().lower()
    if x in ("ham10000", "ham10000_512"):
        return "ham10000_512"
    if x in ("celeba", "celebahq", "celebahq9k", "celebahq9k_320"):
        return "celebahq9k_320"
    if x in ("celebahq9k_256", "celeba_256"):
        return "celebahq9k_256"
    if x in ("lyme", "lyme_demographic", "lyme_320"):
        return "lyme_320"
    raise RuntimeError(f"Unknown DISTILL_PRESET={name}. choices={list(DISTILL_PRESETS.keys())}")


def collect_images_from_csv(csv_path, exts=(".png", ".jpg", ".jpeg"), limit=100000):
    csv_path = resolve_repo_local_path(csv_path)
    exts_set = {str(x).lower() for x in exts}
    files = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        column = next((c for c in CSV_IMAGE_COLUMNS if c in fieldnames), None)
        if column is None:
            raise ValueError(
                f"CSV {csv_path} does not contain an image-path column. "
                f"Expected one of: {', '.join(CSV_IMAGE_COLUMNS)}"
            )
        for row in reader:
            raw = str(row.get(column, "")).strip()
            if len(raw) == 0:
                continue
            p = resolve_repo_local_path(raw)
            if p.suffix.lower() in exts_set:
                files.append(p)
            if len(files) >= int(limit):
                break
    return files


# ============================================================
# Dataset
# ============================================================

class ImageSourceDataset(Dataset):
    def __init__(self, root_dir, transform=None, exts=(".png", ".jpg", ".jpeg"), limit=100000):
        super().__init__()
        self.root_dir = resolve_repo_local_path(root_dir)
        self.transform = transform
        self.exts = {e.lower() for e in exts}
        self.files = []

        if self.root_dir.is_file() and self.root_dir.suffix.lower() == ".csv":
            self.files.extend(collect_images_from_csv(self.root_dir, exts=exts, limit=limit))
        elif self.root_dir.exists():
            for p in sorted(self.root_dir.iterdir())[:limit]:
                if p.is_file() and p.suffix.lower() in self.exts:
                    self.files.append(p.resolve())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, str(path)


DISTILL_PRESETS = {
    "ham10000_512": {
        "img_root": "../datasets/HAM10000/train",
        "val_root": None,
        "vq_ckpt_path": "vqgan_finetuned_ham10000/vqgan_320_gan_final.ckpt",
        "out_ckpt_dir": "checkpoints_softvq_distill_vaeenc",
        "out_vis_dir": "softvq_visuals_vaeenc",
        "img_size": 512,
        "batch_size": 8,
        "epochs": 20,
        "lr": 1e-4,
        "train_n": 6000,
    },
    "celebahq9k_320": {
        "img_root": "../datasets/celebahq256_9k/train",
        "val_root": "../datasets/celebahq256_9k/val",
        "vq_ckpt_path": "vqgan_finetuned_celebahq9k_256_320/vqgan_256_320_final.ckpt",
        "out_ckpt_dir": "checkpoints_softvq_distill_vaeenc_celebahq320",
        "out_vis_dir": "softvq_visuals_vaeenc_celebahq320",
        "img_size": 320,
        "batch_size": 8,
        "epochs": 20,
        "lr": 1e-4,
        "train_n": 8000,
    },
    "celebahq9k_256": {
        "img_root": "../datasets/celebahq256_9k/train",
        "val_root": "../datasets/celebahq256_9k/val",
        "vq_ckpt_path": "vqgan_finetuned_celebahq9k_256_320/vqgan_256_320_final.ckpt",
        "out_ckpt_dir": "checkpoints_softvq_distill_vaeenc_celebahq256",
        "out_vis_dir": "softvq_visuals_vaeenc_celebahq256",
        "img_size": 256,
        "batch_size": 8,
        "epochs": 20,
        "lr": 1e-4,
        "train_n": 8000,
    },
    "lyme_320": {
        "img_root": "data/lyme_demographic_train.csv",
        "val_root": "data/lyme_demographic_test.csv",
        "vq_ckpt_path": "vqgan_finetuned_lyme/vqgan_320_gan_final.ckpt",
        "out_ckpt_dir": "checkpoints_softvq_distill_vaeenc_lyme",
        "out_vis_dir": "softvq_visuals_vaeenc_lyme",
        "img_size": 320,
        "batch_size": 8,
        "epochs": 20,
        "lr": 1e-4,
        "train_n": -1,
    },
}


def _subset_dataset(ds: ImageSourceDataset, n: int) -> ImageSourceDataset:
    if int(n) <= 0 or int(n) >= len(ds.files):
        return ds
    out = ImageSourceDataset(ds.root_dir, transform=ds.transform, limit=0)
    out.files = ds.files[: int(n)]
    return out


def build_train_val_sets(root_dir, transform, train_n=6000, limit=100000, val_root=None, val_limit=50000):
    train_ds = ImageSourceDataset(root_dir, transform=transform, limit=limit)
    train_ds = _subset_dataset(train_ds, int(train_n))

    if val_root is not None and resolve_repo_local_path(val_root).exists():
        val_ds = ImageSourceDataset(val_root, transform=transform, limit=val_limit)
        return train_ds, val_ds

    # Fallback: split from same folder
    dataset = ImageSourceDataset(root_dir, transform=transform, limit=limit)
    files = dataset.files
    train_files = files[:train_n] if int(train_n) > 0 else files
    val_files = files[train_n:] if int(train_n) > 0 else []
    train_set = ImageSourceDataset(root_dir, transform=transform, limit=0)
    val_set = ImageSourceDataset(root_dir, transform=transform, limit=0)
    train_set.files = train_files
    val_set.files = val_files
    return train_set, val_set


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def psnr_torch(a, b, data_range=1.0, eps=1e-12):
    mse = F.mse_loss(a.float(), b.float(), reduction="mean").item()
    mse = max(mse, eps)
    return float(10.0 * torch.log10(torch.tensor((data_range ** 2) / mse)).item())


def grad_loss(a, b):
    ax = a[:, :, :, 1:] - a[:, :, :, :-1]
    ay = a[:, :, 1:, :] - a[:, :, :-1, :]
    bx = b[:, :, :, 1:] - b[:, :, :, :-1]
    by = b[:, :, 1:, :] - b[:, :, :-1, :]
    return F.l1_loss(ax, bx) + F.l1_loss(ay, by)


# ============================================================
# Load VQ teacher
# ============================================================

def load_vqgan_taming(config_path: str, ckpt_path: str, device="cuda", verbose=True):
    """
    Load taming-transformers VQGAN (VQModel) for inference.
    Compatible with:
      - original taming ckpt (dict with 'state_dict')
      - your finetuned ckpt (maybe dict with 'state_dict' OR raw state_dict)
    """
    cfg = OmegaConf.load(config_path)
    params = cfg.model.params

    # remove lossconfig for inference
    if isinstance(params, dict) and "lossconfig" in params:
        if verbose:
            print("[VQGAN] Removing lossconfig for inference...")
        params = dict(params)
        params.pop("lossconfig", None)

    model = VQModel(**params)

    ckpt = safe_torch_load(ckpt_path, map_location="cpu")

    # ---- get state_dict ----
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith(("encoder.", "decoder.", "quantize.", "loss.")) for k in ckpt.keys()):
        # looks like a raw state_dict dict
        sd = ckpt
    else:
        # fallback: treat as raw state_dict
        sd = ckpt

    # ---- strip unwanted keys + possible prefixes ----
    new_sd = {}
    for k, v in sd.items():
        # drop loss / discriminator if present
        if k.startswith(("loss", "perceptual_loss", "discriminator")):
            continue

        kk = k
        # common prefixes you might have in finetune script
        if kk.startswith("vq."):
            kk = kk[len("vq."):]
        if kk.startswith("module."):
            kk = kk[len("module."):]

        new_sd[kk] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    if verbose:
        print(f"[VQGAN] Loaded ckpt={ckpt_path}")
        print(f"[VQGAN] Missing={len(missing)}, Unexpected={len(unexpected)}")
        print("missing example:", missing[:20])
        print("unexpected example:", unexpected[:20])
        if hasattr(model, "quantize") and hasattr(model.quantize, "n_e"):
            print(f"[VQGAN] codebook n_e = {model.quantize.n_e}")
        if hasattr(model, "quantize") and hasattr(model.quantize, "e_dim"):
            print(f"[VQGAN] codebook e_dim = {model.quantize.e_dim}")

    model.eval().to(device)
    return model

class VQCodec:
    def __init__(self, vq_model, codebook_size: int):
        self.vq = vq_model
        self.codebook_size = int(codebook_size)

    @torch.no_grad()
    def encode_to_indices(self, x):
        """
        x: (B,3,H,W) in [-1,1]
        return: indices (B,Ht,Wt) int64
        """
        out = self.vq.encode(x)

        # taming-transformers: (quant, emb_loss, info)
        if isinstance(out, (tuple, list)) and len(out) >= 3:
            quant, _, info = out[0], out[1], out[2]
            indices = info[2]   # min_encoding_indices
        else:
            if hasattr(out, "quant") and hasattr(out, "indices"):
                quant, indices = out.quant, out.indices
            elif isinstance(out, dict) and ("quant" in out) and ("indices" in out):
                quant, indices = out["quant"], out["indices"]
            else:
                raise RuntimeError("VQ encode output format not recognized. Please adapt encode_to_indices().")

        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices)

        indices = indices.to(torch.int64)

        B = x.size(0)
        Ht, Wt = int(quant.shape[-2]), int(quant.shape[-1])

        if indices.dim() == 1:
            # (B*Ht*Wt,)
            if indices.numel() != B * Ht * Wt:
                raise RuntimeError(f"indices numel mismatch: got {indices.numel()}, expect {B*Ht*Wt}")
            indices = indices.view(B, Ht, Wt)

        elif indices.dim() == 2:
            # (B, Ht*Wt)
            if indices.size(0) != B:
                raise RuntimeError(f"indices batch mismatch: got {indices.size(0)}, expect {B}")
            if indices.size(1) != Ht * Wt:
                raise RuntimeError(f"indices length mismatch: got {indices.size(1)}, expect {Ht*Wt}")
            indices = indices.view(B, Ht, Wt)

        elif indices.dim() == 3:
            # already (B,Ht,Wt)
            pass

        else:
            raise RuntimeError(f"Unsupported indices dim={indices.dim()}, shape={tuple(indices.shape)}")

        return indices

    @torch.no_grad()
    def decode_from_indices(self, indices):
        """
        indices: (B, Ht, Wt) int64
        return: x_hat (B,3,H,W) in [-1,1]
        """
        B, Ht, Wt = indices.shape
        e_dim = int(self.vq.quantize.e_dim)

        indices_hw = indices.to(torch.long).view(B, Ht, Wt)

        # taming: get_codebook_entry(indices, shape=(B,H,W,C))
        z_q = self.vq.quantize.get_codebook_entry(
            indices_hw, shape=(B, Ht, Wt, e_dim)
        )

        if z_q.dim() == 4 and z_q.shape[1] == e_dim:
            z_q_nchw = z_q
        elif z_q.dim() == 4 and z_q.shape[-1] == e_dim:
            z_q_nchw = z_q.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(f"Unexpected z_q shape from codebook: {tuple(z_q.shape)}")

        x = self.vq.decode(z_q_nchw)
        return x.clamp(-1, 1)


# ============================================================
# Teacher outputs
# ============================================================

@torch.no_grad()
def get_teacher_outputs(vq_obj, x):
    """
    Supports either:
      1) raw taming VQModel
      2) your VQCodec wrapper

    Returns:
      quant_teacher: (B, Cq, Hq, Wq)
      index_teacher: (B, Hq, Wq)
      codebook:      (K, Cq)
    """
    # --------------------------------------------------
    # Case A: your VQCodec wrapper
    # --------------------------------------------------
    if hasattr(vq_obj, "encode_to_indices") and hasattr(vq_obj, "decode_from_indices"):
        indices = vq_obj.encode_to_indices(x)   # (B,Hq,Wq)

        B, Hq, Wq = indices.shape
        e_dim = int(vq_obj.vq.quantize.e_dim)

        # codebook
        codebook = vq_obj.vq.quantize.embedding.weight   # (K, Cq)

        # rebuild quant_teacher from indices
        z_q = vq_obj.vq.quantize.get_codebook_entry(
            indices.to(torch.long),
            shape=(B, Hq, Wq, e_dim)
        )

        # support NHWC / NCHW
        if z_q.dim() == 4 and z_q.shape[1] == e_dim:
            quant_teacher = z_q
        elif z_q.dim() == 4 and z_q.shape[-1] == e_dim:
            quant_teacher = z_q.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(f"Unexpected z_q shape from codebook: {tuple(z_q.shape)}")

        return quant_teacher, indices.long(), codebook

    # --------------------------------------------------
    # Case B: raw taming VQModel
    # --------------------------------------------------
    out = vq_obj.encode(x)
    quant_teacher, _, info = out
    index_teacher = info[2]

    B, Cq, Hq, Wq = quant_teacher.shape

    if index_teacher.dim() == 1:
        index_teacher = index_teacher.view(B, Hq, Wq)
    elif index_teacher.dim() == 2:
        if index_teacher.shape[1] == Hq * Wq:
            index_teacher = index_teacher.view(B, Hq, Wq)

    index_teacher = index_teacher.long()
    codebook = vq_obj.quantize.embedding.weight  # (K, Cq)

    return quant_teacher, index_teacher, codebook

@torch.no_grad()
def decode_teacher_quant(vq_obj, quant):
    """
    quant: (B, Cq, Hq, Wq)
    """
    # VQCodec wrapper
    if hasattr(vq_obj, "vq"):
        x_hat = vq_obj.vq.decode(quant).clamp(-1, 1)
        return x_hat

    # raw VQModel
    x_hat = vq_obj.decode(quant).clamp(-1, 1)
    return x_hat


# ============================================================
# Student: VAE-Encoder-style head
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        g = 8 if ch >= 8 else 1
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(g, ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(g, ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class VAEEncoderStyleVQHead(nn.Module):
    """
    Reuse SD1.5 VAE encoder blocks as backbone (same family structure),
    then map features to VQ codebook logits.

    For 512 input:
      VAE encoder features are at 64x64 (downsample x8),
      then one extra stride-2 projection gives 32x32 token map.
    """
    def __init__(
        self,
        codebook_size: int,
        sd_model_id: str = SD_MODEL_ID,
        proj_ch: int = 256,
        freeze_backbone: bool = True,
        train_last_n_downblocks: int = 0,
        train_mid_block: bool = False,
    ):
        super().__init__()
        vae = AutoencoderKL.from_pretrained(
            sd_model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        enc = vae.encoder

        # backbone (VAE encoder family)
        self.conv_in = enc.conv_in
        self.down_blocks = enc.down_blocks
        self.mid_block = enc.mid_block
        self.conv_norm_out = enc.conv_norm_out
        self.conv_act = enc.conv_act

        # channel count after conv_norm_out in SD1.5 VAE encoder is typically 512
        feat_ch = int(getattr(self.conv_norm_out, "num_channels", 512))
        g = 8 if int(proj_ch) >= 8 else 1
        self.to_token = nn.Sequential(
            nn.Conv2d(feat_ch, int(proj_ch), 3, padding=1),
            nn.GroupNorm(g, int(proj_ch)),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(proj_ch), int(proj_ch), 4, 2, 1),  # /8 -> /16 spatial
            nn.GroupNorm(g, int(proj_ch)),
            nn.SiLU(inplace=True),
            ResBlock(int(proj_ch)),
            ResBlock(int(proj_ch)),
        )
        self.out = nn.Conv2d(int(proj_ch), int(codebook_size), 1)

        if bool(freeze_backbone):
            for p in self.conv_in.parameters():
                p.requires_grad = False
            for b in self.down_blocks:
                for p in b.parameters():
                    p.requires_grad = False
            for p in self.mid_block.parameters():
                p.requires_grad = False
            for p in self.conv_norm_out.parameters():
                p.requires_grad = False

            n = int(max(0, train_last_n_downblocks))
            if n > 0:
                for b in self.down_blocks[-n:]:
                    for p in b.parameters():
                        p.requires_grad = True
            if bool(train_mid_block):
                for p in self.mid_block.parameters():
                    p.requires_grad = True

        del vae

    def forward(self, x):
        h = self.conv_in(x)
        for down in self.down_blocks:
            h = down(h)
        h = self.mid_block(h)
        h = self.conv_act(self.conv_norm_out(h))
        h = self.to_token(h)
        logits = self.out(h)
        return logits


# ============================================================
# Soft lookup
# ============================================================

def soft_lookup(logits, codebook, tau=1.0):
    """
    logits:   (B, K, H, W)
    codebook: (K, C)
    returns:
      probs:  (B, K, H, W)
      q_soft: (B, C, H, W)
    """
    probs = F.softmax(logits / tau, dim=1)
    q_soft = torch.einsum("bkhw,kc->bchw", probs, codebook)
    return probs, q_soft


# ============================================================
# Loss
# ============================================================

def compute_softvq_loss(
    q_soft,
    q_teacher,
    x_hat,
    x,
    logits=None,
    index_teacher=None,
    w_q=1.0,
    w_img=1.0,
    w_grad=0.25,
    w_ce=0.05,
):
    loss_q = F.l1_loss(q_soft, q_teacher)
    loss_img = F.l1_loss(x_hat, x)
    loss_g = grad_loss(x_hat, x)

    loss = w_q * loss_q + w_img * loss_img + w_grad * loss_g

    loss_ce = torch.tensor(0.0, device=x.device)
    if logits is not None and index_teacher is not None:
        loss_ce = F.cross_entropy(logits, index_teacher)
        loss = loss + w_ce * loss_ce

    metrics = {
        "loss": loss.item(),
        "loss_q": loss_q.item(),
        "loss_img": loss_img.item(),
        "loss_grad": loss_g.item(),
        "loss_ce": loss_ce.item(),
    }
    return loss, metrics


# ============================================================
# Visualization
# ============================================================

@torch.no_grad()
def save_visuals(x, x_hat, paths, save_dir, epoch, step, max_save=4):
    os.makedirs(save_dir, exist_ok=True)

    x01 = ((x[:max_save] + 1) * 0.5).clamp(0, 1).cpu()
    xhat01 = ((x_hat[:max_save] + 1) * 0.5).clamp(0, 1).cpu()

    for i in range(min(max_save, x.size(0))):
        stem = Path(paths[i]).stem
        row = torch.cat([x01[i:i+1], xhat01[i:i+1]], dim=3)
        save_path = os.path.join(save_dir, f"epoch_{epoch:03d}_step_{step:04d}_{stem}.png")
        save_image(row, save_path)


# ============================================================
# Train / Eval
# ============================================================

def train_one_epoch(vq_model, student_head, loader, optimizer, device, epoch, tau=1.0):
    student_head.train()
    if hasattr(vq_model, "eval"):
        vq_model.eval()
    elif hasattr(vq_model, "vq") and hasattr(vq_model.vq, "eval"):
        vq_model.vq.eval()

    running = {
        "loss": 0.0,
        "loss_q": 0.0,
        "loss_img": 0.0,
        "loss_grad": 0.0,
        "loss_ce": 0.0,
        "psnr": 0.0,
    }

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for step, (x, _) in enumerate(pbar):
        x = x.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            q_teacher, index_teacher, codebook = get_teacher_outputs(vq_model, x)

        logits = student_head(x)
        _, q_soft = soft_lookup(logits, codebook, tau=tau)
        x_hat = decode_teacher_quant(vq_model, q_soft)

        loss, metrics = compute_softvq_loss(
            q_soft=q_soft,
            q_teacher=q_teacher.float(),
            x_hat=x_hat.float(),
            x=x.float(),
            logits=logits,
            index_teacher=index_teacher,
            w_q=1.0,
            w_img=1.0,
            w_grad=0.25,
            w_ce=0.05,
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x01 = ((x + 1) * 0.5).clamp(0, 1)
            xhat01 = ((x_hat + 1) * 0.5).clamp(0, 1)
            metrics["psnr"] = psnr_torch(x01, xhat01)

        for k in running:
            running[k] += metrics[k]

        denom = step + 1
        pbar.set_postfix_str(
            f"loss={running['loss']/denom:.4f} | "
            f"q={running['loss_q']/denom:.4f} | "
            f"img={running['loss_img']/denom:.4f} | "
            f"grad={running['loss_grad']/denom:.4f} | "
            f"ce={running['loss_ce']/denom:.4f} | "
            f"psnr={running['psnr']/denom:.2f}"
        )

    for k in running:
        running[k] /= len(loader)
    return running


@torch.no_grad()
def eval_one_epoch(vq_model, student_head, loader, device, epoch, visual_dir="softvq_visuals", num_visual_batches=5, tau=1.0):
    student_head.eval()
    if hasattr(vq_model, "eval"):
        vq_model.eval()
    elif hasattr(vq_model, "vq") and hasattr(vq_model.vq, "eval"):
        vq_model.vq.eval()

    running = {
        "loss": 0.0,
        "loss_q": 0.0,
        "loss_img": 0.0,
        "loss_grad": 0.0,
        "loss_ce": 0.0,
        "psnr": 0.0,
    }

    saved_batches = 0
    pbar = tqdm(loader, desc=f"Val Epoch {epoch}")
    for step, (x, paths) in enumerate(pbar):
        x = x.to(device)

        q_teacher, index_teacher, codebook = get_teacher_outputs(vq_model, x)
        logits = student_head(x)
        _, q_soft = soft_lookup(logits, codebook, tau=tau)
        x_hat = decode_teacher_quant(vq_model, q_soft)

        _, metrics = compute_softvq_loss(
            q_soft=q_soft,
            q_teacher=q_teacher.float(),
            x_hat=x_hat.float(),
            x=x.float(),
            logits=logits,
            index_teacher=index_teacher,
            w_q=1.0,
            w_img=1.0,
            w_grad=0.25,
            w_ce=0.05,
        )

        with torch.no_grad():
            x01 = ((x + 1) * 0.5).clamp(0, 1)
            xhat01 = ((x_hat + 1) * 0.5).clamp(0, 1)
            metrics["psnr"] = psnr_torch(x01, xhat01)

        for k in running:
            running[k] += metrics[k]

        if saved_batches < num_visual_batches:
            save_visuals(
                x=x,
                x_hat=x_hat,
                paths=paths,
                save_dir=os.path.join(visual_dir, f"epoch_{epoch:03d}"),
                epoch=epoch,
                step=step,
                max_save=4,
            )
            saved_batches += 1

    for k in running:
        running[k] /= len(loader)
    return running


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Distill a VAE-encoder-style VQ projection head.")
    parser.add_argument(
        "--distill-preset",
        type=str,
        default=DISTILL_PRESET,
        help=f"One of: {', '.join(DISTILL_PRESETS.keys())}",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print(
        f"[Student] VAE-encoder-style head | freeze_backbone={FREEZE_VAE_BACKBONE} | "
        f"train_last_n_downblocks={TRAIN_LAST_N_DOWNBLOCKS} | train_mid_block={TRAIN_MID_BLOCK} | proj_ch={HEAD_PROJ_CH}"
    )

    distill_preset = resolve_distill_preset_name(args.distill_preset)
    preset = DISTILL_PRESETS[distill_preset]

    img_root = str(preset.get("img_root", IMG_ROOT))
    val_root = preset.get("val_root", VAL_IMG_ROOT)
    if val_root is not None:
        val_root = str(val_root)
    vq_ckpt_path = str(preset.get("vq_ckpt_path", VQ_CKPT_PATH))
    out_ckpt_dir = str(resolve_repo_output_path(preset.get("out_ckpt_dir", OUT_CKPT_DIR)))
    out_vis_dir = str(resolve_repo_output_path(preset.get("out_vis_dir", OUT_VIS_DIR)))
    img_size = int(preset.get("img_size", 512))
    batch_size = int(preset.get("batch_size", 8))
    epochs = int(preset.get("epochs", 20))
    lr = float(preset.get("lr", 1e-4))
    train_n = int(preset.get("train_n", 6000))

    print(
        f"[Preset] {distill_preset} | img_root={img_root} | val_root={val_root} | "
        f"vq_ckpt={vq_ckpt_path} | img_size={img_size} | bs={batch_size} | epochs={epochs} | lr={lr}"
    )

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_set, val_set = build_train_val_sets(
        root_dir=img_root,
        transform=transform,
        train_n=train_n,
        limit=100000,
        val_root=val_root,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # teacher
    vq = load_vqgan_taming(
        VQ_CONFIG_PATH,
        vq_ckpt_path,
        device=device,
    )
    codebook_size = int(vq.quantize.n_e) if hasattr(vq, "quantize") else 16384
    vq_model = VQCodec(vq_model=vq, codebook_size=codebook_size)

    # probe teacher
    with torch.no_grad():
        x0, _ = next(iter(train_loader))
        x0 = x0.to(device)
        q_teacher, index_teacher, codebook = get_teacher_outputs(vq_model, x0[:1])

        K = codebook.shape[0]
        print("Teacher quant shape:", tuple(q_teacher.shape))
        print("Teacher index shape:", tuple(index_teacher.shape))
        print("Codebook shape:", tuple(codebook.shape))

    student_head = VAEEncoderStyleVQHead(
        codebook_size=codebook_size,
        sd_model_id=SD_MODEL_ID,
        proj_ch=int(HEAD_PROJ_CH),
        freeze_backbone=bool(FREEZE_VAE_BACKBONE),
        train_last_n_downblocks=int(TRAIN_LAST_N_DOWNBLOCKS),
        train_mid_block=bool(TRAIN_MID_BLOCK),
    ).to(device)
    n_train = sum(p.numel() for p in student_head.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in student_head.parameters())
    print(f"[Student] params trainable={n_train}, total={n_all}")
    trainable_params = [p for p in student_head.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable params in student head. Check freeze/train_last_n_downblocks settings.")
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    os.makedirs(out_ckpt_dir, exist_ok=True)
    best_val = 1e9

    for epoch in range(1, epochs + 1):
        tau = max(0.3, 1.0 * (0.95 ** (epoch - 1)))

        train_stats = train_one_epoch(
            vq_model=vq_model,
            student_head=student_head,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            tau=tau,
        )

        val_stats = eval_one_epoch(
            vq_model=vq_model,
            student_head=student_head,
            loader=val_loader,
            device=device,
            epoch=epoch,
            visual_dir=out_vis_dir,
            num_visual_batches=2,
            tau=tau,
        )

        print(
            f"[Epoch {epoch}] "
            f"train loss={train_stats['loss']:.4f}, psnr={train_stats['psnr']:.2f} || "
            f"val loss={val_stats['loss']:.4f}, psnr={val_stats['psnr']:.2f} || "
            f"tau={tau:.3f}"
        )

        ckpt = {
            "student_head": student_head.state_dict(),
            "epoch": epoch,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "student_type": "VAEEncoderStyleVQHead",
            "sd_model_id": SD_MODEL_ID,
            "freeze_backbone": bool(FREEZE_VAE_BACKBONE),
            "train_last_n_downblocks": int(TRAIN_LAST_N_DOWNBLOCKS),
            "train_mid_block": bool(TRAIN_MID_BLOCK),
            "head_proj_ch": int(HEAD_PROJ_CH),
            "distill_preset": str(distill_preset),
            "img_root": img_root,
            "val_root": val_root,
            "teacher_vq_ckpt": vq_ckpt_path,
            "img_size": int(img_size),
        }
        # torch.save(ckpt, os.path.join(OUT_CKPT_DIR, "latest.pt"))

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(ckpt, os.path.join(out_ckpt_dir, "best.pt"))


if __name__ == "__main__":
    main()
