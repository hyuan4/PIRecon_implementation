import csv
import math
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

TAMING_ROOT = "taming-transformers"
if TAMING_ROOT not in sys.path:
    sys.path.insert(0, TAMING_ROOT)

from taming.models.vqgan import VQModel


# Runtime-shared config defaults (overridden by training scripts)
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_RANK = 8
LORA_ALPHA = 8.0
LORA_DECODER_LAST_N_UPBLOCKS = 2
LORA_ENCODER_FIRST_N_DOWNBLOCKS = 2
LORA_INCLUDE_CONV_IO = True
AUG_RESIZE_P = 0.7
AUG_RESIZE_MIN = 0.85
AUG_BLUR_P = 0.4
AUG_NOISE_P = 0.4
AUG_NOISE_STD = 0.01
NUM_LATENT_CLAMP = 20.0
NUM_QIM_INDEX_CLAMP = 4096.0
QIM_LOGIT_CLAMP = 20.0
MSB_MANUAL_SCHEDULE = []
PAYLOAD_MSB_BITS = 14
MSB_STAGE_START = 7
MSB_STAGE_STEP = 2
MSB_STAGE_EPOCHS = 4
MSB_STAGE_EPOCHS_MAP = {}
CSV_IMAGE_COLUMNS = ("PATIENT_ID", "image", "path", "filepath", "file", "img", "img_path")
REPO_ROOT = Path(__file__).resolve().parent


def set_conditioner_util_config(**kwargs):
    """Set runtime constants used by shared helpers."""
    globals().update(kwargs)

def safe_torch_load(path, map_location="cpu"):
    """
    Prefer weights_only=True to avoid torch.load FutureWarning and reduce pickle surface.
    Fallback keeps compatibility with older torch versions.
    """
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


def resolve_repo_local_path(path_like, repo_root: Path = REPO_ROOT) -> Path:
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


def _resolve_cache_root(cache_root, repo_root: Path = REPO_ROOT) -> Path:
    raw = str(cache_root).strip()
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return repo_root / p


def collect_input_images_from_csv(csv_path, exts=(".png", ".jpg", ".jpeg"), limit=100000):
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
            img_path = resolve_repo_local_path(raw)
            if img_path.suffix.lower() not in exts_set:
                continue
            files.append(img_path)
            if len(files) >= int(limit):
                break
    return files


def collect_input_images(input_root, exts=(".png", ".jpg", ".jpeg"), limit=100000):
    src = resolve_repo_local_path(input_root)
    exts_set = {str(x).lower() for x in exts}
    if src.is_file():
        if src.suffix.lower() == ".csv":
            return collect_input_images_from_csv(src, exts=exts, limit=limit)
        if src.suffix.lower() in exts_set:
            return [src]
        raise ValueError(f"Unsupported input file: {src}")
    if src.is_dir():
        files = []
        for p in sorted(src.iterdir())[:limit]:
            if p.is_file() and p.suffix.lower() in exts_set:
                files.append(p.resolve())
        return files
    raise FileNotFoundError(f"Input path does not exist: {src}")


def collect_cached_image_files(img_root, cache_root, exts=(".png", ".jpg", ".jpeg"), limit=100000):
    files = []
    cache_root = _resolve_cache_root(cache_root)
    for p in collect_input_images(img_root, exts=exts, limit=limit):
        stem = p.stem
        latent_path = cache_root / "latents" / f"{stem}.pt"
        image_path = cache_root / "images" / f"{stem}.pt"
        if latent_path.exists() and image_path.exists():
            files.append(p)
    return files

class LoRAConv2d(nn.Module):
    """
    Conv2d LoRA adapter:
      y = base(x) + scale * up(down(x))
    """
    def __init__(self, base_conv: nn.Conv2d, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        if int(rank) <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.down = nn.Conv2d(
            in_channels=base_conv.in_channels,
            out_channels=self.rank,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.up = nn.Conv2d(
            in_channels=self.rank,
            out_channels=base_conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        ref_dev = base_conv.weight.device
        ref_dtype = base_conv.weight.dtype
        self.down.to(device=ref_dev, dtype=ref_dtype)
        self.up.to(device=ref_dev, dtype=ref_dtype)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale

def _resolve_child(module: nn.Module, name: str):
    if name.isdigit():
        return module[int(name)]
    return getattr(module, name)

def _set_child(module: nn.Module, name: str, child: nn.Module):
    if name.isdigit():
        module[int(name)] = child
    else:
        setattr(module, name, child)

def _get_parent_and_key(root: nn.Module, full_name: str):
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = _resolve_child(parent, p)
    return parent, parts[-1]

def inject_vae_lora(sd_vae):
    """
    Inject LoRA into selected encoder/decoder conv layers.
    We train encoder+decoder to preserve injected payload through roundtrip.
    """
    lora_modules = {}

    # decoder: last N up-blocks + optional conv_out
    dec = sd_vae.decoder
    dec_up = getattr(dec, "up_blocks", [])
    dec_start = max(0, len(dec_up) - int(LORA_DECODER_LAST_N_UPBLOCKS))
    for name, mod in list(dec.named_modules()):
        if not isinstance(mod, nn.Conv2d):
            continue
        use = False
        if name.startswith("up_blocks."):
            try:
                bid = int(name.split(".")[1])
            except Exception:
                bid = -1
            if bid >= dec_start and (
                name.endswith(".conv1") or name.endswith(".conv2") or name.endswith(".conv_shortcut")
            ):
                use = True
        elif bool(LORA_INCLUDE_CONV_IO) and name == "conv_out":
            use = True
        if not use:
            continue
        parent, key = _get_parent_and_key(dec, name)
        wrapped = LoRAConv2d(mod, rank=int(LORA_RANK), alpha=float(LORA_ALPHA))
        _set_child(parent, key, wrapped)
        lora_modules[f"decoder.{name}"] = wrapped

    # encoder: first N down-blocks + optional conv_in
    enc = sd_vae.encoder
    for name, mod in list(enc.named_modules()):
        if not isinstance(mod, nn.Conv2d):
            continue
        use = False
        if name.startswith("down_blocks."):
            try:
                bid = int(name.split(".")[1])
            except Exception:
                bid = 1_000_000
            if bid < int(LORA_ENCODER_FIRST_N_DOWNBLOCKS) and (
                name.endswith(".conv1") or name.endswith(".conv2") or name.endswith(".conv_shortcut")
            ):
                use = True
        elif bool(LORA_INCLUDE_CONV_IO) and name == "conv_in":
            use = True
        if not use:
            continue
        parent, key = _get_parent_and_key(enc, name)
        wrapped = LoRAConv2d(mod, rank=int(LORA_RANK), alpha=float(LORA_ALPHA))
        _set_child(parent, key, wrapped)
        lora_modules[f"encoder.{name}"] = wrapped

    return lora_modules

def dump_lora_state(lora_modules: dict):
    return {name: m.state_dict() for name, m in lora_modules.items()}

def load_lora_state(lora_modules: dict, ckpt_obj: dict):
    blob = ckpt_obj.get("vae_lora", None)
    if not isinstance(blob, dict):
        return
    for name, module in lora_modules.items():
        if name in blob:
            module.load_state_dict(blob[name], strict=False)

def load_writer_reader_state(writer: nn.Module, reader: nn.Module, ckpt_obj: dict):
    def _partial_load(module: nn.Module, blob: dict, tag: str):
        cur = module.state_dict()
        keep = {}
        dropped = 0
        for k, v in blob.items():
            if k in cur and torch.is_tensor(v) and tuple(v.shape) == tuple(cur[k].shape):
                keep[k] = v
            else:
                dropped += 1
        missing, unexpected = module.load_state_dict(keep, strict=False)
        if dropped > 0:
            print(
                f"[Resume:{tag}] partial load due to shape mismatch: loaded={len(keep)}, "
                f"dropped={dropped}, missing={len(missing)}, unexpected={len(unexpected)}"
            )

    if writer is not None and isinstance(ckpt_obj.get("writer", None), dict):
        _partial_load(writer, ckpt_obj["writer"], "W")
    if reader is not None and isinstance(ckpt_obj.get("reader", None), dict):
        _partial_load(reader, ckpt_obj["reader"], "R")

def _extract_unet_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "unet" in ckpt_obj and isinstance(ckpt_obj["unet"], dict):
            state = ckpt_obj["unet"]
        elif "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            state = ckpt_obj["state_dict"]
        else:
            state = ckpt_obj
    else:
        raise RuntimeError("Unsupported UNet checkpoint format.")

    out = {}
    for k, v in state.items():
        nk = str(k)
        for pfx in ("unet.", "model.diffusion_model.", "module.unet.", "module.model.diffusion_model."):
            if nk.startswith(pfx):
                nk = nk[len(pfx):]
                break
        out[nk] = v
    return out

class CachedMessageDataset(Dataset):
    def __init__(
        self,
        img_root,
        cache_root,
        transform=None,
        exts=(".png", ".jpg", ".jpeg"),
        limit=100000,
        files_list=None,
    ):
        super().__init__()
        self.img_root = resolve_repo_local_path(img_root)
        self.cache_root = _resolve_cache_root(cache_root)
        self.transform = transform
        self.exts = {e.lower() for e in exts}
        self.files = []

        if files_list is not None:
            self.files = files_list
        else:
            self.files = collect_cached_image_files(
                img_root=self.img_root,
                cache_root=self.cache_root,
                exts=tuple(self.exts),
                limit=limit,
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = Path(self.files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        stem = img_path.stem
        latent_path = self.cache_root / "latents" / f"{stem}.pt"
        image_path = self.cache_root / "images" / f"{stem}.pt"

        cache_lat = safe_torch_load(latent_path, map_location="cpu")
        cache_img = safe_torch_load(image_path, map_location="cpu")

        z_clean = cache_lat["z_clean"].squeeze(0)   # (4,H,W)
        y_clean = cache_img["y_clean"].squeeze(0)   # (3,H,W)

        return img, z_clean, y_clean, str(img_path)

def build_train_val_sets(img_root, cache_root, transform, train_n=6000, limit=100000, val_root=None):
    img_root_resolved = resolve_repo_local_path(img_root)
    cache_root_resolved = _resolve_cache_root(cache_root)
    files = collect_cached_image_files(
        img_root=img_root_resolved,
        cache_root=cache_root_resolved,
        limit=limit,
    )

    if val_root:
        train_files = files[:train_n] if int(train_n) > 0 else files
        val_files = collect_cached_image_files(
            img_root=val_root,
            cache_root=cache_root_resolved,
            limit=limit,
        )
    else:
        train_files = files[:train_n]
        val_files = files[train_n:]

    train_set = CachedMessageDataset(
        img_root=img_root_resolved,
        cache_root=cache_root_resolved,
        transform=transform,
        files_list=train_files,
    )
    val_set = CachedMessageDataset(
        img_root=val_root or img_root_resolved,
        cache_root=cache_root_resolved,
        transform=transform,
        files_list=val_files,
    )
    return train_set, val_set

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

def load_vqgan_taming(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    params = config.model.params

    model = VQModel(**params)

    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Loaded VQ teacher.")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

class VQCodec:
    def __init__(self, vq_model, codebook_size: int):
        self.vq = vq_model
        self.codebook_size = int(codebook_size)

    @torch.no_grad()
    def encode_to_indices(self, x):
        out = self.vq.encode(x)

        if isinstance(out, (tuple, list)) and len(out) >= 3:
            quant, _, info = out[0], out[1], out[2]
            indices = info[2]
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
            if indices.numel() != B * Ht * Wt:
                raise RuntimeError(f"indices numel mismatch: got {indices.numel()}, expect {B*Ht*Wt}")
            indices = indices.view(B, Ht, Wt)

        elif indices.dim() == 2:
            if indices.size(0) != B:
                raise RuntimeError(f"indices batch mismatch: got {indices.size(0)}, expect {B}")
            if indices.size(1) != Ht * Wt:
                raise RuntimeError(f"indices length mismatch: got {indices.size(1)}, expect {Ht*Wt}")
            indices = indices.view(B, Ht, Wt)

        elif indices.dim() == 3:
            pass
        else:
            raise RuntimeError(f"Unsupported indices dim={indices.dim()}, shape={tuple(indices.shape)}")

        return indices

    @torch.no_grad()
    def decode_from_indices(self, indices):
        B, Ht, Wt = indices.shape
        e_dim = int(self.vq.quantize.e_dim)

        indices_hw = indices.to(torch.long).view(B, Ht, Wt)

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

@torch.no_grad()
def decode_teacher_quant(vq_codec, quant):
    return vq_codec.vq.decode(quant).clamp(-1, 1)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        g = 8 if out_ch >= 8 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

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

class StudentSoftVQHead(nn.Module):
    """
    image -> logits over codebook
    output spatial size assumed 32x32 for 512 input
    """
    def __init__(self, codebook_size, base=64):
        super().__init__()
        self.stem = ConvBlock(3, base)                    # 512

        self.d1 = nn.Conv2d(base, base * 2, 4, 2, 1)     # 256
        self.b1 = nn.Sequential(ResBlock(base * 2), ResBlock(base * 2))

        self.d2 = nn.Conv2d(base * 2, base * 4, 4, 2, 1) # 128
        self.b2 = nn.Sequential(ResBlock(base * 4), ResBlock(base * 4))

        self.d3 = nn.Conv2d(base * 4, base * 8, 4, 2, 1) # 64
        self.b3 = nn.Sequential(ResBlock(base * 8), ResBlock(base * 8))

        self.d4 = nn.Conv2d(base * 8, base * 8, 4, 2, 1) # 32
        self.b4 = nn.Sequential(ResBlock(base * 8), ResBlock(base * 8))

        self.out = nn.Conv2d(base * 8, codebook_size, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(self.d1(x))
        x = self.b2(self.d2(x))
        x = self.b3(self.d3(x))
        x = self.b4(self.d4(x))
        logits = self.out(x)
        return logits

class VAEEncoderStyleVQHead(nn.Module):
    """
    Reuse SD1.5 VAE encoder trunk (partial exact structure) and
    project to VQ codebook logits at token resolution.
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

        self.conv_in = enc.conv_in
        self.down_blocks = enc.down_blocks
        self.mid_block = enc.mid_block
        self.conv_norm_out = enc.conv_norm_out
        self.conv_act = enc.conv_act

        feat_ch = int(getattr(self.conv_norm_out, "num_channels", 512))
        g = 8 if int(proj_ch) >= 8 else 1
        self.to_token = nn.Sequential(
            nn.Conv2d(feat_ch, int(proj_ch), 3, padding=1),
            nn.GroupNorm(g, int(proj_ch)),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(proj_ch), int(proj_ch), 4, 2, 1),  # /8 -> /16
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

def soft_lookup(logits, codebook, tau=1.0):
    probs = F.softmax(logits / tau, dim=1)
    q_soft = torch.einsum("bkhw,kc->bchw", probs, codebook)
    return probs, q_soft

def _radial_rfft_mask(h, w, r_low, r_high, device, dtype):
    fy = torch.fft.fftfreq(h, d=1.0, device=device)  # [-0.5, 0.5)
    fx = torch.fft.rfftfreq(w, d=1.0, device=device)  # [0, 0.5]
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    rr = torch.sqrt(xx * xx + yy * yy) / math.sqrt(0.5 ** 2 + 0.5 ** 2)
    mask = ((rr >= float(r_low)) & (rr <= float(r_high))).to(dtype)
    return mask.view(1, 1, h, w // 2 + 1)

def apply_fft_bandpass(delta, r_low=0.18, r_high=0.70):
    h, w = int(delta.shape[-2]), int(delta.shape[-1])
    mask = _radial_rfft_mask(
        h=h,
        w=w,
        r_low=r_low,
        r_high=r_high,
        device=delta.device,
        dtype=delta.dtype,
    )
    f = torch.fft.rfft2(delta, dim=(-2, -1))
    f = f * mask
    y = torch.fft.irfft2(f, s=(h, w), dim=(-2, -1))

    # Keep perturbation scale stable after masking.
    rms_in = delta.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
    rms_out = y.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
    scale = (rms_in / (rms_out + 1e-6)).clamp(0.0, 3.0)
    return y * scale

def apply_roundtrip_aug(y):
    x = y
    h, w = int(x.shape[-2]), int(x.shape[-1])

    if float(torch.rand(1, device=x.device).item()) < float(AUG_RESIZE_P):
        factor = float(torch.empty(1, device=x.device).uniform_(AUG_RESIZE_MIN, 1.0).item())
        hh = max(16, int(round(h * factor)))
        ww = max(16, int(round(w * factor)))
        x = F.interpolate(x, size=(hh, ww), mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    if float(torch.rand(1, device=x.device).item()) < float(AUG_BLUR_P):
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    if float(torch.rand(1, device=x.device).item()) < float(AUG_NOISE_P):
        x = x + torch.randn_like(x) * float(AUG_NOISE_STD)

    return x.clamp(-1, 1)

def _ste_round(x):
    return x + (x.round() - x).detach()

def apply_qim_hybrid_latent(z, bits, delta=0.02, r_low=0.18, r_high=0.70):
    """
    QIM-hybrid on latent rFFT real part using message bits as dither.
    z:    (B,C,H,W)
    bits: (B,Ab,Hm,Wm) in {0,1}
    """
    if bits is None or bits.numel() == 0:
        return z

    bsz, ch, h, w = z.shape
    wf = w // 2 + 1

    mask = _radial_rfft_mask(
        h=h,
        w=w,
        r_low=r_low,
        r_high=r_high,
        device=z.device,
        dtype=torch.float32,
    )[0, 0].to(torch.bool)  # (H, Wf)

    npos = int(mask.sum().item())
    if npos <= 0:
        return z

    bits_flat = bits.reshape(bsz, -1)
    if bits_flat.size(1) <= 0:
        return z

    rep = int(math.ceil(float(npos) / float(bits_flat.size(1))))
    bits_pos = bits_flat.repeat(1, rep)[:, :npos]  # (B, npos)

    f = torch.fft.rfft2(z, dim=(-2, -1))
    re = f.real.reshape(bsz, ch, -1)   # (B,C,H*Wf)
    im = f.imag

    mask_flat = mask.reshape(-1)
    vals = re[:, :, mask_flat]  # (B,C,npos)

    bits_bc = bits_pos[:, None, :].expand(bsz, ch, npos).to(vals.dtype)
    dither = 0.5 * float(delta) * bits_bc
    vals_q = _ste_round((vals - dither) / float(delta)) * float(delta) + dither

    re[:, :, mask_flat] = vals_q
    re = re.view(bsz, ch, h, wf)

    f_q = torch.complex(re, im)
    z_q = torch.fft.irfft2(f_q, s=(h, w), dim=(-2, -1))
    return z_q

def indices_to_msb_bit_map(indices, codebook_size, msb_bits=4, stride=1, out_channels=None):
    """
    indices: (B,H,W) int64
    return:
      bit_map: (B, msb_bits, Hs, Ws) in {0,1} float
    """
    if indices.dim() != 3:
        raise RuntimeError(f"indices should be (B,H,W), got {tuple(indices.shape)}")

    bits_per = int(math.ceil(math.log2(int(codebook_size))))
    msb_bits = int(msb_bits)
    if msb_bits < 1 or msb_bits > bits_per:
        raise ValueError(f"msb_bits must be in [1,{bits_per}], got {msb_bits}")

    if int(stride) > 1:
        indices = indices[:, ::int(stride), ::int(stride)]

    indices = indices.to(torch.int64)
    shifts = torch.arange(bits_per - 1, bits_per - msb_bits - 1, -1, device=indices.device, dtype=torch.int64)
    bits = ((indices.unsqueeze(1) >> shifts.view(1, msb_bits, 1, 1)) & 1).to(torch.float32)
    if out_channels is not None:
        out_channels = int(out_channels)
        if out_channels < msb_bits:
            raise ValueError(f"out_channels({out_channels}) must be >= msb_bits({msb_bits})")
        if out_channels > msb_bits:
            pad = torch.zeros(
                (bits.size(0), out_channels - msb_bits, bits.size(2), bits.size(3)),
                dtype=bits.dtype,
                device=bits.device,
            )
            bits = torch.cat([bits, pad], dim=1)
    return bits

def msb_bit_logits_to_indices(bit_logits, codebook_size, target_hw, msb_bits=4, stride=1):
    """
    bit_logits: (B, msb_bits, Hs, Ws), logits from reader
    return:
      indices_hat: (B, H, W) int64
    """
    if bit_logits.dim() != 4:
        raise RuntimeError(f"bit_logits should be (B,C,H,W), got {tuple(bit_logits.shape)}")

    bits_per = int(math.ceil(math.log2(int(codebook_size))))
    msb_bits = int(msb_bits)
    if bit_logits.size(1) < msb_bits:
        raise RuntimeError(f"bit channel mismatch: got {bit_logits.size(1)}, need >= {msb_bits}")

    bits_hat = (torch.sigmoid(bit_logits[:, :msb_bits]) >= 0.5).to(torch.int64)
    values = torch.zeros(
        (bit_logits.size(0), bit_logits.size(2), bit_logits.size(3)),
        dtype=torch.int64,
        device=bit_logits.device,
    )
    for i in range(msb_bits):
        shift = bits_per - 1 - i
        values = values | (bits_hat[:, i] << shift)

    if int(stride) > 1:
        values = F.interpolate(values.unsqueeze(1).float(), size=target_hw, mode="nearest").squeeze(1).to(torch.int64)

    values = values.clamp_(0, int(codebook_size) - 1)
    return values

def indices_to_msb_bitstream(indices: torch.Tensor, codebook_size: int, msb_bits: int):
    """
    indices: (B,Ht,Wt) int64
    return:
      bits: (B, Nt*msb_bits) uint8
      bits_per_token: full bits-per-token
    """
    if indices.dim() != 3:
        raise RuntimeError(f"indices should be (B,H,W), got {tuple(indices.shape)}")
    bits_per = int(math.ceil(math.log2(int(codebook_size))))
    msb_bits = int(max(1, min(bits_per, int(msb_bits))))
    flat = indices.reshape(indices.size(0), -1).to(torch.int64)
    shifts = torch.arange(bits_per - 1, bits_per - msb_bits - 1, -1, device=flat.device, dtype=torch.int64)
    bits = ((flat.unsqueeze(-1) >> shifts.view(1, 1, msb_bits)) & 1).to(torch.uint8)
    return bits.reshape(indices.size(0), -1).contiguous(), bits_per

def msb_bitstream_to_indices(bits: torch.Tensor, codebook_size: int, target_hw, msb_bits: int):
    """
    bits: (B, Nt*msb_bits) in {0,1}
    return: indices_hat (B,Ht,Wt) int64
    """
    if bits.dim() != 2:
        raise RuntimeError(f"bits should be (B,L), got {tuple(bits.shape)}")
    bsz = bits.size(0)
    ht, wt = int(target_hw[0]), int(target_hw[1])
    nt = int(ht * wt)
    bits_per = int(math.ceil(math.log2(int(codebook_size))))
    msb_bits = int(max(1, min(bits_per, int(msb_bits))))
    need = nt * msb_bits

    if bits.size(1) < need:
        pad = torch.zeros((bsz, need - bits.size(1)), device=bits.device, dtype=bits.dtype)
        bits = torch.cat([bits, pad], dim=1)
    elif bits.size(1) > need:
        bits = bits[:, :need]

    bits = bits.view(bsz, nt, msb_bits).to(torch.int64)
    values = torch.zeros((bsz, nt), device=bits.device, dtype=torch.int64)
    for i in range(msb_bits):
        shift = bits_per - 1 - i
        values = values | (bits[:, :, i] << shift)
    values = values.view(bsz, ht, wt).clamp_(0, int(codebook_size) - 1)
    return values

def sample_qim_positions(H, W, C, nbits_total, seed=0, r_low=0.18, r_high=0.70, bits_per_pos=2):
    """
    Sample one-step disjoint positions per channel for stream QIM embedding.
    Capacity = C * n_pos * bits_per_pos.
    """
    Wr = W // 2 + 1
    yy = torch.arange(H, device="cpu", dtype=torch.float32)
    xx = torch.arange(Wr, device="cpu", dtype=torch.float32)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    cy = (H - 1) / 2.0
    R = torch.sqrt((Y - cy) ** 2 + X ** 2)
    R_max = (cy ** 2 + (Wr - 1) ** 2) ** 0.5
    R = R / (R_max + 1e-8)
    cand = ((R >= float(r_low)) & (R <= float(r_high))).reshape(-1).nonzero(as_tuple=False).squeeze(1)
    if cand.numel() <= 0:
        raise RuntimeError("No QIM candidate frequency positions found in selected band.")

    n_per_ch = int(math.ceil(float(max(1, int(nbits_total))) / float(max(1, int(C * bits_per_pos)))))
    if n_per_ch > int(cand.numel()):
        raise RuntimeError(
            f"QIM capacity insufficient: need n_per_ch={n_per_ch}, available={int(cand.numel())}, "
            f"nbits={nbits_total}, C={C}, bits_per_pos={bits_per_pos}, band=({r_low},{r_high})"
        )

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    pos_ch = []
    for _ in range(int(C)):
        p = torch.randperm(int(cand.numel()), generator=g)
        pos = cand[p[:n_per_ch]].clone()
        pos_ch.append(pos)

    capacity = int(C * n_per_ch * bits_per_pos)
    return pos_ch, capacity

def qim_embed_latent_rfft_stream(lat, bits, pos_ch, delta=0.014, bits_per_pos=2):
    """
    Stream QIM embedding in latent rFFT (single step).
    bits: (B,L) uint8/float in {0,1}
    """
    if int(bits_per_pos) != 2:
        raise ValueError("Current stream QIM implementation expects bits_per_pos=2")
    B, C, H, W = lat.shape
    d = float(delta)
    bits_u8 = bits.to(torch.uint8)
    total = int(bits_u8.size(1))

    lat_safe = torch.nan_to_num(lat, nan=0.0, posinf=float(NUM_LATENT_CLAMP), neginf=-float(NUM_LATENT_CLAMP))
    lat_safe = lat_safe.clamp(-float(NUM_LATENT_CLAMP), float(NUM_LATENT_CLAMP))
    lat_fft = torch.fft.rfft2(lat_safe.to(torch.float64), dim=(-2, -1))
    lat_mod = lat_fft.clone()
    ptr = 0
    for c in range(C):
        if c >= len(pos_ch) or ptr >= total:
            break
        pos = pos_ch[c].to(lat.device)
        if pos.numel() <= 0:
            continue
        Lpos = int(pos.numel())
        flat = lat_mod[:, c].reshape(B, -1)

        n_real = min(Lpos, total - ptr)
        if n_real > 0:
            p = pos[:n_real]
            b = bits_u8[:, ptr:ptr + n_real].to(torch.int64)
            ptr += n_real
            x = flat[:, p].real
            n0 = torch.round(x / d - 0.5).clamp(
                -float(NUM_QIM_INDEX_CLAMP), float(NUM_QIM_INDEX_CLAMP)
            ).to(torch.int64)
            nt = (n0 - (n0 & 1)) + b
            x_new = (nt.to(x.dtype) + 0.5) * d
            flat[:, p] = torch.complex(x_new, flat[:, p].imag)

        n_im = min(Lpos, total - ptr)
        if n_im > 0:
            p = pos[:n_im]
            b = bits_u8[:, ptr:ptr + n_im].to(torch.int64)
            ptr += n_im
            x = flat[:, p].imag
            n0 = torch.round(x / d - 0.5).clamp(
                -float(NUM_QIM_INDEX_CLAMP), float(NUM_QIM_INDEX_CLAMP)
            ).to(torch.int64)
            nt = (n0 - (n0 & 1)) + b
            x_new = (nt.to(x.dtype) + 0.5) * d
            flat[:, p] = torch.complex(flat[:, p].real, x_new)

    return torch.fft.irfft2(lat_mod, s=(H, W), dim=(-2, -1)).to(lat.dtype)

def qim_extract_latent_rfft_bits(lat, pos_ch, delta=0.014, bits_per_pos=2, nbits_needed=None):
    if int(bits_per_pos) != 2:
        raise ValueError("Current stream QIM implementation expects bits_per_pos=2")
    B, C, _, _ = lat.shape
    d = float(delta)
    need = None if nbits_needed is None else int(nbits_needed)

    lat_safe = torch.nan_to_num(lat, nan=0.0, posinf=float(NUM_LATENT_CLAMP), neginf=-float(NUM_LATENT_CLAMP))
    lat_safe = lat_safe.clamp(-float(NUM_LATENT_CLAMP), float(NUM_LATENT_CLAMP))
    lat_fft = torch.fft.rfft2(lat_safe.to(torch.float64), dim=(-2, -1))
    out = []
    got = 0
    for c in range(C):
        if c >= len(pos_ch):
            break
        if need is not None and got >= need:
            break
        pos = pos_ch[c].to(lat.device)
        if pos.numel() <= 0:
            continue
        flat = lat_fft[:, c].reshape(B, -1)
        Lpos = int(pos.numel())

        n_real = Lpos if need is None else min(Lpos, need - got)
        if n_real > 0:
            p = pos[:n_real]
            x = flat[:, p].real
            n_hat = torch.round(x / d - 0.5).clamp(
                -float(NUM_QIM_INDEX_CLAMP), float(NUM_QIM_INDEX_CLAMP)
            ).to(torch.int64)
            out.append((n_hat & 1).to(torch.uint8))
            got += n_real

        if need is not None and got >= need:
            break
        n_im = Lpos if need is None else min(Lpos, need - got)
        if n_im > 0:
            p = pos[:n_im]
            x = flat[:, p].imag
            n_hat = torch.round(x / d - 0.5).clamp(
                -float(NUM_QIM_INDEX_CLAMP), float(NUM_QIM_INDEX_CLAMP)
            ).to(torch.int64)
            out.append((n_hat & 1).to(torch.uint8))
            got += n_im

    if len(out) == 0:
        bits_hat = torch.zeros((B, 0), device=lat.device, dtype=torch.uint8)
    else:
        bits_hat = torch.cat(out, dim=1)
    if need is not None:
        bits_hat = bits_hat[:, :need]
    return bits_hat

def qim_extract_latent_rfft_logits(lat, pos_ch, delta=0.014, bits_per_pos=2, nbits_needed=None, logit_scale=6.0):
    """
    Differentiable proxy for QIM bit extraction.
    logit > 0 => bit=1.
    """
    if int(bits_per_pos) != 2:
        raise ValueError("Current stream QIM implementation expects bits_per_pos=2")
    B, C, _, _ = lat.shape
    d = float(delta)
    need = None if nbits_needed is None else int(nbits_needed)
    sc = float(logit_scale)

    lat_safe = torch.nan_to_num(lat, nan=0.0, posinf=float(NUM_LATENT_CLAMP), neginf=-float(NUM_LATENT_CLAMP))
    lat_safe = lat_safe.clamp(-float(NUM_LATENT_CLAMP), float(NUM_LATENT_CLAMP))
    lat_fft = torch.fft.rfft2(lat_safe, dim=(-2, -1))
    out = []
    got = 0
    for c in range(C):
        if c >= len(pos_ch):
            break
        if need is not None and got >= need:
            break
        pos = pos_ch[c].to(lat.device)
        if pos.numel() <= 0:
            continue
        flat = lat_fft[:, c].reshape(B, -1)
        Lpos = int(pos.numel())

        n_real = Lpos if need is None else min(Lpos, need - got)
        if n_real > 0:
            p = pos[:n_real]
            x = flat[:, p].real
            n_hat = _ste_round(x / d - 0.5).clamp(
                -float(NUM_QIM_INDEX_CLAMP), float(NUM_QIM_INDEX_CLAMP)
            )
            logits = (-sc * torch.cos(math.pi * n_hat)).clamp(
                -float(QIM_LOGIT_CLAMP), float(QIM_LOGIT_CLAMP)
            )
            out.append(logits.to(torch.float32))
            got += n_real

        if need is not None and got >= need:
            break
        n_im = Lpos if need is None else min(Lpos, need - got)
        if n_im > 0:
            p = pos[:n_im]
            x = flat[:, p].imag
            n_hat = _ste_round(x / d - 0.5).clamp(
                -float(NUM_QIM_INDEX_CLAMP), float(NUM_QIM_INDEX_CLAMP)
            )
            logits = (-sc * torch.cos(math.pi * n_hat)).clamp(
                -float(QIM_LOGIT_CLAMP), float(QIM_LOGIT_CLAMP)
            )
            out.append(logits.to(torch.float32))
            got += n_im

    if len(out) == 0:
        bit_logits = torch.zeros((B, 0), device=lat.device, dtype=torch.float32)
    else:
        bit_logits = torch.cat(out, dim=1)
    if need is not None:
        bit_logits = bit_logits[:, :need]
    return bit_logits

def infer_decoder_film_targets(sd_vae, last_n_upblocks=2):
    dec = sd_vae.decoder
    up_blocks = getattr(dec, "up_blocks", [])
    if len(up_blocks) == 0:
        raise RuntimeError("sd_vae.decoder.up_blocks is empty.")
    start = max(0, len(up_blocks) - int(last_n_upblocks))
    idxs = list(range(start, len(up_blocks)))
    chs = []
    for i in idxs:
        blk = up_blocks[i]
        c = None
        if hasattr(blk, "resnets") and len(blk.resnets) > 0:
            rn = blk.resnets[-1]
            if hasattr(rn, "conv2") and hasattr(rn.conv2, "out_channels"):
                c = int(rn.conv2.out_channels)
            elif hasattr(rn, "out_channels"):
                c = int(rn.out_channels)
        if c is None:
            raise RuntimeError(f"Cannot infer out channels for decoder up_block[{i}]")
        chs.append(c)
    return idxs, chs

class Writer(nn.Module):
    def __init__(
        self,
        m_ch,
        target_channels,
        z_ch=4,
        hidden=64,
        gamma_scale=0.25,
        beta_scale=0.25,
    ):
        super().__init__()
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)

        self.m_proj = nn.Sequential(
            nn.Conv2d(m_ch, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
        )
        self.z_proj = nn.Sequential(
            nn.Conv2d(z_ch, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResBlock(hidden * 2),
            ResBlock(hidden * 2),
            nn.Conv2d(hidden * 2, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
        )
        self.gamma_heads = nn.ModuleList([nn.Conv2d(hidden, int(c), 3, padding=1) for c in target_channels])
        self.beta_heads = nn.ModuleList([nn.Conv2d(hidden, int(c), 3, padding=1) for c in target_channels])

    def forward(self, m, z_clean):
        m_up = F.interpolate(m, size=z_clean.shape[-2:], mode="bilinear", align_corners=False)
        m_feat = self.m_proj(m_up)
        z_feat = self.z_proj(z_clean)
        h = torch.cat([m_feat, z_feat], dim=1)
        feat = self.body(h)
        gb = []
        for gh, bh in zip(self.gamma_heads, self.beta_heads):
            gamma = torch.tanh(gh(feat)) * self.gamma_scale
            beta = torch.tanh(bh(feat)) * self.beta_scale
            gb.append((gamma, beta))
        return gb

def decode_with_conditioner(sd_vae, z_in, film_pairs, target_block_idxs):
    """
    Inject FiLM modulation into decoder up_blocks via forward hooks.
    """
    dec = sd_vae.decoder
    hooks = []

    for bi, (gamma, beta) in zip(target_block_idxs, film_pairs):
        block = dec.up_blocks[int(bi)]

        def _make_hook(gm, bt):
            def _hook(_module, _inputs, output):
                x = output[0] if isinstance(output, tuple) else output
                gm_i = F.interpolate(gm.to(dtype=x.dtype), size=x.shape[-2:], mode="bilinear", align_corners=False)
                bt_i = F.interpolate(bt.to(dtype=x.dtype), size=x.shape[-2:], mode="bilinear", align_corners=False)
                x = x * (1.0 + gm_i) + bt_i
                if isinstance(output, tuple):
                    return (x,) + tuple(output[1:])
                return x
            return _hook

        hooks.append(block.register_forward_hook(_make_hook(gamma, beta)))

    try:
        y = sd_vae.decode(z_in).sample
    finally:
        for h in hooks:
            h.remove()
    return y

class Reader(nn.Module):
    def __init__(self, m_ch, out_hw=(32, 32), in_ch=4, hidden=64):
        super().__init__()
        self.out_hw = out_hw

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden, m_ch, 3, padding=1)

    def forward(self, z_rt):
        h = self.stem(z_rt)
        h = self.body(h)
        h = F.interpolate(h, size=self.out_hw, mode="bilinear", align_corners=False)
        m_hat = self.out(h)
        return m_hat

@torch.no_grad()
def save_roundtrip_visuals(x, y_clean, y_stego, x_rec, paths, save_dir, epoch, step, max_save=4):
    os.makedirs(save_dir, exist_ok=True)

    x01 = ((x[:max_save] + 1) * 0.5).clamp(0, 1).cpu()
    clean01 = ((y_clean[:max_save] + 1) * 0.5).clamp(0, 1).cpu()
    stego01 = ((y_stego[:max_save] + 1) * 0.5).clamp(0, 1).cpu()
    rec01 = ((x_rec[:max_save] + 1) * 0.5).clamp(0, 1).cpu()
    if rec01.shape[-2:] != x01.shape[-2:]:
        rec01 = F.interpolate(rec01, size=x01.shape[-2:], mode="bilinear", align_corners=False)

    for i in range(min(max_save, x.size(0))):
        stem = Path(paths[i]).stem
        row = torch.cat([x01[i:i+1], clean01[i:i+1], stego01[i:i+1], rec01[i:i+1]], dim=3)
        save_path = os.path.join(save_dir, f"epoch_{epoch:03d}_step_{step:04d}_{stem}.png")
        save_image(row, save_path)

@torch.no_grad()
def get_frozen_message(proj_head, codebook_size, x, msb_bits=4, stride=1):
    logits = proj_head(x)
    indices = torch.argmax(logits, dim=1)
    m = indices_to_msb_bit_map(
        indices=indices,
        codebook_size=codebook_size,
        msb_bits=msb_bits,
        stride=stride,
    )
    return m

def build_msb_schedule(bits_per_token: int):
    bits_per_token = int(bits_per_token)

    if len(MSB_MANUAL_SCHEDULE) > 0:
        vals = []
        for v in MSB_MANUAL_SCHEDULE:
            vv = max(1, min(bits_per_token, int(v)))
            vals.append(vv)
        vals = sorted(set(vals))
        if vals[-1] != bits_per_token:
            vals.append(bits_per_token)
        return vals

    if int(PAYLOAD_MSB_BITS) > 0:
        return [max(1, min(bits_per_token, int(PAYLOAD_MSB_BITS)))]

    start = max(1, min(bits_per_token, int(MSB_STAGE_START)))
    step = max(1, int(MSB_STAGE_STEP))
    vals = list(range(start, bits_per_token + 1, step))
    if len(vals) == 0:
        vals = [bits_per_token]
    if vals[-1] != bits_per_token:
        vals.append(bits_per_token)
    return vals

def get_stage_epochs(stage_msb: int):
    return int(MSB_STAGE_EPOCHS_MAP.get(int(stage_msb), int(MSB_STAGE_EPOCHS)))
