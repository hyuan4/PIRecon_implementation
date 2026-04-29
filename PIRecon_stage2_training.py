import argparse
import os
import csv
import json
from pathlib import Path
import sys
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from diffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline
from PIRecon_util import (
    CachedMessageDataset,
    ConvBlock,
    LoRAConv2d,
    Reader,
    ResBlock,
    StudentSoftVQHead,
    VAEEncoderStyleVQHead,
    VQCodec,
    Writer,
    _extract_unet_state_dict,
    _get_parent_and_key,
    _radial_rfft_mask,
    _resolve_child,
    _set_child,
    _ste_round,
    apply_fft_bandpass,
    apply_qim_hybrid_latent,
    apply_roundtrip_aug,
    build_msb_schedule,
    build_train_val_sets,
    collect_cached_image_files,
    collect_input_images,
    decode_teacher_quant,
    decode_with_conditioner,
    dump_lora_state,
    get_frozen_message,
    get_stage_epochs,
    indices_to_msb_bit_map,
    indices_to_msb_bitstream,
    infer_decoder_film_targets,
    inject_vae_lora,
    load_lora_state,
    load_vqgan_taming,
    load_writer_reader_state,
    msb_bit_logits_to_indices,
    msb_bitstream_to_indices,
    psnr_torch,
    qim_embed_latent_rfft_stream,
    qim_extract_latent_rfft_bits,
    qim_extract_latent_rfft_logits,
    safe_torch_load,
    sample_qim_positions,
    save_roundtrip_visuals,
    set_conditioner_util_config,
    soft_lookup,
)


SCRIPT_DIR = Path(__file__).resolve().parent


class RawEvalMessageDataset(Dataset):
    def __init__(self, files_list, transform=None, image_size=512):
        super().__init__()
        self.files = [Path(p) for p in files_list]
        self.transform = transform
        self.latent_hw = max(1, int(image_size) // 8)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = Path(self.files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        z_placeholder = torch.zeros(4, self.latent_hw, self.latent_hw, dtype=torch.float32)
        y_placeholder = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
        return img, z_placeholder, y_placeholder, str(img_path)

# ============================================================
# Paths: change these if needed
# ============================================================

# dataset/model preset selector
DATA_PRESET = "lyme"
ARTIFACT_TAG = DATA_PRESET
DATA_PRESETS = {
    "ham10000": {
        "img_root": "../datasets/HAM10000/train",
        "cache_root": "./cached_data/cache_clean_train_ham10000",
        "vq_config_path": "checkpoints/vqgan_imagenet_f16_16384.yaml",
        "vq_ckpt_path": "vqgan_finetuned_ham10000/vqgan_320_gan_final.ckpt",
        "proj_head_ckpt": "checkpoints_softvq_distill_vaeenc/best.pt",
        "message_img_size": 320,
        "train_n": 6000,
        "sd_model_id": "runwayml/stable-diffusion-v1-5",
        "prompt": "a dermoscopic photo of a skin lesion with natural skin texture",
        "num_steps": 20,
        "strength": 0.75,
        "cfg": 2.0,
        "seed": 12345,
        "use_custom_unet": True,
        "custom_unet_ckpt_path": "checkpoints/SDI2I_unet_lesion_pretrain",
    },
    "celebahq256": {
        "img_root": "../datasets/celebahq256_9k/train",
        "cache_root": "./cached_data/cache_clean_train_celebahq256",
        "vq_config_path": "checkpoints/vqgan_imagenet_f16_16384.yaml",
        "vq_ckpt_path": "vqgan_finetuned_celebahq9k_256_320/vqgan_256_320_final.ckpt",
        "proj_head_ckpt": "checkpoints_softvq_distill_vaeenc_celebahq256/best.pt",
        "message_img_size": 256,
        "train_n": 6000,
        "sd_model_id": "nitrosocke/Ghibli-Diffusion",
        "use_custom_unet": False,
        "custom_unet_ckpt_path": "",
    },
    "celebahq320": {
        "img_root": "../datasets/celebahq256_9k/train",
        "cache_root": "./cached_data/cache_clean_train_celebahq256",
        "vq_config_path": "checkpoints/vqgan_imagenet_f16_16384.yaml",
        "vq_ckpt_path": "vqgan_finetuned_celebahq9k_256_320/vqgan_256_320_final.ckpt",
        "proj_head_ckpt": "checkpoints_softvq_distill_vaeenc_celebahq320/best.pt",
        "message_img_size": 320,
        "train_n": 6000,
        "sd_model_id": "nitrosocke/Ghibli-Diffusion",
        "use_custom_unet": False,
        "custom_unet_ckpt_path": "",
    },
    "lyme": {
        "img_root": "data/lyme_demographic_train.csv",
        "val_root": "data/lyme_demographic_test.csv",
        "cache_root": "./cached_data/cache_clean_train_lyme",
        "vq_config_path": "checkpoints/vqgan_imagenet_f16_16384.yaml",
        "vq_ckpt_path": "vqgan_finetuned_lyme/vqgan_320_gan_final.ckpt",
        "proj_head_ckpt": "checkpoints_softvq_distill_vaeenc_lyme/best.pt",
        "message_img_size": 320,
        "train_n": -1,
        "sd_model_id": "runwayml/stable-diffusion-v1-5",
        "use_custom_unet": True,
        "custom_unet_ckpt_path": "checkpoints/unet_pretrain_lyme_strength_0.6",
    },
}
def _artifact_tag() -> str:
    return str(ARTIFACT_TAG or DATA_PRESET)


def _polish_dirname() -> str:
    return "polish_reduced_loss"


def _summary_csv_name() -> str:
    p = Path(str(ABLATION_SUMMARY_CSV).strip())
    if p.is_absolute():
        return str(p)
    return str(p.with_name(f"{p.stem}_reduced_loss{p.suffix}"))


def resolve_data_preset(name: str) -> str:
    low = str(name).strip().lower()
    alias = {
        "ham": "ham10000",
        "ham10000": "ham10000",
        "celeb": "celebahq320",
        "celeba": "celebahq320",
        "celeb256": "celebahq256",
        "celeb320": "celebahq320",
        "celebahq256": "celebahq256",
        "celebahq320": "celebahq320",
        "lyme": "lyme",
    }
    if low in alias:
        return alias[low]
    for key in DATA_PRESETS.keys():
        if key.lower() == low:
            return key
    return str(name)


def apply_data_preset(name: str):
    global DATA_PRESET
    global ARTIFACT_TAG
    global _ACTIVE_PRESET
    global IMG_ROOT
    global VAL_ROOT
    global CACHE_ROOT
    global SD_MODEL_ID
    global USE_CUSTOM_UNET
    global CUSTOM_UNET_CKPT_PATH
    global VQ_CONFIG_PATH
    global VQ_CKPT_PATH
    global PROJ_HEAD_CKPT
    global MESSAGE_IMG_SIZE

    name = resolve_data_preset(name)
    if name not in DATA_PRESETS:
        raise ValueError(f"Unknown DATA_PRESET={name}. Available: {list(DATA_PRESETS.keys())}")

    DATA_PRESET = str(name)
    ARTIFACT_TAG = DATA_PRESET
    _ACTIVE_PRESET = DATA_PRESETS[DATA_PRESET]

    IMG_ROOT = _ACTIVE_PRESET["img_root"]
    VAL_ROOT = str(_ACTIVE_PRESET.get("val_root", ""))
    CACHE_ROOT = _ACTIVE_PRESET["cache_root"]

    SD_MODEL_ID = str(_ACTIVE_PRESET.get("sd_model_id", "runwayml/stable-diffusion-v1-5"))
    USE_CUSTOM_UNET = bool(_ACTIVE_PRESET.get("use_custom_unet", True))
    CUSTOM_UNET_CKPT_PATH = str(_ACTIVE_PRESET.get("custom_unet_ckpt_path", "checkpoints/SDI2I_unet_lesion_pretrain"))

    VQ_CONFIG_PATH = _ACTIVE_PRESET["vq_config_path"]
    VQ_CKPT_PATH = _ACTIVE_PRESET["vq_ckpt_path"]
    PROJ_HEAD_CKPT = _ACTIVE_PRESET["proj_head_ckpt"]
    MESSAGE_IMG_SIZE = int(_ACTIVE_PRESET["message_img_size"])


apply_data_preset(DATA_PRESET)
PROJ_HEAD_FREEZE_BACKBONE = True
PROJ_HEAD_TRAIN_LAST_N_DOWNBLOCKS = 0
PROJ_HEAD_TRAIN_MID_BLOCK = False
PROJ_HEAD_CH = 256

# payload settings
# For realism finetune on conditioned model, default is fixed MSB14.
PAYLOAD_MSB_BITS = 14
PAYLOAD_STRIDE = 1
MSB_STAGE_START = 14
MSB_STAGE_STEP = 2
MSB_STAGE_EPOCHS = 4
MSB_MANUAL_SCHEDULE = [14]
MSB_STAGE_EPOCHS_MAP = {
    14: 20,
}

# base channel settings
ALPHA = 1.0
MSG_WARMUP_EPOCHS = 3
LOSS_WEIGHTS_WARMUP = {
    "w_stego": 0.08,
}
LOSS_WEIGHTS_MAIN = {
    "w_stego": 0.12,
}
NEW_BITS_WEIGHT = 5.0

# reduced realism objective
REALISM_FINE_TUNE = False
CHANNEL_STEP_INTERVAL = 4

# explicit visibility loss on residual delta = stego-clean
W_VIS_FLAT = 0.12
VIS_EDGE_SCALE = 6.0

# QIM injection/extraction (trainingfree-style)
QIM_DELTA = 0.014
QIM_R_LOW = 0.18
QIM_R_HIGH = 0.70
QIM_BITS_PER_POS = 2
QIM_POS_SEED = 123
QIM_LOGIT_SCALE = 6.0
QIM_LOGIT_CLAMP = 20.0
NUM_LATENT_CLAMP = 20.0
NUM_QIM_INDEX_CLAMP = 4096.0

# spectral write constraints (trainingfree-style frequency prior)
FFT_BAND_ENABLE = False
FFT_R_LOW = 0.18
FFT_R_HIGH = 0.70

# LoRA-only training on VAE encoder+decoder
LORA_RANK = 8
LORA_ALPHA = 8.0
LORA_LR = 8e-6
LORA_GRAD_CLIP = 1.0
LORA_DECODER_LAST_N_UPBLOCKS = 2
LORA_ENCODER_FIRST_N_DOWNBLOCKS = 2
LORA_INCLUDE_CONV_IO = True

# writer/reader settings (train with VAE LoRA)
USE_WRITER_READER = True
WRITER_HIDDEN = 128
READER_HIDDEN = 128
WRITER_READER_LR = 5e-5
WRITER_READER_WEIGHT_DECAY = 0.0
# writer architecture:
# - "legacy": current compact writer
# - "decoder_style": U-shaped decoder-like conditioner trunk + optional aux logits head
WRITER_ARCH = "legacy"
WRITER_AUX_LOGITS = True
# decoder conditioner settings
CONDITIONER_DECODER_LAST_N_UPBLOCKS = 2
CONDITIONER_GAMMA_SCALE = 0.25
CONDITIONER_BETA_SCALE = 0.25

# channel augmentation between decode->encode for roundtrip robustness
ROUNDTRIP_AUG_ENABLE = True
AUG_RESIZE_P = 0.7
AUG_RESIZE_MIN = 0.85
AUG_BLUR_P = 0.4
AUG_NOISE_P = 0.4
AUG_NOISE_STD = 0.01

# stage progression gate (advance only when new bits are reliable)
BER_NEW_ADVANCE_THRESHOLD = 0.08
STAGE_MAX_EXTRA_EPOCHS = 8
STAGE_STRICT_ADVANCE = True
STAGE_EARLY_ADVANCE = False
BEST_CKPT_BER_MAX = 0.01

# BER constraint controls
BER_BUDGET_ENABLE = True
BER_TARGET = 0.005
BER_HARD_MAX = 0.01
BER_MSG_MIN_SCALE = 0.40
BER_BUDGET_POWER = 1.0
BER_RT_TRIGGER = 0.005
CONSTRAINT_BER_SOFT_PENALTY = 2.0
CONSTRAINT_RT_PENALTY = 0.02

# stage resume controls
START_FROM_MSB = 14
# load from vaehead base best by default (no realism pre-run required)
RESUME_CKPT_PATH = f"ckpt/{DATA_PRESET}/base/msb_14/best.pt"
AUTO_RESUME_FIRST_STAGE_BEST = False
EVAL_ONLY = False
EVAL_ONLY_CKPT_PATH = ""
EVAL_ONLY_BATCH_SIZE = 1
EVAL_ONLY_NUM_WORKERS = 0
EVAL_ONLY_MAX_VAL_BATCHES = -1
EVAL_ONLY_NUM_VISUAL_BATCHES = 0
EVAL_ONLY_SPLIT = "val"
EVAL_ONLY_SAVE_SRNET_PAIRS = False
EVAL_ONLY_SRNET_ROOT = ""
EVAL_ONLY_SRNET_COVER_SOURCE = "clean"
EVAL_ONLY_NO_KNOWLEDGE = False
EVAL_ONLY_NO_KNOWLEDGE_SEED = -1
EVAL_ONLY_VAL_ROOT = ""
EVAL_ONLY_VAL_TAIL_N = 0
EVAL_ONLY_GENERATE_CLEAN = False
FINAL_METRICS_COMPUTE_SSIM = True
FINAL_METRICS_COMPUTE_LPIPS = True
ABLATION_SUMMARY_CSV = "ablation_decoder_block_number.csv"
EVAL_ONLY_EXPORT_ROOT_DEFAULT = "vis/StegoBackdoor/eval_only"
EVAL_ONLY_VISUAL_ROOT_DEFAULT = "vis"

def _refresh_conditioner_util_config():
    set_conditioner_util_config(
        SD_MODEL_ID=SD_MODEL_ID,
        LORA_RANK=LORA_RANK,
        LORA_ALPHA=LORA_ALPHA,
        LORA_DECODER_LAST_N_UPBLOCKS=LORA_DECODER_LAST_N_UPBLOCKS,
        LORA_ENCODER_FIRST_N_DOWNBLOCKS=LORA_ENCODER_FIRST_N_DOWNBLOCKS,
        LORA_INCLUDE_CONV_IO=LORA_INCLUDE_CONV_IO,
        AUG_RESIZE_P=AUG_RESIZE_P,
        AUG_RESIZE_MIN=AUG_RESIZE_MIN,
        AUG_BLUR_P=AUG_BLUR_P,
        AUG_NOISE_P=AUG_NOISE_P,
        AUG_NOISE_STD=AUG_NOISE_STD,
        NUM_LATENT_CLAMP=NUM_LATENT_CLAMP,
        NUM_QIM_INDEX_CLAMP=NUM_QIM_INDEX_CLAMP,
        QIM_LOGIT_CLAMP=QIM_LOGIT_CLAMP,
        MSB_MANUAL_SCHEDULE=MSB_MANUAL_SCHEDULE,
        PAYLOAD_MSB_BITS=PAYLOAD_MSB_BITS,
        MSB_STAGE_START=MSB_STAGE_START,
        MSB_STAGE_STEP=MSB_STAGE_STEP,
        MSB_STAGE_EPOCHS=MSB_STAGE_EPOCHS,
        MSB_STAGE_EPOCHS_MAP=MSB_STAGE_EPOCHS_MAP,
    )


_refresh_conditioner_util_config()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conditioner FiLM realism-polish training / eval-only runner."
    )
    parser.add_argument(
        "--source-preset",
        type=str,
        default=DATA_PRESET,
        help="Dataset preset: ham10000 | celeb320 | celeb256 | celebahq320 | celebahq256 | lyme",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only on a checkpoint instead of training.",
    )
    parser.add_argument(
        "--eval-only-ckpt-path",
        type=str,
        default="",
        help="Checkpoint path for eval-only mode. Falls back to configured resume path if empty.",
    )
    parser.add_argument(
        "--eval-only-batch-size",
        type=int,
        default=-1,
        help="Validation batch size in eval-only mode. Use a positive value to override the config.",
    )
    parser.add_argument(
        "--eval-only-num-workers",
        type=int,
        default=-1,
        help="Validation dataloader workers in eval-only mode. Use 0 for lightweight deterministic runs.",
    )
    parser.add_argument(
        "--eval-only-max-val-batches",
        type=int,
        default=-1,
        help="Stop eval-only after this many validation batches. Use 30 with batch-size 1 for first-30 samples.",
    )
    parser.add_argument(
        "--eval-only-num-visual-batches",
        type=int,
        default=-1,
        help="How many validation batches to save visualization for in eval-only mode.",
    )
    parser.add_argument(
        "--eval-only-split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which dataset split to use in eval-only mode.",
    )
    parser.add_argument(
        "--eval-only-val-root",
        type=str,
        default="",
        help="Eval-only override for validation image root, e.g. HAM10000/test.",
    )
    parser.add_argument(
        "--eval-only-val-tail-n",
        type=int,
        default=0,
        help="Eval-only: keep only the last N cached validation images after sorting.",
    )
    parser.add_argument(
        "--eval-only-save-srnet-pairs",
        action="store_true",
        help="Eval-only mode: export clean/stego image pairs to root/{cover,stego} for SRNet training.",
    )
    parser.add_argument(
        "--eval-only-srnet-root",
        type=str,
        default="",
        help="Optional output root for --eval-only-save-srnet-pairs. Empty uses the stage eval directory.",
    )
    parser.add_argument(
        "--eval-only-srnet-cover-source",
        type=str,
        default="clean",
        choices=["clean", "original"],
        help="Cover source for exported SRNet pairs: clean synthesized image or original input image.",
    )
    parser.add_argument(
        "--no-knowledge",
        action="store_true",
        help="Eval-only mode: ignore cached clean prompt/seed by regenerating clean latents with empty prompt and a fresh seed.",
    )
    parser.add_argument(
        "--no-knowledge-seed",
        type=int,
        default=-1,
        help="Base seed for --no-knowledge eval. Negative means sample a fresh random seed for this run.",
    )
    parser.add_argument(
        "--custom-unet-ckpt-path",
        type=str,
        default="",
        help="Override preset custom UNet path. In eval-only this lets metrics use a different SD img2img UNet.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default="",
        help="Override preset cached clean latents/images root.",
    )
    return parser.parse_args()


def load_discriminator_state(discriminator: nn.Module, ckpt_obj: dict):
    if discriminator is None:
        return
    blob = ckpt_obj.get("discriminator", None)
    if not isinstance(blob, dict):
        return

    cur = discriminator.state_dict()
    keep = {}
    dropped = 0
    for k, v in blob.items():
        if k in cur and torch.is_tensor(v) and tuple(v.shape) == tuple(cur[k].shape):
            keep[k] = v
        else:
            dropped += 1
    missing, unexpected = discriminator.load_state_dict(keep, strict=False)
    if dropped > 0:
        print(
            f"[Resume:D] partial load due to shape mismatch: loaded={len(keep)}, "
            f"dropped={dropped}, missing={len(missing)}, unexpected={len(unexpected)}"
        )


def _normalize_export_stem(path_like: str) -> str:
    stem = Path(str(path_like)).stem
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "sample"


@torch.no_grad()
def save_eval_only_original_recovered(x, x_rec, paths, save_root: str, step: int):
    root = Path(str(save_root))
    original_dir = root / "original"
    recover_dir = root / "recover"
    original_dir.mkdir(parents=True, exist_ok=True)
    recover_dir.mkdir(parents=True, exist_ok=True)

    x01 = ((x + 1) * 0.5).clamp(0, 1).cpu()
    rec01 = ((x_rec + 1) * 0.5).clamp(0, 1).cpu()
    if rec01.shape[-2:] != x01.shape[-2:]:
        rec01 = F.interpolate(rec01, size=x01.shape[-2:], mode="bilinear", align_corners=False)

    bsz = min(int(x01.size(0)), len(paths))
    for i in range(bsz):
        stem = _normalize_export_stem(paths[i])
        fname = f"{int(step):05d}_{int(i):02d}_{stem}.png"
        save_image(x01[i], str(original_dir / fname))
        save_image(rec01[i], str(recover_dir / fname))
    return bsz


def _to_gray(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def set_requires_grad(module: nn.Module, enabled: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = bool(enabled)


def _active_bits_view(outputs, prev_active_bits=0, new_bits_weight=1.0):
    bits = outputs["bits"]
    bit_logits = outputs["bit_logits"]
    indices = outputs["indices"]
    bits_per_token = int(outputs.get("bits_per_token", 14))
    active_bits = int(max(1, min(int(outputs["active_bits"]), bits_per_token)))
    prev_active_bits = int(max(0, min(int(prev_active_bits), active_bits)))
    nt = int(indices.shape[-2] * indices.shape[-1])
    active_len = int(nt * active_bits)
    prev_len = int(nt * prev_active_bits)

    bits_act = bits[:, :active_len]
    logits_act = bit_logits[:, :active_len]
    if prev_len > 0 and prev_len < active_len and float(new_bits_weight) != 1.0:
        weights = torch.ones_like(bits_act)
        weights[:, prev_len:active_len] = float(new_bits_weight)
    else:
        weights = None
    return bits_act, logits_act, weights


def message_bce_loss(outputs, prev_active_bits=0, new_bits_weight=1.0):
    bits_act, logits_act, weights = _active_bits_view(
        outputs=outputs,
        prev_active_bits=prev_active_bits,
        new_bits_weight=new_bits_weight,
    )
    if weights is None:
        return F.binary_cross_entropy_with_logits(logits_act, bits_act)
    return F.binary_cross_entropy_with_logits(logits_act, bits_act, weight=weights)


def soft_ber_tensor(outputs, prev_active_bits=0):
    bits_act, logits_act, _ = _active_bits_view(
        outputs=outputs,
        prev_active_bits=prev_active_bits,
        new_bits_weight=1.0,
    )
    return (torch.sigmoid(logits_act) - bits_act).abs().mean()


def edge_aware_residual_loss(y_stego, y_clean, edge_scale=6.0):
    clean_gray = _to_gray(y_clean)
    dx = clean_gray[:, :, :, 1:] - clean_gray[:, :, :, :-1]
    dy = clean_gray[:, :, 1:, :] - clean_gray[:, :, :-1, :]
    dx = F.pad(dx.abs(), (0, 1, 0, 0))
    dy = F.pad(dy.abs(), (0, 0, 0, 1))
    edge = dx + dy
    edge_n = edge / (edge.mean(dim=(-2, -1), keepdim=True) + 1e-6)
    flat_weight = torch.exp(-float(edge_scale) * edge_n).to(dtype=y_stego.dtype)
    resid = (y_stego - y_clean).abs().mean(dim=1, keepdim=True)
    return (flat_weight * resid).mean()


def _gaussian_window(window_size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.view(1, 1, 1, -1)
    window_2d = window_1d.transpose(-1, -2) @ window_1d
    return (window_2d / window_2d.sum()).contiguous()


@torch.no_grad()
def ssim_torch(img1, img2, data_range=1.0, window_size=11, sigma=1.5, k1=0.01, k2=0.03):
    """
    img1,img2: (B,3,H,W) in [0,1]
    returns mean SSIM over batch
    """
    img1 = img1.float()
    img2 = img2.float()
    c = int(img1.size(1))
    window = _gaussian_window(window_size, sigma, device=img1.device, dtype=img1.dtype)
    window = window.expand(c, 1, window_size, window_size)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=c)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=c) - mu12

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / den
    return float(ssim_map.mean().item())


def build_lpips_model(device: str):
    """
    Returns:
      - LPIPS model callable if installed
      - None if LPIPS package is unavailable
    """
    try:
        import lpips  # type: ignore
    except Exception:
        print("[FinalMetrics] lpips package not found; LPIPS metric will be NaN.")
        return None

    try:
        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print("[FinalMetrics] LPIPS(alex) enabled.")
        return model
    except Exception as e:
        print(f"[FinalMetrics] LPIPS init failed; metric will be NaN. err={e}")
        return None


@torch.no_grad()
def lpips_torch(img1_01, img2_01, lpips_model) -> float:
    if lpips_model is None:
        return float("nan")
    # LPIPS expects [-1,1]
    x = (img1_01 * 2.0 - 1.0).clamp(-1, 1)
    y = (img2_01 * 2.0 - 1.0).clamp(-1, 1)
    val = lpips_model(x, y)
    return float(val.mean().item())


@torch.no_grad()
def _encode_prompt(pipe, prompt: str, device: str, batch_size: int):
    text_inputs = pipe.tokenizer(
        [str(prompt)] * int(batch_size),
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    return pipe.text_encoder(text_input_ids)[0]


@torch.no_grad()
def run_img2img_clean(pipe, x0, prompt, num_steps=20, strength=0.75, cfg=2.0, seed=12345):
    device = x0.device
    sf = pipe.vae.config.scaling_factor
    bsz = int(x0.size(0))

    z0 = pipe.vae.encode(x0).latent_dist.mode() * sf

    text = _encode_prompt(pipe, prompt=prompt, device=device, batch_size=bsz)
    uncond = _encode_prompt(pipe, prompt="", device=device, batch_size=bsz)
    cond = torch.cat([uncond, text], dim=0)

    pipe.scheduler.set_timesteps(int(num_steps), device=device)
    timesteps_all = pipe.scheduler.timesteps
    init_t = min(int(num_steps * float(strength)), int(num_steps))
    timesteps = timesteps_all[max(int(num_steps) - init_t, 0):]
    if len(timesteps) == 0:
        timesteps = timesteps_all[-1:]

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    noise = torch.randn(z0.shape, generator=g, device=device, dtype=z0.dtype)
    lat = pipe.scheduler.add_noise(z0, noise, timesteps[0])

    for t in timesteps:
        lat_in = torch.cat([lat, lat], dim=0)
        lat_in = pipe.scheduler.scale_model_input(lat_in, t)
        eps = pipe.unet(lat_in, t, encoder_hidden_states=cond).sample
        eps_u, eps_c = eps.chunk(2)
        eps_pred = eps_u + float(cfg) * (eps_c - eps_u)
        lat = pipe.scheduler.step(eps_pred, t, lat).prev_sample

    z_clean = lat
    y_clean = pipe.vae.decode(z_clean / sf).sample.clamp(-1, 1)
    return z_clean, y_clean


def _build_eval_img2img_pipe(device: str):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    if bool(USE_CUSTOM_UNET) and len(str(CUSTOM_UNET_CKPT_PATH).strip()) > 0:
        custom_path = Path(str(CUSTOM_UNET_CKPT_PATH).strip())
        if not custom_path.is_absolute():
            custom_path = SCRIPT_DIR / custom_path
        if custom_path.is_dir():
            unet_dir = custom_path / "unet" if (custom_path / "unet").is_dir() else custom_path
            try:
                pipe.unet = pipe.unet.from_pretrained(
                    str(unet_dir),
                    subfolder=None,
                    use_safetensors=True,
                ).to(device)
            except Exception:
                pipe.unet = pipe.unet.from_pretrained(
                    str(unet_dir),
                    subfolder=None,
                    use_safetensors=False,
                ).to(device)
        else:
            unet_ckpt = safe_torch_load(str(custom_path), map_location="cpu")
            unet_state = _extract_unet_state_dict(unet_ckpt)
            pipe.unet.load_state_dict(unet_state, strict=False)

    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    for module in (pipe.unet, pipe.vae, pipe.text_encoder):
        for p in module.parameters():
            p.requires_grad = False
    return pipe


@torch.no_grad()
def save_srnet_pairs(cover, y_stego, paths, save_root: str, step: int):
    root = Path(str(save_root))
    cover_dir = root / "cover"
    stego_dir = root / "stego"
    cover_dir.mkdir(parents=True, exist_ok=True)
    stego_dir.mkdir(parents=True, exist_ok=True)

    cover01 = ((cover + 1) * 0.5).clamp(0, 1).cpu()
    stego01 = ((y_stego + 1) * 0.5).clamp(0, 1).cpu()
    bsz = min(int(cover01.size(0)), len(paths))
    for i in range(bsz):
        stem = Path(str(paths[i])).stem
        fname = f"{int(step):05d}_{int(i):02d}_{stem}.png"
        save_image(cover01[i], str(cover_dir / fname))
        save_image(stego01[i], str(stego_dir / fname))


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.GroupNorm(8, base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.GroupNorm(8, base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, 1, 1),
            nn.GroupNorm(8, base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, 1, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def forward_roundtrip(
    sd_vae,
    proj_head,
    codebook_size,
    vq_codec,
    writer,
    reader,
    x,
    z_clean,
    y_clean,
    alpha=0.01,
    message_img_size=MESSAGE_IMG_SIZE,
    message_ch=14,
    apply_channel_aug=False,
    payload_msb_bits=4,
    payload_stride=1,
    target_block_idxs=None,
):
    sf = sd_vae.config.scaling_factor
    sd_dtype = sd_vae.dtype

    if writer is None or reader is None:
        raise RuntimeError("Writer/Reader must be provided in WR+LoRA mode.")
    if target_block_idxs is None:
        raise RuntimeError("target_block_idxs must be provided for conditioner writer mode.")

    # projection-head message source
    with torch.no_grad():
        x_msg = F.interpolate(
            x.float(),
            size=(int(message_img_size), int(message_img_size)),
            mode="bilinear",
            align_corners=False,
        )
        logits = proj_head(x_msg)
        indices = torch.argmax(logits, dim=1)
        bits_per_token = int(math.ceil(math.log2(int(codebook_size))))
        ht, wt = int(indices.shape[-2]), int(indices.shape[-1])

    writer_dtype = next(writer.parameters()).dtype
    reader_dtype = next(reader.parameters()).dtype

    m = indices_to_msb_bit_map(
        indices=indices,
        codebook_size=codebook_size,
        msb_bits=payload_msb_bits,
        stride=payload_stride,
        out_channels=message_ch,
    ).to(dtype=writer_dtype, device=z_clean.device)
    z_for_writer = z_clean.to(dtype=writer_dtype)
    writer_out = writer(m, z_for_writer)
    writer_aux_logits = None
    if isinstance(writer_out, tuple):
        film_pairs, writer_aux_logits = writer_out
    else:
        film_pairs = writer_out
    z_stego = z_clean.float()

    # VAE decode (with internal conditioner) -> stego image -> encode roundtrip
    z_decode = (z_stego.to(dtype=sd_dtype)) / sf
    y_stego_raw = decode_with_conditioner(
        sd_vae=sd_vae,
        z_in=z_decode,
        film_pairs=film_pairs,
        target_block_idxs=target_block_idxs,
    )
    y_stego = y_stego_raw.clamp(-1, 1)
    y_chan = apply_roundtrip_aug(y_stego) if bool(apply_channel_aug and ROUNDTRIP_AUG_ENABLE) else y_stego
    z_rt = sd_vae.encode(y_chan.to(dtype=sd_dtype)).latent_dist.mode() * sf
    z_rt = torch.nan_to_num(
        z_rt, nan=0.0, posinf=float(NUM_LATENT_CLAMP), neginf=-float(NUM_LATENT_CLAMP)
    ).clamp(-float(NUM_LATENT_CLAMP), float(NUM_LATENT_CLAMP))

    m_hat_logits = reader(z_rt.to(dtype=reader_dtype))
    bits_map = indices_to_msb_bit_map(
        indices=indices,
        codebook_size=codebook_size,
        msb_bits=payload_msb_bits,
        stride=payload_stride,
    ).to(torch.float32)
    hs, ws = int(bits_map.shape[-2]), int(bits_map.shape[-1])
    if m_hat_logits.shape[-2:] != (hs, ws):
        m_hat_logits = F.interpolate(m_hat_logits, size=(hs, ws), mode="bilinear", align_corners=False)
    bit_logits_map = m_hat_logits[:, :int(payload_msb_bits), :, :]
    bits_map = bits_map.to(bit_logits_map.dtype)
    bit_logits = bit_logits_map.flatten(1)
    bits = bits_map.flatten(1).to(torch.float32)
    bits_hat = (torch.sigmoid(bit_logits) >= 0.5).to(torch.uint8)
    indices_hat = msb_bit_logits_to_indices(
        bit_logits=bit_logits_map,
        codebook_size=codebook_size,
        target_hw=(ht, wt),
        msb_bits=payload_msb_bits,
        stride=payload_stride,
    )
    nbits = int(bits.size(1))

    x_rec = vq_codec.decode_from_indices(indices_hat)
    if x_rec.shape[-2:] != x_msg.shape[-2:]:
        x_rec = F.interpolate(x_rec, size=(x_msg.shape[-2], x_msg.shape[-1]), mode="bilinear", align_corners=False)

    return {
        "bits": bits.to(torch.float32),
        "bit_logits": bit_logits,
        "bits_hat": bits_hat.to(torch.float32),
        "indices": indices,
        "indices_hat": indices_hat,
        "active_bits": int(payload_msb_bits),
        "bits_per_token": int(bits_per_token),
        "nbits": int(nbits),
        "x_msg": x_msg.float(),
        "y_clean": y_clean.float(),
        "y_stego": y_stego.float(),  # for loss / vis
        "y_chan": y_chan.float(),
        "y_stego_raw": y_stego_raw.float(),  # optional debug
        "x_rec": x_rec.float(),
        "z_stego": z_stego.float(),
        "z_rt": z_rt.float(),
        "writer_aux_logits": writer_aux_logits.float() if torch.is_tensor(writer_aux_logits) else None,
        "writer_aux_target": m.float(),
        "delta": torch.zeros_like(z_stego, dtype=torch.float32),
    }


# ============================================================
# Loss
# ============================================================

def compute_losses(
    outputs,
    w_msg=1.0,
    w_stego=0.1,
    prev_active_bits=0,
    new_bits_weight=1.0,
):
    bits = outputs["bits"]                 # (B,L) float(0/1)
    bit_logits = outputs["bit_logits"]     # (B,L) logits
    bits_hat = outputs["bits_hat"]         # (B,L) float(0/1)
    indices = outputs["indices"]
    indices_hat = outputs["indices_hat"]
    y_clean = outputs["y_clean"]
    y_stego = outputs["y_stego"]

    bits_per_token = int(outputs.get("bits_per_token", 14))
    active_bits = int(max(1, min(int(outputs["active_bits"]), bits_per_token)))
    prev_active_bits = int(max(0, min(prev_active_bits, active_bits)))
    nt = int(indices.shape[-2] * indices.shape[-1])
    active_len = int(nt * active_bits)
    prev_len = int(nt * prev_active_bits)

    bits_act = bits[:, :active_len]
    logits_act = bit_logits[:, :active_len]
    hats_act = bits_hat[:, :active_len]

    if prev_len > 0 and prev_len < active_len and float(new_bits_weight) != 1.0:
        w = torch.ones_like(bits_act)
        w[:, prev_len:active_len] = float(new_bits_weight)
        loss_msg = F.binary_cross_entropy_with_logits(logits_act, bits_act, weight=w)
    else:
        loss_msg = F.binary_cross_entropy_with_logits(logits_act, bits_act)

    # differentiable BER proxy for budgeted message optimization
    soft_bits = torch.sigmoid(logits_act)
    soft_ber = (soft_bits - bits_act).abs().mean()
    msg_scale = torch.ones((), device=soft_ber.device, dtype=soft_ber.dtype)
    loss_ber_hard = torch.zeros((), device=soft_ber.device, dtype=soft_ber.dtype)
    if bool(BER_BUDGET_ENABLE):
        tgt = float(max(0.0, min(0.5, BER_TARGET)))
        denom = float(max(1e-6, 1.0 - tgt))
        excess = F.relu(soft_ber - tgt)
        msg_scale = (excess / denom).pow(float(BER_BUDGET_POWER))
        msg_scale = torch.clamp(msg_scale, min=float(BER_MSG_MIN_SCALE), max=1.0)
        hmax = float(max(tgt, BER_HARD_MAX))
        loss_ber_hard = F.relu(soft_ber - hmax)
    w_msg_eff = float(w_msg) * msg_scale

    loss_stego = F.l1_loss(y_stego, y_clean)
    ber = (hats_act - bits_act).abs().mean().item()
    ber_old = (hats_act[:, :prev_len] - bits_act[:, :prev_len]).abs().mean().item() if prev_len > 0 else 0.0
    ber_new = (hats_act[:, prev_len:active_len] - bits_act[:, prev_len:active_len]).abs().mean().item() if prev_len < active_len else 0.0
    token_acc = (indices_hat == indices).float().mean().item()
    if active_bits < bits_per_token:
        low_bits = int(bits_per_token - active_bits)
        token_acc_coarse = (((indices_hat >> low_bits) == (indices >> low_bits)).float().mean().item())
    else:
        token_acc_coarse = token_acc

    loss = (
        w_msg_eff * loss_msg
        + w_stego * loss_stego
    )

    metrics = {
        "loss": loss.item(),
        "loss_msg": loss_msg.item(),
        "loss_msg_eff": (w_msg_eff * loss_msg).item(),
        "loss_ber_hard": float(loss_ber_hard.item()),
        "loss_stego": loss_stego.item(),
        "ber": ber,
        "ber_soft": float(soft_ber.item()),
        "ber_old": ber_old,
        "ber_new": ber_new,
        "ber_res": 0.0,
        "msg_scale": float(msg_scale.item()),
        "w_msg_eff": float(w_msg_eff.item()),
        "token_acc": token_acc,
        "token_acc_coarse": token_acc_coarse,
    }
    for bi in range(active_bits):
        s = int(bi * nt)
        e = int((bi + 1) * nt)
        metrics[f"ber_b{bi+1:02d}"] = (hats_act[:, s:e] - bits_act[:, s:e]).abs().mean().item()
    return loss, metrics


def get_loss_weights(epoch: int):
    if int(epoch) <= int(MSG_WARMUP_EPOCHS):
        return dict(LOSS_WEIGHTS_WARMUP)
    return dict(LOSS_WEIGHTS_MAIN)


def get_realism_weights(epoch: int):
    return {
        "w_adv": 0.0,
        "w_tv": 0.0,
        "w_freq": 0.0,
    }


def get_constraint_weights():
    return {
        "w_ber_soft": float(CONSTRAINT_BER_SOFT_PENALTY),
        "w_rt_trig": float(CONSTRAINT_RT_PENALTY),
    }


def format_progress_postfix(running: dict, denom: float, include_disc: bool = False):
    parts = [
        f"L={running['loss']/denom:.3f}",
        f"B={running['ber']/denom:.3f}",
        f"SB={running.get('ber_soft', 0.0)/denom:.3f}",
        f"N={running['ber_new']/denom:.3f}",
        f"PS={running['psnr_stego']/denom:.2f}",
        f"RTt={running.get('loss_rt_trig', 0.0)/denom:.3f}",
        f"V={running.get('loss_vis', 0.0)/denom:.3f}",
        f"T={running['token_acc']/denom:.4f}",
    ]
    if include_disc:
        parts.append(f"D={running['loss_disc']/denom:.3f}")
    return " | ".join(parts)






# ============================================================
# Train / Eval
# ============================================================

def train_one_epoch(
    sd_vae,
    proj_head,
    codebook_size,
    vq_codec,
    writer,
    reader,
    discriminator,
    loader,
    optimizer_vis,
    optimizer_reader,
    optim_d,
    device,
    epoch,
    alpha=0.01,
    message_img_size=MESSAGE_IMG_SIZE,
    message_ch=14,
    prev_payload_msb_bits=0,
    new_bits_weight=1.0,
    payload_msb_bits=4,
    payload_stride=1,
    target_block_idxs=None,
):
    sd_vae.eval()
    if writer is not None:
        writer.train()
    if reader is not None:
        reader.train()
    if discriminator is not None:
        discriminator.train()
    loss_w = get_loss_weights(epoch)
    realism_w = get_realism_weights(epoch)
    constraint_w = get_constraint_weights()

    running = {
        "loss": 0.0,
        "loss_base": 0.0,
        "loss_msg": 0.0,
        "loss_msg_eff": 0.0,
        "loss_ber_hard": 0.0,
        "loss_stego": 0.0,
        "loss_vis_lowfreq": 0.0,
        "loss_vis_chroma": 0.0,
        "loss_vis_flat": 0.0,
        "loss_vis": 0.0,
        "loss_ber_soft_pen": 0.0,
        "loss_rt_trig": 0.0,
        "loss_channel": 0.0,
        "loss_adv_g": 0.0,
        "loss_tv": 0.0,
        "loss_freq": 0.0,
        "loss_disc": 0.0,
        "psnr": 0.0,
        "psnr_stego": 0.0,
        "ber": 0.0,
        "ber_soft": 0.0,
        "ber_old": 0.0,
        "ber_new": 0.0,
        "ber_res": 0.0,
        "msg_scale": 0.0,
        "w_msg_eff": 0.0,
        "token_acc": 0.0,
        "token_acc_coarse": 0.0,
    }

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for step, (x, z_clean, y_clean, _) in enumerate(pbar):
        x = x.to(device)
        z_clean = z_clean.to(device)
        y_clean = y_clean.to(device)
        do_channel_step = (int(CHANNEL_STEP_INTERVAL) > 0 and (step % int(CHANNEL_STEP_INTERVAL) == 0))
        loss_disc = torch.zeros((), device=device)
        loss_adv_g = torch.zeros((), device=device)
        loss_tv = torch.zeros((), device=device)
        loss_freq = torch.zeros((), device=device)
        loss_vis_lowfreq = torch.zeros((), device=device)
        loss_vis_chroma = torch.zeros((), device=device)
        loss_vis_flat = torch.zeros((), device=device)
        loss_vis = torch.zeros((), device=device)
        loss_ber_soft_pen = torch.zeros((), device=device)
        loss_rt_trig = torch.zeros((), device=device)

        if do_channel_step:
            set_requires_grad(writer, False)
            set_requires_grad(reader, True)
            optimizer_reader.zero_grad()

            outputs = forward_roundtrip(
                sd_vae=sd_vae,
                proj_head=proj_head,
                codebook_size=codebook_size,
                vq_codec=vq_codec,
                writer=writer,
                reader=reader,
                x=x,
                z_clean=z_clean,
                y_clean=y_clean,
                alpha=alpha,
                message_img_size=message_img_size,
                message_ch=message_ch,
                apply_channel_aug=True,
                payload_msb_bits=payload_msb_bits,
                payload_stride=payload_stride,
                target_block_idxs=target_block_idxs,
            )
            loss_channel = message_bce_loss(
                outputs=outputs,
                prev_active_bits=prev_payload_msb_bits,
                new_bits_weight=new_bits_weight,
            )
            if not torch.isfinite(loss_channel):
                raise RuntimeError(f"Non-finite channel loss at epoch={epoch}, step={step}.")
            loss_channel.backward()
            if float(LORA_GRAD_CLIP) > 0 and reader is not None:
                torch.nn.utils.clip_grad_norm_(list(reader.parameters()), float(LORA_GRAD_CLIP))
            optimizer_reader.step()

            with torch.no_grad():
                loss_base, metrics = compute_losses(
                    outputs,
                    w_msg=1.0,
                    w_stego=0.0,
                    prev_active_bits=prev_payload_msb_bits,
                    new_bits_weight=new_bits_weight,
                )
                metrics["loss"] = float(loss_channel.item())
                metrics["loss_base"] = float(loss_base.item())
                metrics["loss_channel"] = float(loss_channel.item())
                metrics["loss_vis_lowfreq"] = 0.0
                metrics["loss_vis_chroma"] = 0.0
                metrics["loss_vis_flat"] = 0.0
                metrics["loss_vis"] = 0.0
                metrics["loss_ber_soft_pen"] = 0.0
                metrics["loss_rt_trig"] = 0.0
                metrics["loss_adv_g"] = 0.0
                metrics["loss_tv"] = 0.0
                metrics["loss_freq"] = 0.0
                metrics["loss_disc"] = 0.0
                metrics["msg_scale"] = 1.0

        else:
            set_requires_grad(writer, True)
            set_requires_grad(reader, False)

            outputs = forward_roundtrip(
                sd_vae=sd_vae,
                proj_head=proj_head,
                codebook_size=codebook_size,
                vq_codec=vq_codec,
                writer=writer,
                reader=reader,
                x=x,
                z_clean=z_clean,
                y_clean=y_clean,
                alpha=alpha,
                message_img_size=message_img_size,
                message_ch=message_ch,
                apply_channel_aug=False,
                payload_msb_bits=payload_msb_bits,
                payload_stride=payload_stride,
                target_block_idxs=target_block_idxs,
            )

            optimizer_vis.zero_grad()
            loss_base, metrics = compute_losses(
                outputs,
                w_msg=0.0,
                w_stego=float(loss_w.get("w_stego", 0.0)),
                prev_active_bits=prev_payload_msb_bits,
                new_bits_weight=new_bits_weight,
            )
            soft_ber = soft_ber_tensor(outputs, prev_active_bits=prev_payload_msb_bits)
            loss_ber_soft_pen = F.relu(soft_ber - float(BER_TARGET))
            rt_ratio = ((soft_ber - float(BER_RT_TRIGGER)) / float(max(1e-6, 1.0 - float(BER_RT_TRIGGER)))).clamp(0.0, 1.0)
            loss_rt_raw = F.l1_loss(outputs["z_rt"].float(), outputs["z_stego"].float().detach())
            loss_rt_trig = rt_ratio * loss_rt_raw

            delta_img = outputs["y_stego"] - outputs["y_clean"]
            loss_vis_lowfreq = torch.zeros((), device=device)
            loss_vis_chroma = torch.zeros((), device=device)
            loss_vis_flat = edge_aware_residual_loss(
                y_stego=outputs["y_stego"],
                y_clean=outputs["y_clean"],
                edge_scale=float(VIS_EDGE_SCALE),
            )
            loss_vis = (
                float(W_VIS_FLAT) * loss_vis_flat
            )

            loss_tv = torch.zeros((), device=device)
            loss_freq = torch.zeros((), device=device)
            loss = (
                loss_base
                + loss_vis
                + float(constraint_w["w_ber_soft"]) * loss_ber_soft_pen
                + float(constraint_w["w_rt_trig"]) * loss_rt_trig
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected at epoch={epoch}, step={step}. "
                    f"Try lowering learning rates (LORA_LR/WRITER_READER_LR) and payload MSB."
                )
            loss.backward()
            if float(LORA_GRAD_CLIP) > 0:
                clip_params = []
                for group in optimizer_vis.param_groups:
                    clip_params.extend(group["params"])
                torch.nn.utils.clip_grad_norm_(clip_params, float(LORA_GRAD_CLIP))
            optimizer_vis.step()

            with torch.no_grad():
                metrics["loss"] = float(loss.item())
                metrics["loss_base"] = float(loss_base.item())
                metrics["loss_channel"] = 0.0
                metrics["loss_vis_lowfreq"] = float(loss_vis_lowfreq.item())
                metrics["loss_vis_chroma"] = float(loss_vis_chroma.item())
                metrics["loss_vis_flat"] = float(loss_vis_flat.item())
                metrics["loss_vis"] = float(loss_vis.item())
                metrics["loss_ber_soft_pen"] = float(loss_ber_soft_pen.item())
                metrics["loss_rt_trig"] = float(loss_rt_trig.item())
                metrics["loss_adv_g"] = float(loss_adv_g.item())
                metrics["loss_tv"] = float(loss_tv.item())
                metrics["loss_freq"] = float(loss_freq.item())
                metrics["loss_disc"] = float(loss_disc.item())
                metrics["ber_soft"] = float(soft_ber.item())

        for k, v in metrics.items():
            if k not in running:
                running[k] = 0.0
            running[k] += v

        denom = step + 1
        pbar.set_postfix_str(format_progress_postfix(running, denom, include_disc=True))

    for k in list(running.keys()):
        running[k] /= len(loader)
    set_requires_grad(reader, True)
    set_requires_grad(writer, True)
    return running


@torch.no_grad()
def eval_one_epoch(
    sd_vae,
    proj_head,
    codebook_size,
    vq_codec,
    writer,
    reader,
    discriminator,
    loader,
    device,
    epoch,
    visual_dir="roundtrip_visuals",
    num_visual_batches=5,
    alpha=0.01,
    message_img_size=MESSAGE_IMG_SIZE,
    message_ch=14,
    prev_payload_msb_bits=0,
    new_bits_weight=1.0,
    payload_msb_bits=4,
    payload_stride=1,
    target_block_idxs=None,
    compute_perceptual=False,
    lpips_model=None,
    max_eval_batches=-1,
    no_knowledge_pipe=None,
    no_knowledge_seed_base=None,
    no_knowledge_prompt="",
    no_knowledge_num_steps=20,
    no_knowledge_strength=0.75,
    no_knowledge_cfg=2.0,
    clean_generation_tag="no_knowledge",
    srnet_pair_root=None,
    srnet_cover_source="clean",
    export_original_recovered_root=None,
):
    sd_vae.eval()
    if writer is not None:
        writer.eval()
    if reader is not None:
        reader.eval()
    if discriminator is not None:
        discriminator.eval()
    loss_w = get_loss_weights(epoch)
    realism_w = get_realism_weights(epoch)
    constraint_w = get_constraint_weights()

    running = {
        "loss": 0.0,
        "loss_base": 0.0,
        "loss_msg": 0.0,
        "loss_msg_eff": 0.0,
        "loss_ber_hard": 0.0,
        "loss_stego": 0.0,
        "loss_vis_lowfreq": 0.0,
        "loss_vis_chroma": 0.0,
        "loss_vis_flat": 0.0,
        "loss_vis": 0.0,
        "loss_ber_soft_pen": 0.0,
        "loss_rt_trig": 0.0,
        "loss_channel": 0.0,
        "loss_adv_g": 0.0,
        "loss_tv": 0.0,
        "loss_freq": 0.0,
        "loss_disc": 0.0,
        "psnr": 0.0,
        "psnr_stego": 0.0,
        "ber": 0.0,
        "ber_soft": 0.0,
        "ber_old": 0.0,
        "ber_new": 0.0,
        "ber_res": 0.0,
        "msg_scale": 0.0,
        "w_msg_eff": 0.0,
        "token_acc": 0.0,
        "token_acc_coarse": 0.0,
        "psnr_rec": 0.0,
        "ssim_rec": 0.0,
        "lpips_rec": 0.0,
        "mse_rec": 0.0,
        "mse_stego": 0.0,
        "ssim_stego": 0.0,
        "lpips_stego": 0.0,
    }

    saved_batches = 0
    seen_steps = 0
    exported_images = 0
    pbar = tqdm(loader, desc=f"Val Epoch {epoch}")
    for step, (x, z_clean, y_clean, paths) in enumerate(pbar):
        if int(max_eval_batches) > 0 and int(step) >= int(max_eval_batches):
            break
        seen_steps += 1
        x = x.to(device)
        z_clean = z_clean.to(device)
        y_clean = y_clean.to(device)
        if no_knowledge_pipe is not None:
            nk_seed = int(no_knowledge_seed_base) + int(step)
            x_pipe = x.to(
                device=device,
                dtype=no_knowledge_pipe.vae.dtype if hasattr(no_knowledge_pipe.vae, "dtype") else x.dtype,
            )
            z_clean, y_clean = run_img2img_clean(
                pipe=no_knowledge_pipe,
                x0=x_pipe,
                prompt=str(no_knowledge_prompt),
                num_steps=int(no_knowledge_num_steps),
                strength=float(no_knowledge_strength),
                cfg=float(no_knowledge_cfg),
                seed=nk_seed,
            )
            z_clean = z_clean.to(device=device, dtype=torch.float32)
            y_clean = y_clean.to(device=device, dtype=torch.float32)

        outputs = forward_roundtrip(
            sd_vae=sd_vae,
            proj_head=proj_head,
            codebook_size=codebook_size,
            vq_codec=vq_codec,
            writer=writer,
            reader=reader,
            x=x,
            z_clean=z_clean,
            y_clean=y_clean,
            alpha=alpha,
            message_img_size=message_img_size,
            message_ch=message_ch,
            apply_channel_aug=False,
            payload_msb_bits=payload_msb_bits,
            payload_stride=payload_stride,
            target_block_idxs=target_block_idxs,
        )

        loss_base, metrics = compute_losses(
            outputs,
            w_msg=0.0,
            w_stego=float(loss_w.get("w_stego", 0.0)),
            prev_active_bits=prev_payload_msb_bits,
            new_bits_weight=new_bits_weight,
        )
        loss_adv_g = torch.zeros((), device=device)
        loss_tv = torch.zeros((), device=device)
        loss_freq = torch.zeros((), device=device)
        soft_ber = soft_ber_tensor(outputs, prev_active_bits=prev_payload_msb_bits)
        loss_ber_soft_pen = F.relu(soft_ber - float(BER_TARGET))
        rt_ratio = ((soft_ber - float(BER_RT_TRIGGER)) / float(max(1e-6, 1.0 - float(BER_RT_TRIGGER)))).clamp(0.0, 1.0)
        loss_rt_raw = F.l1_loss(outputs["z_rt"].float(), outputs["z_stego"].float().detach())
        loss_rt_trig = rt_ratio * loss_rt_raw
        delta_img = outputs["y_stego"] - outputs["y_clean"]
        loss_vis_lowfreq = torch.zeros((), device=device)
        loss_vis_chroma = torch.zeros((), device=device)
        loss_vis_flat = edge_aware_residual_loss(
            y_stego=outputs["y_stego"],
            y_clean=outputs["y_clean"],
            edge_scale=float(VIS_EDGE_SCALE),
        )
        loss_vis = (
            float(W_VIS_FLAT) * loss_vis_flat
        )
        loss_total = (
            loss_base
            + loss_vis
            + float(constraint_w["w_ber_soft"]) * loss_ber_soft_pen
            + float(constraint_w["w_rt_trig"]) * loss_rt_trig
        )

        x01 = ((outputs["x_msg"] + 1) * 0.5).clamp(0, 1)
        xrec01 = ((outputs["x_rec"] + 1) * 0.5).clamp(0, 1)
        metrics["psnr"] = psnr_torch(x01, xrec01)
        metrics["psnr_rec"] = metrics["psnr"]
        metrics["mse_rec"] = float(F.mse_loss(xrec01, x01, reduction="mean").item())
        clean01 = ((outputs["y_clean"] + 1) * 0.5).clamp(0, 1)
        stego01 = ((outputs["y_stego"] + 1) * 0.5).clamp(0, 1)
        metrics["psnr_stego"] = psnr_torch(stego01, clean01)
        metrics["mse_stego"] = float(F.mse_loss(stego01, clean01, reduction="mean").item())
        if bool(compute_perceptual):
            metrics["ssim_rec"] = ssim_torch(xrec01, x01, data_range=1.0)
            metrics["lpips_rec"] = lpips_torch(xrec01, x01, lpips_model=lpips_model)
            metrics["ssim_stego"] = ssim_torch(stego01, clean01, data_range=1.0)
            metrics["lpips_stego"] = lpips_torch(stego01, clean01, lpips_model=lpips_model)
        metrics["loss_base"] = loss_base.item()
        metrics["loss_adv_g"] = loss_adv_g.item()
        metrics["loss_tv"] = loss_tv.item()
        metrics["loss_freq"] = loss_freq.item()
        metrics["loss_vis_lowfreq"] = float(loss_vis_lowfreq.item())
        metrics["loss_vis_chroma"] = float(loss_vis_chroma.item())
        metrics["loss_vis_flat"] = float(loss_vis_flat.item())
        metrics["loss_vis"] = float(loss_vis.item())
        metrics["loss_ber_soft_pen"] = float(loss_ber_soft_pen.item())
        metrics["loss_rt_trig"] = float(loss_rt_trig.item())
        metrics["loss_channel"] = 0.0
        metrics["ber_soft"] = float(soft_ber.item())
        metrics["loss_disc"] = 0.0
        metrics["loss"] = loss_total.item()

        for k, v in metrics.items():
            if k not in running:
                running[k] = 0.0
            running[k] += v

        denom = step + 1
        pbar.set_postfix_str(format_progress_postfix(running, denom, include_disc=False))

        if export_original_recovered_root is not None:
            exported_images += int(
                save_eval_only_original_recovered(
                    x=x,
                    x_rec=outputs["x_rec"],
                    paths=paths,
                    save_root=str(export_original_recovered_root),
                    step=int(step),
                )
            )
        if saved_batches < num_visual_batches:
            save_roundtrip_visuals(
                x=x,
                y_clean=outputs["y_clean"],
                y_stego=outputs["y_stego"],
                x_rec=outputs["x_rec"],
                paths=paths,
                save_dir=os.path.join(visual_dir, f"epoch_{epoch:03d}"),
                epoch=epoch,
                step=step,
                max_save=4,
            )
            saved_batches += 1
        if srnet_pair_root is not None:
            srnet_cover = x if str(srnet_cover_source).strip().lower() == "original" else outputs["y_clean"]
            save_srnet_pairs(
                cover=srnet_cover,
                y_stego=outputs["y_stego"],
                paths=paths,
                save_root=str(srnet_pair_root),
                step=int(step),
            )
    if int(seen_steps) <= 0:
        raise RuntimeError("no validation batches evaluated")
    for k in list(running.keys()):
        running[k] /= float(seen_steps)
    running["exported_images"] = float(exported_images)
    return running


def _resolve_summary_csv_path(tag_suffix: str = "") -> Path:
    p = Path(_summary_csv_name())
    if len(str(tag_suffix).strip()) > 0:
        p = p.with_name(f"{p.stem}_{str(tag_suffix).strip()}{p.suffix}")
    if p.is_absolute():
        return p
    return SCRIPT_DIR / f"ckpt/{_artifact_tag()}" / p


def _append_summary_csv_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "data_preset",
        "stage_msb",
        "lora_decoder_last_n_upblocks",
        "conditioner_decoder_last_n_upblocks",
        "ber",
        "psnr_rec",
        "ssim_rec",
        "lpips_rec",
        "mse_rec",
        "psnr_stego",
        "ssim_stego",
        "lpips_stego",
        "mse_stego",
        "best_ckpt",
    ]
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def evaluate_final_metrics_for_ckpt(
    ckpt_path: Path,
    stage_dir: Path,
    lora_modules: dict,
    sd_vae,
    proj_head,
    codebook_size,
    vq_codec,
    writer,
    reader,
    discriminator,
    eval_loader,
    device: str,
    alpha: float,
    message_img_size: int,
    bits_per_token: int,
    payload_stride: int,
    target_block_idxs,
    eval_split="val",
    lpips_model=None,
    num_visual_batches=0,
    no_knowledge_pipe=None,
    no_knowledge_seed_base=None,
    no_knowledge_prompt="",
    no_knowledge_num_steps=20,
    no_knowledge_strength=0.75,
    no_knowledge_cfg=2.0,
    clean_generation_tag="no_knowledge",
    srnet_pair_root=None,
    srnet_cover_source="clean",
    export_original_recovered_root=None,
    visual_dir_override=None,
):
    ckpt_obj = safe_torch_load(str(ckpt_path), map_location="cpu")
    load_lora_state(lora_modules=lora_modules, ckpt_obj=ckpt_obj)
    load_writer_reader_state(writer, reader, ckpt_obj)
    load_discriminator_state(discriminator, ckpt_obj)

    stage_msb = int(ckpt_obj.get("stage_msb", ckpt_obj.get("payload_msb_bits", PAYLOAD_MSB_BITS)))
    prev_stage_msb = int(ckpt_obj.get("prev_stage_msb", 0))
    new_bits_weight = float(ckpt_obj.get("new_bits_weight", NEW_BITS_WEIGHT))
    eval_epoch = int(ckpt_obj.get("epoch", 0))
    eval_split = str(eval_split).strip().lower() or "val"
    eval_tag_parts = []
    if eval_split != "val":
        eval_tag_parts.append(eval_split)
    if no_knowledge_pipe is not None:
        eval_tag_parts.append(str(clean_generation_tag or "generated_clean"))
    eval_tag = "_".join(eval_tag_parts)
    visual_dir_name = "final_eval_vis" if len(eval_tag) == 0 else f"final_eval_vis_{eval_tag}"
    if visual_dir_override is not None:
        visual_dir = str(Path(visual_dir_override).expanduser().resolve())
    else:
        visual_dir = str(stage_dir / visual_dir_name)
    final_stats = eval_one_epoch(
        sd_vae=sd_vae,
        proj_head=proj_head,
        codebook_size=codebook_size,
        vq_codec=vq_codec,
        writer=writer,
        reader=reader,
        discriminator=discriminator,
        loader=eval_loader,
        device=device,
        epoch=eval_epoch,
        visual_dir=visual_dir,
        num_visual_batches=int(num_visual_batches),
        alpha=alpha,
        message_img_size=message_img_size,
        message_ch=bits_per_token,
        prev_payload_msb_bits=prev_stage_msb,
        new_bits_weight=new_bits_weight,
        payload_msb_bits=stage_msb,
        payload_stride=payload_stride,
        target_block_idxs=target_block_idxs,
        compute_perceptual=bool(FINAL_METRICS_COMPUTE_SSIM or FINAL_METRICS_COMPUTE_LPIPS),
        lpips_model=lpips_model if bool(FINAL_METRICS_COMPUTE_LPIPS) else None,
        max_eval_batches=int(EVAL_ONLY_MAX_VAL_BATCHES),
        no_knowledge_pipe=no_knowledge_pipe,
        no_knowledge_seed_base=no_knowledge_seed_base,
        no_knowledge_prompt=no_knowledge_prompt,
        no_knowledge_num_steps=no_knowledge_num_steps,
        no_knowledge_strength=no_knowledge_strength,
        no_knowledge_cfg=no_knowledge_cfg,
        srnet_pair_root=srnet_pair_root,
        srnet_cover_source=srnet_cover_source,
        export_original_recovered_root=export_original_recovered_root,
    )
    final_metrics = {
        "data_preset": str(DATA_PRESET),
        "eval_mode": str(eval_tag or "standard"),
        "eval_split": str(eval_split),
        "reduced_loss": True,
        "stage_msb": int(stage_msb),
        "lora_decoder_last_n_upblocks": int(LORA_DECODER_LAST_N_UPBLOCKS),
        "conditioner_decoder_last_n_upblocks": int(CONDITIONER_DECODER_LAST_N_UPBLOCKS),
        "ber": float(final_stats.get("ber", float("nan"))),
        "psnr_rec": float(final_stats.get("psnr_rec", final_stats.get("psnr", float("nan")))),
        "ssim_rec": float(final_stats.get("ssim_rec", float("nan"))),
        "lpips_rec": float(final_stats.get("lpips_rec", float("nan"))),
        "mse_rec": float(final_stats.get("mse_rec", float("nan"))),
        "psnr_stego": float(final_stats.get("psnr_stego", float("nan"))),
        "ssim_stego": float(final_stats.get("ssim_stego", float("nan"))),
        "lpips_stego": float(final_stats.get("lpips_stego", float("nan"))),
        "mse_stego": float(final_stats.get("mse_stego", float("nan"))),
        "best_ckpt": str(ckpt_path),
    }
    if no_knowledge_pipe is not None:
        final_metrics["generated_clean_mode"] = str(clean_generation_tag or "generated_clean")
        final_metrics["generated_clean_prompt"] = str(no_knowledge_prompt)
        final_metrics["generated_clean_seed_base"] = int(no_knowledge_seed_base)
        final_metrics["generated_clean_num_steps"] = int(no_knowledge_num_steps)
        final_metrics["generated_clean_strength"] = float(no_knowledge_strength)
        final_metrics["generated_clean_cfg"] = float(no_knowledge_cfg)
    if export_original_recovered_root is not None:
        final_metrics["eval_only_export_root"] = str(export_original_recovered_root)
        final_metrics["eval_only_exported_images"] = int(final_stats.get("exported_images", 0.0))
    stage_dir.mkdir(parents=True, exist_ok=True)
    final_json = stage_dir / ("final_metrics.json" if len(eval_tag) == 0 else f"final_metrics_{eval_tag}.json")
    with final_json.open("w") as f:
        json.dump(final_metrics, f, indent=2)

    ckpt_key = "final_metrics" if len(eval_tag) == 0 else f"final_metrics_{eval_tag}"
    ckpt_obj[ckpt_key] = final_metrics
    torch.save(ckpt_obj, str(ckpt_path))

    summary_csv = _resolve_summary_csv_path(tag_suffix=eval_tag)
    _append_summary_csv_row(summary_csv, final_metrics)
    print(
        "[FinalMetrics] "
        f"split={final_metrics['eval_split']} | "
        f"BER={final_metrics['ber']:.4f}, "
        f"rec(psnr/ssim/lpips/mse)={final_metrics['psnr_rec']:.2f}/{final_metrics['ssim_rec']:.4f}/{final_metrics['lpips_rec']:.4f}/{final_metrics['mse_rec']:.6f}, "
        f"stego(psnr/ssim/lpips/mse)={final_metrics['psnr_stego']:.2f}/{final_metrics['ssim_stego']:.4f}/{final_metrics['lpips_stego']:.4f}/{final_metrics['mse_stego']:.6f} | "
        f"saved={final_json} | summary={summary_csv}"
    )
    return final_metrics


# ============================================================
# Main
# ============================================================

def main():
    global EVAL_ONLY
    global EVAL_ONLY_CKPT_PATH
    global EVAL_ONLY_BATCH_SIZE
    global EVAL_ONLY_NUM_WORKERS
    global EVAL_ONLY_MAX_VAL_BATCHES
    global EVAL_ONLY_NUM_VISUAL_BATCHES
    global EVAL_ONLY_SPLIT
    global EVAL_ONLY_SAVE_SRNET_PAIRS
    global EVAL_ONLY_SRNET_ROOT
    global EVAL_ONLY_SRNET_COVER_SOURCE
    global EVAL_ONLY_NO_KNOWLEDGE
    global EVAL_ONLY_NO_KNOWLEDGE_SEED
    global EVAL_ONLY_VAL_ROOT
    global EVAL_ONLY_VAL_TAIL_N
    global EVAL_ONLY_GENERATE_CLEAN
    global USE_CUSTOM_UNET
    global CUSTOM_UNET_CKPT_PATH
    global CACHE_ROOT

    args = parse_args()
    apply_data_preset(args.source_preset)
    if len(str(args.cache_root).strip()) > 0:
        CACHE_ROOT = str(args.cache_root).strip()
    if len(str(args.custom_unet_ckpt_path).strip()) > 0:
        USE_CUSTOM_UNET = True
        CUSTOM_UNET_CKPT_PATH = str(args.custom_unet_ckpt_path).strip()
    if bool(args.eval_only):
        EVAL_ONLY = True
    if len(str(args.eval_only_ckpt_path).strip()) > 0:
        EVAL_ONLY_CKPT_PATH = str(args.eval_only_ckpt_path).strip()
    if int(args.eval_only_batch_size) > 0:
        EVAL_ONLY_BATCH_SIZE = int(args.eval_only_batch_size)
    if int(args.eval_only_num_workers) >= 0:
        EVAL_ONLY_NUM_WORKERS = int(args.eval_only_num_workers)
    if int(args.eval_only_max_val_batches) != -1:
        EVAL_ONLY_MAX_VAL_BATCHES = int(args.eval_only_max_val_batches)
    if int(args.eval_only_num_visual_batches) != -1:
        EVAL_ONLY_NUM_VISUAL_BATCHES = int(args.eval_only_num_visual_batches)
    EVAL_ONLY_SPLIT = str(args.eval_only_split).strip().lower()
    EVAL_ONLY_SAVE_SRNET_PAIRS = bool(args.eval_only_save_srnet_pairs)
    EVAL_ONLY_SRNET_ROOT = str(args.eval_only_srnet_root).strip()
    EVAL_ONLY_SRNET_COVER_SOURCE = str(args.eval_only_srnet_cover_source).strip().lower()
    EVAL_ONLY_NO_KNOWLEDGE = bool(args.no_knowledge)
    EVAL_ONLY_NO_KNOWLEDGE_SEED = int(args.no_knowledge_seed)
    EVAL_ONLY_VAL_ROOT = str(args.eval_only_val_root).strip()
    EVAL_ONLY_VAL_TAIL_N = int(args.eval_only_val_tail_n)
    if bool(EVAL_ONLY_NO_KNOWLEDGE) and not bool(EVAL_ONLY):
        raise RuntimeError("--no-knowledge currently only works together with --eval-only.")

    _refresh_conditioner_util_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if not bool(USE_WRITER_READER):
        raise RuntimeError("This script is locked to Writer+Reader+VAE LoRA mode. Set USE_WRITER_READER=True.")
    print("[Mode] Alternating Channel/Realism training + BER-constrained visual polish")
    print("[LossMode] reduced objective: keeping L_ref + L_constraint + L_vis_flat only.")
    if bool(EVAL_ONLY_NO_KNOWLEDGE):
        print("[EvalMode] no_knowledge enabled: eval-only clean latents/images will be regenerated with empty prompt and fresh seed.")

    img_size = 512
    batch_size = 2
    train_n = int(_ACTIVE_PRESET.get("train_n", 6000))

    alpha = ALPHA
    message_img_size = MESSAGE_IMG_SIZE
    payload_stride = PAYLOAD_STRIDE

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])

    val_root = None if len(str(VAL_ROOT).strip()) == 0 else VAL_ROOT
    try:
        train_set, val_set = build_train_val_sets(
            IMG_ROOT,
            CACHE_ROOT,
            transform,
            train_n,
            100000,
            val_root,
        )
    except TypeError as exc:
        if val_root is not None and "positional arguments" in str(exc):
            warnings.warn(
                "build_train_val_sets() does not support val_root in this environment; "
                "falling back to default train/val split."
            )
        train_set, val_set = build_train_val_sets(
            IMG_ROOT,
            CACHE_ROOT,
            transform,
            train_n,
            100000,
        )

    if bool(EVAL_ONLY) and len(str(EVAL_ONLY_VAL_ROOT).strip()) > 0:
        eval_files = collect_cached_image_files(
            img_root=EVAL_ONLY_VAL_ROOT,
            cache_root=CACHE_ROOT,
            limit=100000,
        )
        used_cache = len(eval_files) > 0
        if not used_cache:
            eval_files = collect_input_images(
                EVAL_ONLY_VAL_ROOT,
                exts=(".png", ".jpg", ".jpeg"),
                limit=100000,
            )
            EVAL_ONLY_GENERATE_CLEAN = True
        if int(EVAL_ONLY_VAL_TAIL_N) > 0:
            eval_files = eval_files[-int(EVAL_ONLY_VAL_TAIL_N):]
        if len(eval_files) == 0:
            raise RuntimeError(f"No eval image files found for --eval-only-val-root={EVAL_ONLY_VAL_ROOT}.")
        if used_cache:
            val_set = CachedMessageDataset(
                img_root=EVAL_ONLY_VAL_ROOT,
                cache_root=CACHE_ROOT,
                transform=transform,
                files_list=eval_files,
            )
        else:
            val_set = RawEvalMessageDataset(
                files_list=eval_files,
                transform=transform,
                image_size=img_size,
            )
        print(
            f"[EvalOnly] validation override: root={EVAL_ONLY_VAL_ROOT}, "
            f"tail_n={int(EVAL_ONLY_VAL_TAIL_N)}, size={len(val_set)}, "
            f"cache={'yes' if used_cache else 'no/on-the-fly SDI2I clean generation'}, cache_root={CACHE_ROOT}"
        )

    val_batch_size = int(EVAL_ONLY_BATCH_SIZE) if bool(EVAL_ONLY) and int(EVAL_ONLY_BATCH_SIZE) > 0 else batch_size
    val_num_workers = int(EVAL_ONLY_NUM_WORKERS) if bool(EVAL_ONLY) and int(EVAL_ONLY_NUM_WORKERS) >= 0 else 4
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_eval_loader = DataLoader(
        train_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=True,
    )

    # ========================================================
    # SD VAE (+ optional custom UNet loading in pipeline)
    # ========================================================
    pipe_ref = None
    if bool(USE_CUSTOM_UNET):
        pipe_ref = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        if len(str(CUSTOM_UNET_CKPT_PATH).strip()) > 0:
            custom_path = Path(str(CUSTOM_UNET_CKPT_PATH).strip())
            if not custom_path.is_absolute():
                custom_path = SCRIPT_DIR / custom_path

            if custom_path.is_dir():
                # Follow trainingfree style:
                #   1) if user gives parent dir, try parent/unet
                #   2) otherwise use dir directly
                unet_dir = custom_path / "unet" if (custom_path / "unet").is_dir() else custom_path
                try:
                    pipe_ref.unet = pipe_ref.unet.from_pretrained(
                        str(unet_dir),
                        subfolder=None,
                        use_safetensors=True,
                    ).to(device)
                except Exception:
                    # fallback for non-safetensors checkpoints
                    pipe_ref.unet = pipe_ref.unet.from_pretrained(
                        str(unet_dir),
                        subfolder=None,
                        use_safetensors=False,
                    ).to(device)
                print(f"[UNet] loaded from diffusers dir: {unet_dir}")
            else:
                unet_ckpt = safe_torch_load(str(custom_path), map_location="cpu")
                unet_state = _extract_unet_state_dict(unet_ckpt)
                missing, unexpected = pipe_ref.unet.load_state_dict(unet_state, strict=False)
                print(
                    f"[UNet] loaded from file: {custom_path} | "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )
        else:
            print("[UNet] USE_CUSTOM_UNET=True but CUSTOM_UNET_CKPT_PATH is empty; using base UNet.")
        sd_vae = pipe_ref.vae
        print(
            "[Note] This script optimizes VAE LoRA (+ optional writer/reader); UNet weights are loaded for consistency "
            "with your SD checkpoint but are not used in the roundtrip forward path."
        )
    else:
        sd_vae = AutoencoderKL.from_pretrained(
            SD_MODEL_ID,
            subfolder="vae",
            # LoRA optimization on VAE is numerically fragile in fp16; keep fp32 for stability.
            torch_dtype=torch.float32,
        ).to(device)

    sd_vae.eval()
    for p in sd_vae.parameters():
        p.requires_grad = False

    # ========================================================
    # frozen VQ teacher + codec
    # ========================================================
    vq_raw = load_vqgan_taming(VQ_CONFIG_PATH, VQ_CKPT_PATH, device=device)
    codebook_size = int(vq_raw.quantize.n_e) if hasattr(vq_raw, "quantize") else 16384
    vq_codec = VQCodec(vq_model=vq_raw, codebook_size=codebook_size)
    bits_per_token = int(math.ceil(math.log2(codebook_size)))
    msb_schedule = build_msb_schedule(bits_per_token)
    if int(START_FROM_MSB) > 1:
        msb_schedule = [m for m in msb_schedule if int(m) >= int(START_FROM_MSB)]
    if len(msb_schedule) == 0:
        raise RuntimeError(f"Empty MSB schedule after START_FROM_MSB={START_FROM_MSB}")
    first_stage_msb = int(msb_schedule[0])

    # ========================================================
    # frozen projection head (VAE-encoder-style distilled)
    # ========================================================
    proj_head = VAEEncoderStyleVQHead(
        codebook_size=codebook_size,
        sd_model_id=SD_MODEL_ID,
        proj_ch=int(PROJ_HEAD_CH),
        freeze_backbone=bool(PROJ_HEAD_FREEZE_BACKBONE),
        train_last_n_downblocks=int(PROJ_HEAD_TRAIN_LAST_N_DOWNBLOCKS),
        train_mid_block=bool(PROJ_HEAD_TRAIN_MID_BLOCK),
    ).to(device)
    ckpt = safe_torch_load(PROJ_HEAD_CKPT, map_location="cpu")
    ph_state = ckpt["student_head"] if isinstance(ckpt, dict) and "student_head" in ckpt else ckpt
    proj_head.load_state_dict(ph_state, strict=True)
    proj_head.eval()
    for p in proj_head.parameters():
        p.requires_grad = False

    # infer token map + latent shape
    with torch.no_grad():
        x0, z0, _, _ = next(iter(train_loader))
        x0 = x0.to(device)
        z0 = z0.to(device)
        x0_msg = F.interpolate(
            x0[:1].float(),
            size=(int(message_img_size), int(message_img_size)),
            mode="bilinear",
            align_corners=False,
        )
        logits0 = proj_head(x0_msg)
        idx0 = torch.argmax(logits0, dim=1)
        Hm, Wm = int(idx0.shape[-2]), int(idx0.shape[-1])
        Nt = int(Hm * Wm)
        lat_c = int(z0.shape[1])
        lat_h = int(z0.shape[-2])
        lat_w = int(z0.shape[-1])
        print(
            f"Message branch: msg_img={message_img_size}, map={Hm}x{Wm}, "
            f"Nt={Nt}, bits_per_token={bits_per_token}"
        )
        print(
            f"Writer/Reader channel: latent={lat_c}x{lat_h}x{lat_w}, "
            f"writer_hidden={WRITER_HIDDEN}, reader_hidden={READER_HIDDEN}"
        )
        print(f"MSB schedule: {msb_schedule}")

    # ========================================================
    # trainable components
    # ========================================================
    for p in sd_vae.parameters():
        p.requires_grad = False
    lora_modules = inject_vae_lora(sd_vae)
    lora_params = [p for m in lora_modules.values() for p in m.parameters() if p.requires_grad]
    if len(lora_params) == 0:
        raise RuntimeError("No trainable LoRA parameters were injected into VAE.")
    print(
        f"[LoRA] modules={len(lora_modules)}, trainable_params={sum(p.numel() for p in lora_params)}, "
        f"rank={LORA_RANK}, alpha={LORA_ALPHA}, lr={LORA_LR}"
    )

    target_block_idxs, target_channels = infer_decoder_film_targets(
        sd_vae=sd_vae,
        last_n_upblocks=int(CONDITIONER_DECODER_LAST_N_UPBLOCKS),
    )
    print(
        f"[Conditioner] target_up_blocks={target_block_idxs}, target_channels={target_channels}, "
        f"gamma_scale={CONDITIONER_GAMMA_SCALE}, beta_scale={CONDITIONER_BETA_SCALE}"
    )

    writer = Writer(
        m_ch=bits_per_token,
        target_channels=target_channels,
        z_ch=lat_c,
        hidden=int(WRITER_HIDDEN),
        gamma_scale=float(CONDITIONER_GAMMA_SCALE),
        beta_scale=float(CONDITIONER_BETA_SCALE),
    ).to(device)

    out_hs = (Hm + max(1, int(payload_stride)) - 1) // max(1, int(payload_stride))
    out_ws = (Wm + max(1, int(payload_stride)) - 1) // max(1, int(payload_stride))
    reader = Reader(
        m_ch=bits_per_token,
        out_hw=(out_hs, out_ws),
        in_ch=lat_c,
        hidden=int(READER_HIDDEN),
    ).to(device)
    writer_params = [p for p in writer.parameters() if p.requires_grad]
    reader_params = [p for p in reader.parameters() if p.requires_grad]
    wr_params = [p for p in list(writer_params) + list(reader_params) if p.requires_grad]
    print(
        f"[WR] aux_logits={bool(WRITER_AUX_LOGITS)}, "
        f"trainable_params={sum(p.numel() for p in wr_params)}, lr={WRITER_READER_LR}, wd={WRITER_READER_WEIGHT_DECAY}"
    )

    discriminator = None
    disc_params = []

    ckpt_root = SCRIPT_DIR / f"ckpt/{_artifact_tag()}/{_polish_dirname()}"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    vis_root = SCRIPT_DIR / f"vis/{_artifact_tag()}/{_polish_dirname()}"
    vis_root.mkdir(parents=True, exist_ok=True)
    prev_best_path = None
    lpips_model = build_lpips_model(device) if bool(FINAL_METRICS_COMPUTE_LPIPS) else None
    effective_realism_w = get_realism_weights(epoch=max(1, int(MSG_WARMUP_EPOCHS) + 1))
    effective_constraint_w = get_constraint_weights()

    if len(RESUME_CKPT_PATH.strip()) > 0:
        resume_p = Path(RESUME_CKPT_PATH.strip())
        if not resume_p.is_absolute():
            resume_p = SCRIPT_DIR / resume_p
        if resume_p.exists():
            prev_best_path = str(resume_p)
            print(f"[Resume] Using explicit checkpoint: {prev_best_path}")
        else:
            print(f"[Resume] checkpoint not found, ignore: {resume_p}")
    elif bool(AUTO_RESUME_FIRST_STAGE_BEST):
        auto_p = ckpt_root / f"msb_{first_stage_msb:02d}" / "best.pt"
        if auto_p.exists():
            prev_best_path = str(auto_p)
            print(f"[Resume] Auto-loaded first stage best: {prev_best_path}")

    if bool(EVAL_ONLY):
        eval_ckpt = str(EVAL_ONLY_CKPT_PATH).strip() or str(RESUME_CKPT_PATH).strip()
        if len(eval_ckpt) == 0:
            raise RuntimeError("EVAL_ONLY=True but no checkpoint path provided (EVAL_ONLY_CKPT_PATH/RESUME_CKPT_PATH).")
        eval_ckpt_p = Path(eval_ckpt)
        if not eval_ckpt_p.is_absolute():
            eval_ckpt_p = SCRIPT_DIR / eval_ckpt_p
        if not eval_ckpt_p.exists():
            raise RuntimeError(f"EVAL_ONLY checkpoint not found: {eval_ckpt_p}")
        eval_obj = safe_torch_load(str(eval_ckpt_p), map_location="cpu")
        eval_stage_msb = int(eval_obj.get("stage_msb", eval_obj.get("payload_msb_bits", PAYLOAD_MSB_BITS)))
        eval_stage_dir = ckpt_root / f"msb_{eval_stage_msb:02d}"
        no_knowledge_pipe = None
        no_knowledge_seed_base = None
        no_knowledge_prompt = ""
        no_knowledge_num_steps = int(_ACTIVE_PRESET.get("num_steps", 20))
        no_knowledge_strength = float(_ACTIVE_PRESET.get("strength", 0.75))
        no_knowledge_cfg = float(_ACTIVE_PRESET.get("cfg", 2.0))
        eval_split = str(EVAL_ONLY_SPLIT).strip().lower() or "val"
        eval_loader = train_eval_loader if eval_split == "train" else val_loader
        eval_tag_parts = []
        if eval_split != "val":
            eval_tag_parts.append(eval_split)
        if bool(EVAL_ONLY_NO_KNOWLEDGE):
            eval_tag_parts.append("no_knowledge")
        elif bool(EVAL_ONLY_GENERATE_CLEAN):
            eval_tag_parts.append("on_the_fly_clean")
        eval_tag = "_".join(eval_tag_parts)
        export_root = (Path(EVAL_ONLY_EXPORT_ROOT_DEFAULT).expanduser().resolve() / _artifact_tag()).resolve()
        eval_visual_root = (Path(EVAL_ONLY_VISUAL_ROOT_DEFAULT).expanduser().resolve() / _artifact_tag() / _polish_dirname()).resolve()
        eval_visual_dir = (eval_visual_root if len(eval_tag) == 0 else (eval_visual_root / f"final_eval_vis_{eval_tag}")).resolve()
        if len(EVAL_ONLY_SRNET_ROOT) > 0:
            srnet_pair_root = Path(EVAL_ONLY_SRNET_ROOT).expanduser().resolve()
        else:
            srnet_dir_name = "srnet_pairs" if len(eval_tag) == 0 else f"srnet_pairs_{eval_tag}"
            srnet_pair_root = eval_stage_dir / srnet_dir_name
        if not bool(EVAL_ONLY_SAVE_SRNET_PAIRS):
            srnet_pair_root = None
        if bool(EVAL_ONLY_NO_KNOWLEDGE) or bool(EVAL_ONLY_GENERATE_CLEAN):
            no_knowledge_prompt = "" if bool(EVAL_ONLY_NO_KNOWLEDGE) else str(
                _ACTIVE_PRESET.get("prompt", "a dermoscopic photo of a skin lesion with natural skin texture")
            )
            no_knowledge_seed_base = (
                int(EVAL_ONLY_NO_KNOWLEDGE_SEED)
                if int(EVAL_ONLY_NO_KNOWLEDGE_SEED) >= 0
                else int(torch.randint(0, 2**31 - 1, (1,)).item())
            )
            if bool(EVAL_ONLY_GENERATE_CLEAN) and not bool(EVAL_ONLY_NO_KNOWLEDGE):
                # Use a fresh SDI2I pipeline for clean generation so the eval
                # checkpoint's VAE LoRA is not applied while creating y_clean.
                no_knowledge_pipe = _build_eval_img2img_pipe(device)
            else:
                no_knowledge_pipe = pipe_ref if pipe_ref is not None else _build_eval_img2img_pipe(device)
            if bool(EVAL_ONLY_GENERATE_CLEAN) and not bool(EVAL_ONLY_NO_KNOWLEDGE):
                no_knowledge_seed_base = int(_ACTIVE_PRESET.get("seed", 12345))
        print(
            f"[EvalOnly] ckpt={eval_ckpt_p} | split={eval_split} | batch_size={val_batch_size}, "
            f"num_workers={val_num_workers}, max_batches={int(EVAL_ONLY_MAX_VAL_BATCHES)}, "
            f"num_visual_batches={int(EVAL_ONLY_NUM_VISUAL_BATCHES)}, "
            f"srnet_cover_source={EVAL_ONLY_SRNET_COVER_SOURCE}, "
            f"export_root={export_root}, visual_dir={eval_visual_dir}"
        )
        if no_knowledge_pipe is not None:
            mode_label = "no_knowledge" if bool(EVAL_ONLY_NO_KNOWLEDGE) else "on_the_fly_clean"
            print(
                f"[EvalOnly:{mode_label}] prompt='{str(no_knowledge_prompt)}' | seed_base={int(no_knowledge_seed_base)} | "
                f"num_steps={int(no_knowledge_num_steps)}, strength={float(no_knowledge_strength):.3f}, cfg={float(no_knowledge_cfg):.3f}"
            )
        evaluate_final_metrics_for_ckpt(
            ckpt_path=eval_ckpt_p,
            stage_dir=eval_stage_dir,
            lora_modules=lora_modules,
            sd_vae=sd_vae,
            proj_head=proj_head,
            codebook_size=codebook_size,
            vq_codec=vq_codec,
            writer=writer,
            reader=reader,
            discriminator=discriminator,
            eval_loader=eval_loader,
            device=device,
            alpha=alpha,
            message_img_size=message_img_size,
            bits_per_token=bits_per_token,
            payload_stride=payload_stride,
            target_block_idxs=target_block_idxs,
            eval_split=eval_split,
            lpips_model=lpips_model,
            num_visual_batches=int(EVAL_ONLY_NUM_VISUAL_BATCHES),
            no_knowledge_pipe=no_knowledge_pipe,
            no_knowledge_seed_base=no_knowledge_seed_base,
            no_knowledge_prompt=no_knowledge_prompt,
            no_knowledge_num_steps=no_knowledge_num_steps,
            no_knowledge_strength=no_knowledge_strength,
            no_knowledge_cfg=no_knowledge_cfg,
            clean_generation_tag="no_knowledge" if bool(EVAL_ONLY_NO_KNOWLEDGE) else ("on_the_fly_clean" if bool(EVAL_ONLY_GENERATE_CLEAN) else "generated_clean"),
            srnet_pair_root=srnet_pair_root,
            srnet_cover_source=EVAL_ONLY_SRNET_COVER_SOURCE,
            export_original_recovered_root=export_root,
            visual_dir_override=eval_visual_dir,
        )
        if int(EVAL_ONLY_NUM_VISUAL_BATCHES) > 0:
            print(f"[EvalOnly] 4-panel visuals saved under: {eval_visual_dir}")
        if srnet_pair_root is not None:
            print(f"[EvalOnly] SRNet pairs saved under: {srnet_pair_root}")
        print(f"[EvalOnly] original/recover exports saved under: {export_root}")
        return

    for stage_idx, stage_msb in enumerate(msb_schedule, start=1):
        stage_msb = int(stage_msb)
        prev_stage_msb = int(msb_schedule[stage_idx - 2]) if stage_idx > 1 else 0
        stage_new_bits_w = float(NEW_BITS_WEIGHT) if prev_stage_msb < stage_msb else 1.0
        stage_dir = ckpt_root / f"msb_{stage_msb:02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        stage_bits = int(Nt * stage_msb)
        print(f"[Stage {stage_idx}] MSB={stage_msb}: payload_bits={stage_bits}, mode=conditioner_reader")

        if prev_best_path is not None and Path(prev_best_path).exists():
            prev = safe_torch_load(prev_best_path, map_location="cpu")
            load_lora_state(lora_modules, prev)
            load_writer_reader_state(writer, reader, prev)
            load_discriminator_state(discriminator, prev)
            print(f"[Stage {stage_idx}] Loaded previous best: {prev_best_path}")

        vis_groups = [{"params": lora_params, "lr": float(LORA_LR)}]
        if len(writer_params) > 0:
            vis_groups.append(
                {
                    "params": writer_params,
                    "lr": float(WRITER_READER_LR),
                    "weight_decay": float(WRITER_READER_WEIGHT_DECAY),
                }
            )
        optimizer_vis = torch.optim.Adam(vis_groups)
        optimizer_reader = torch.optim.Adam(
            [
                {
                    "params": reader_params,
                    "lr": float(WRITER_READER_LR),
                    "weight_decay": float(WRITER_READER_WEIGHT_DECAY),
                }
            ]
        )
        optim_d = None

        best_val = 1e9
        best_gate_fallback = 1e9
        best_fallback_loss = 1e9
        best_has_qualified_ckpt = False
        best_path = stage_dir / "best.pt"
        stage_epochs = get_stage_epochs(stage_msb)
        max_epochs = int(stage_epochs + STAGE_MAX_EXTRA_EPOCHS)
        gate_key = "ber_all"
        stage_passed = False

        print(
            f"\n===== Stage {stage_idx}/{len(msb_schedule)} | "
            f"MSB={stage_msb} | prev={prev_stage_msb} | epochs={stage_epochs}(+{STAGE_MAX_EXTRA_EPOCHS}) | "
            f"new_bits_w={stage_new_bits_w:.1f} | gate={gate_key}<={BER_NEW_ADVANCE_THRESHOLD:.3f} ====="
        )
        for epoch in range(1, max_epochs + 1):
            train_stats = train_one_epoch(
                sd_vae=sd_vae,
                proj_head=proj_head,
                codebook_size=codebook_size,
                vq_codec=vq_codec,
                writer=writer,
                reader=reader,
                discriminator=discriminator,
                loader=train_loader,
                optimizer_vis=optimizer_vis,
                optimizer_reader=optimizer_reader,
                optim_d=optim_d,
                device=device,
                epoch=epoch,
                alpha=alpha,
                message_img_size=message_img_size,
                message_ch=bits_per_token,
                prev_payload_msb_bits=prev_stage_msb,
                new_bits_weight=stage_new_bits_w,
                payload_msb_bits=stage_msb,
                payload_stride=payload_stride,
                target_block_idxs=target_block_idxs,
            )

            val_stats = eval_one_epoch(
                sd_vae=sd_vae,
                proj_head=proj_head,
                codebook_size=codebook_size,
                vq_codec=vq_codec,
                writer=writer,
                reader=reader,
                discriminator=discriminator,
                loader=val_loader,
                device=device,
                epoch=epoch,
                visual_dir=str(vis_root / f"msb_{stage_msb:02d}"),
                num_visual_batches=1,
                alpha=alpha,
                message_img_size=message_img_size,
                message_ch=bits_per_token,
                prev_payload_msb_bits=prev_stage_msb,
                new_bits_weight=stage_new_bits_w,
                payload_msb_bits=stage_msb,
                payload_stride=payload_stride,
                target_block_idxs=target_block_idxs,
            )

            print(
                f"[Stage {stage_idx} | MSB {stage_msb} | Epoch {epoch}] "
                f"train loss={train_stats['loss']:.4f}, ber={train_stats['ber']:.4f}, old={train_stats['ber_old']:.4f}, new={train_stats['ber_new']:.4f}, "
                f"sb={train_stats.get('ber_soft', train_stats['ber']):.4f}, ms={train_stats.get('msg_scale', 1.0):.3f}, "
                f"vis={train_stats.get('loss_vis', 0.0):.4f}, vl={train_stats.get('loss_vis_lowfreq', 0.0):.4f}, vc={train_stats.get('loss_vis_chroma', 0.0):.4f}, vf={train_stats.get('loss_vis_flat', 0.0):.4f}, "
                f"bsp={train_stats.get('loss_ber_soft_pen', 0.0):.4f}, rtt={train_stats.get('loss_rt_trig', 0.0):.4f}, ch={train_stats.get('loss_channel', 0.0):.4f}, "
                f"rber={train_stats.get('ber_res', 0.0):.4f}, tok={train_stats['token_acc']:.4f}, tokc={train_stats.get('token_acc_coarse', 0.0):.4f}, "
                f"ps={train_stats.get('psnr_stego', 0.0):.2f} || "
                f"val loss={val_stats['loss']:.4f}, ber={val_stats['ber']:.4f}, old={val_stats['ber_old']:.4f}, new={val_stats['ber_new']:.4f}, "
                f"sb={val_stats.get('ber_soft', val_stats['ber']):.4f}, ms={val_stats.get('msg_scale', 1.0):.3f}, "
                f"vis={val_stats.get('loss_vis', 0.0):.4f}, vl={val_stats.get('loss_vis_lowfreq', 0.0):.4f}, vc={val_stats.get('loss_vis_chroma', 0.0):.4f}, vf={val_stats.get('loss_vis_flat', 0.0):.4f}, "
                f"bsp={val_stats.get('loss_ber_soft_pen', 0.0):.4f}, rtt={val_stats.get('loss_rt_trig', 0.0):.4f}, ch={val_stats.get('loss_channel', 0.0):.4f}, "
                f"rber={val_stats.get('ber_res', 0.0):.4f}, tok={val_stats['token_acc']:.4f}, tokc={val_stats.get('token_acc_coarse', 0.0):.4f}, "
                f"ps={val_stats.get('psnr_stego', 0.0):.2f}"
            )

            if stage_msb > prev_stage_msb:
                bit_keys = [f"ber_b{i:02d}" for i in range(prev_stage_msb + 1, stage_msb + 1)]
                bit_vals = [val_stats[k] for k in bit_keys if k in val_stats]
                if len(bit_vals) > 0:
                    bit_msg = ", ".join([f"b{i:02d}={val_stats[f'ber_b{i:02d}']:.3f}" for i in range(prev_stage_msb + 1, stage_msb + 1) if f"ber_b{i:02d}" in val_stats])
                    print(f"  [Val NewBits] {bit_msg}")

            save_obj = {
                "vae_lora": dump_lora_state(lora_modules),
                "writer": writer.state_dict() if writer is not None else None,
                "reader": reader.state_dict() if reader is not None else None,
                "discriminator": None,
                "epoch": epoch,
                "reduced_loss": True,
                "disabled_losses": [
                    "L_adv",
                    "L_ber-hard",
                    "L_tv",
                    "L_freq",
                    "L_vis_lowfreq",
                    "L_vis_chroma",
                ],
                "stage_idx": stage_idx,
                "stage_msb": stage_msb,
                "prev_stage_msb": prev_stage_msb,
                "new_bits_weight": stage_new_bits_w,
                "train_stats": train_stats,
                "val_stats": val_stats,
                "alpha": alpha,
                "message_img_size": message_img_size,
                "message_ch": bits_per_token,
                "payload_msb_bits": stage_msb,
                "payload_stride": payload_stride,
                "use_writer_reader": True,
                "writer_arch": str(WRITER_ARCH),
                "writer_aux_logits": bool(WRITER_AUX_LOGITS),
                "writer_hidden": int(WRITER_HIDDEN),
                "reader_hidden": int(READER_HIDDEN),
                "writer_reader_lr": float(WRITER_READER_LR),
                "effective_w_tv_scale": 0.0,
                "effective_w_tv_main": float(effective_realism_w["w_tv"]),
                "channel_step_interval": int(CHANNEL_STEP_INTERVAL),
                "w_vis_flat": float(W_VIS_FLAT),
                "vis_edge_scale": float(VIS_EDGE_SCALE),
                "best_ckpt_ber_max": float(BEST_CKPT_BER_MAX),
                "ber_budget_enable": bool(BER_BUDGET_ENABLE),
                "ber_target": float(BER_TARGET),
                "ber_hard_max": float(BER_HARD_MAX),
                "ber_msg_min_scale": float(BER_MSG_MIN_SCALE),
                "ber_budget_power": float(BER_BUDGET_POWER),
                "ber_hard_penalty": 0.0,
                "ber_soft_penalty": float(CONSTRAINT_BER_SOFT_PENALTY),
                "effective_ber_soft_penalty": float(effective_constraint_w["w_ber_soft"]),
                "ber_rt_trigger": float(BER_RT_TRIGGER),
                "ber_rt_penalty": float(CONSTRAINT_RT_PENALTY),
                "effective_ber_rt_penalty": float(effective_constraint_w["w_rt_trig"]),
                "conditioner_decoder_last_n_upblocks": int(CONDITIONER_DECODER_LAST_N_UPBLOCKS),
                "conditioner_gamma_scale": float(CONDITIONER_GAMMA_SCALE),
                "conditioner_beta_scale": float(CONDITIONER_BETA_SCALE),
                "conditioner_target_block_idxs": [int(v) for v in target_block_idxs],
                "conditioner_target_channels": [int(v) for v in target_channels],
                "proj_head_ckpt": str(PROJ_HEAD_CKPT),
                "proj_head_type": "VAEEncoderStyleVQHead",
                "proj_head_ch": int(PROJ_HEAD_CH),
                "lora_rank": int(LORA_RANK),
                "lora_alpha": float(LORA_ALPHA),
                "lora_lr": float(LORA_LR),
                "lora_decoder_last_n_upblocks": int(LORA_DECODER_LAST_N_UPBLOCKS),
                "lora_encoder_first_n_downblocks": int(LORA_ENCODER_FIRST_N_DOWNBLOCKS),
                "msb_schedule": msb_schedule,
            }
            torch.save(save_obj, stage_dir / "latest.pt")

            gate_val = float(
                max(
                    val_stats.get("ber", 1.0),
                    val_stats.get("ber_old", 0.0 if prev_stage_msb == 0 else 1.0),
                    val_stats.get("ber_new", 1.0),
                )
            )
            if gate_val <= float(BEST_CKPT_BER_MAX):
                if (not best_has_qualified_ckpt) or (val_stats["loss"] < best_val):
                    best_val = float(val_stats["loss"])
                    best_has_qualified_ckpt = True
                    torch.save(save_obj, best_path)
                    print(
                        f"  [Best] strict BER save at epoch {epoch}: "
                        f"gate={gate_val:.4f} <= {BEST_CKPT_BER_MAX:.4f}, loss={best_val:.4f}"
                    )
            elif (not best_has_qualified_ckpt) and (
                (gate_val < best_gate_fallback)
                or (math.isclose(gate_val, best_gate_fallback) and val_stats["loss"] < best_fallback_loss)
            ):
                best_gate_fallback = gate_val
                best_fallback_loss = float(val_stats["loss"])
                torch.save(save_obj, best_path)
                print(
                    f"  [Best] fallback save at epoch {epoch}: "
                    f"gate={gate_val:.4f}, loss={best_fallback_loss:.4f} "
                    f"(waiting for strict target <= {BEST_CKPT_BER_MAX:.4f})"
                )
            if gate_val <= float(BER_NEW_ADVANCE_THRESHOLD):
                stage_passed = True
                if bool(STAGE_EARLY_ADVANCE):
                    print(f"  [Gate] pass at epoch {epoch}: {gate_key}={gate_val:.4f}, advance to next stage.")
                    break
            if epoch >= int(stage_epochs) and stage_passed:
                print(f"  [Gate] pass at epoch {epoch}: {gate_key}={gate_val:.4f}, finish stage.")
                break

            if epoch == int(stage_epochs) and not stage_passed and int(STAGE_MAX_EXTRA_EPOCHS) > 0:
                print(
                    f"  [Gate] not passed after base epochs: {gate_key}={gate_val:.4f}, "
                    f"ber={val_stats.get('ber', 1.0):.4f}, old={val_stats.get('ber_old', 1.0):.4f}, new={val_stats.get('ber_new', 1.0):.4f}. "
                    f"Continue extra epochs up to {STAGE_MAX_EXTRA_EPOCHS}."
                )

        prev_best_path = str(best_path)
        if best_path.exists():
            evaluate_final_metrics_for_ckpt(
                ckpt_path=best_path,
                stage_dir=stage_dir,
                lora_modules=lora_modules,
                sd_vae=sd_vae,
                proj_head=proj_head,
                codebook_size=codebook_size,
                vq_codec=vq_codec,
                writer=writer,
                reader=reader,
                discriminator=discriminator,
                eval_loader=val_loader,
                device=device,
                alpha=alpha,
                message_img_size=message_img_size,
                bits_per_token=bits_per_token,
                payload_stride=payload_stride,
                target_block_idxs=target_block_idxs,
                lpips_model=lpips_model,
            )

        if bool(STAGE_STRICT_ADVANCE) and not stage_passed:
            print(
                f"[Gate] stop escalation at MSB={stage_msb}: "
                f"target {gate_key}<={BER_NEW_ADVANCE_THRESHOLD:.3f} not reached."
            )
            break


if __name__ == "__main__":
    main()
