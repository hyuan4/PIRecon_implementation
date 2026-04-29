import os
from pathlib import Path
import sys
import math
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
    decode_teacher_quant,
    decode_with_conditioner,
    dump_lora_state,
    get_frozen_message,
    get_stage_epochs,
    grad_loss,
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

# ============================================================
# Paths: change these if needed
# ============================================================

# dataset/model preset selector
DATA_PRESET = "lyme"
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
def apply_data_preset(name: str):
    global DATA_PRESET
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

    if name not in DATA_PRESETS:
        raise ValueError(f"Unknown DATA_PRESET={name}. Available: {list(DATA_PRESETS.keys())}")

    DATA_PRESET = str(name)
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
# PAYLOAD_MSB_BITS <= 0 means full bits-per-token (e.g., 14 for K=16384)
PAYLOAD_MSB_BITS = 14
PAYLOAD_STRIDE = 1
MSB_STAGE_START = 7
MSB_STAGE_STEP = 2
MSB_STAGE_EPOCHS = 4
# For direct MSB14 run: set both to 14 -> MSB_MANUAL_SCHEDULE=[14], START_FROM_MSB=14
MSB_MANUAL_SCHEDULE = [7, 10, 12, 14]
MSB_STAGE_EPOCHS_MAP = {
    7: 6,
    10: 8,
    12: 10,
    14: 12,
}

# stronger channel settings for first runnable convergence
ALPHA = 1.0
MSG_WARMUP_EPOCHS = 6
# Previous defaults:
#   warmup = {"w_msg": 8.0, "w_stego": 0.05, "w_rt": 0.20}
#   main   = {"w_msg": 6.0, "w_stego": 0.08, "w_rt": 0.30}
LOSS_WEIGHTS_WARMUP = {"w_msg": 8.0, "w_stego": 0.05, "w_rt": 0.20}
LOSS_WEIGHTS_MAIN = {"w_msg": 6.0, "w_stego": 0.08, "w_rt": 0.30}
NEW_BITS_WEIGHT = 5.0

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
LORA_LR = 1e-5
LORA_GRAD_CLIP = 1.0
LORA_DECODER_LAST_N_UPBLOCKS = 2
LORA_ENCODER_FIRST_N_DOWNBLOCKS = 2
LORA_INCLUDE_CONV_IO = True

# writer/reader settings (train with VAE LoRA)
USE_WRITER_READER = True
WRITER_HIDDEN = 128
READER_HIDDEN = 128
WRITER_READER_LR = 1e-4
WRITER_READER_WEIGHT_DECAY = 0.0
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
# Advance to next MSB immediately once BER gate is satisfied.
STAGE_EARLY_ADVANCE = True

# stage resume controls
START_FROM_MSB = 7
# For this new architecture, start from MSB7 curriculum by default.
# Optional warm-start: load LoRA/reader from previous MSB14 WR checkpoint.
RESUME_CKPT_PATH = f"ckpt/{DATA_PRESET}/base/msb_14/best.pt"
AUTO_RESUME_FIRST_STAGE_BEST = False
# Continue exactly from a mid-stage checkpoint (for example msb_14/latest.pt).
# When enabled, stage_msb / prev_stage_msb / new_bits_weight / epoch are restored
# from RESUME_CKPT_PATH instead of inferred from START_FROM_MSB.
RESUME_MID_STAGE = True
# Absolute epoch index within the resumed stage to stop at. 0 keeps default stage budget.
RESUME_MID_STAGE_END_EPOCH = 16

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
    film_pairs = writer(m, z_for_writer)
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
        "delta": torch.zeros_like(z_stego, dtype=torch.float32),
    }


# ============================================================
# Loss
# ============================================================

def compute_losses(outputs, w_msg=1.0, w_stego=0.1, w_rt=0.2, prev_active_bits=0, new_bits_weight=1.0):
    bits = outputs["bits"]                 # (B,L) float(0/1)
    bit_logits = outputs["bit_logits"]     # (B,L) logits
    bits_hat = outputs["bits_hat"]         # (B,L) float(0/1)
    indices = outputs["indices"]
    indices_hat = outputs["indices_hat"]
    x_msg = outputs["x_msg"]
    x_rec = outputs["x_rec"]
    y_clean = outputs["y_clean"]
    y_stego = outputs["y_stego"]
    z_stego = outputs["z_stego"]
    z_rt = outputs["z_rt"]

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

    loss_stego = F.l1_loss(y_stego, y_clean)
    loss_rt = F.l1_loss(z_rt.float(), z_stego.float().detach())
    loss_img = F.l1_loss(x_rec, x_msg)
    loss_grad = grad_loss(x_rec, x_msg)

    ber = (hats_act - bits_act).abs().mean().item()
    ber_old = (hats_act[:, :prev_len] - bits_act[:, :prev_len]).abs().mean().item() if prev_len > 0 else 0.0
    ber_new = (hats_act[:, prev_len:active_len] - bits_act[:, prev_len:active_len]).abs().mean().item() if prev_len < active_len else 0.0
    token_acc = (indices_hat == indices).float().mean().item()
    if active_bits < bits_per_token:
        low_bits = int(bits_per_token - active_bits)
        token_acc_coarse = (((indices_hat >> low_bits) == (indices >> low_bits)).float().mean().item())
    else:
        token_acc_coarse = token_acc

    loss = w_msg * loss_msg + w_stego * loss_stego + w_rt * loss_rt

    metrics = {
        "loss": loss.item(),
        "loss_msg": loss_msg.item(),
        "loss_img": loss_img.item(),
        "loss_grad": loss_grad.item(),
        "loss_stego": loss_stego.item(),
        "loss_rt": loss_rt.item(),
        "ber": ber,
        "ber_old": ber_old,
        "ber_new": ber_new,
        "ber_res": 0.0,
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
    loader,
    optimizer,
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
    loss_w = get_loss_weights(epoch)

    running = {
        "loss": 0.0,
        "loss_msg": 0.0,
        "loss_img": 0.0,
        "loss_grad": 0.0,
        "loss_stego": 0.0,
        "loss_rt": 0.0,
        "psnr": 0.0,
        "ber": 0.0,
        "ber_old": 0.0,
        "ber_new": 0.0,
        "ber_res": 0.0,
        "token_acc": 0.0,
        "token_acc_coarse": 0.0,
    }

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for step, (x, z_clean, y_clean, _) in enumerate(pbar):
        x = x.to(device)
        z_clean = z_clean.to(device)
        y_clean = y_clean.to(device)

        optimizer.zero_grad()

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

        loss, metrics = compute_losses(
            outputs,
            prev_active_bits=prev_payload_msb_bits,
            new_bits_weight=new_bits_weight,
            **loss_w,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected at epoch={epoch}, step={step}. "
                f"Try lowering learning rates (LORA_LR/WRITER_READER_LR) and payload MSB."
            )

        loss.backward()

        if float(LORA_GRAD_CLIP) > 0:
            clip_params = []
            for group in optimizer.param_groups:
                clip_params.extend(group["params"])
            torch.nn.utils.clip_grad_norm_(clip_params, float(LORA_GRAD_CLIP))

        optimizer.step()

        with torch.no_grad():
            x01 = ((outputs["x_msg"] + 1) * 0.5).clamp(0, 1)
            xrec01 = ((outputs["x_rec"] + 1) * 0.5).clamp(0, 1)
            metrics["psnr"] = psnr_torch(x01, xrec01)

        for k, v in metrics.items():
            if k not in running:
                running[k] = 0.0
            running[k] += v

        denom = step + 1
        pbar.set_postfix_str(
            f"L={running['loss']/denom:.3f} | "
            f"M={running['loss_msg']/denom:.3f} | "
            f"RT={running['loss_rt']/denom:.3f} | "
            f"I={running['loss_img']/denom:.3f} | "
            f"P={running['psnr']/denom:.2f} | "
            f"B={running['ber']/denom:.3f} | "
            f"N={running['ber_new']/denom:.3f} | "
            f"RB={running['ber_res']/denom:.3f} | "
            f"T={running['token_acc']/denom:.4f} | "
            f"TC={running['token_acc_coarse']/denom:.4f}"
        )

    for k in list(running.keys()):
        running[k] /= len(loader)
    return running


@torch.no_grad()
def eval_one_epoch(
    sd_vae,
    proj_head,
    codebook_size,
    vq_codec,
    writer,
    reader,
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
):
    sd_vae.eval()
    if writer is not None:
        writer.eval()
    if reader is not None:
        reader.eval()
    loss_w = get_loss_weights(epoch)

    running = {
        "loss": 0.0,
        "loss_msg": 0.0,
        "loss_img": 0.0,
        "loss_grad": 0.0,
        "loss_stego": 0.0,
        "loss_rt": 0.0,
        "psnr": 0.0,
        "ber": 0.0,
        "ber_old": 0.0,
        "ber_new": 0.0,
        "ber_res": 0.0,
        "token_acc": 0.0,
        "token_acc_coarse": 0.0,
    }

    saved_batches = 0
    pbar = tqdm(loader, desc=f"Val Epoch {epoch}")
    for step, (x, z_clean, y_clean, paths) in enumerate(pbar):
        x = x.to(device)
        z_clean = z_clean.to(device)
        y_clean = y_clean.to(device)

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

        _, metrics = compute_losses(
            outputs,
            prev_active_bits=prev_payload_msb_bits,
            new_bits_weight=new_bits_weight,
            **loss_w,
        )

        x01 = ((outputs["x_msg"] + 1) * 0.5).clamp(0, 1)
        xrec01 = ((outputs["x_rec"] + 1) * 0.5).clamp(0, 1)
        metrics["psnr"] = psnr_torch(x01, xrec01)

        for k, v in metrics.items():
            if k not in running:
                running[k] = 0.0
            running[k] += v

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

    for k in list(running.keys()):
        running[k] /= len(loader)
    return running


# ============================================================
# Main
# ============================================================

def main():
    _refresh_conditioner_util_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if not bool(USE_WRITER_READER):
        raise RuntimeError("This script is locked to Writer+Reader+VAE LoRA mode. Set USE_WRITER_READER=True.")
    print("[Mode] ConditionerWriter(FiLM@DecoderUpBlocks)+Reader+VAE LoRA training")

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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

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
    wr_params = [p for p in list(writer.parameters()) + list(reader.parameters()) if p.requires_grad]
    print(
        f"[WR] trainable_params={sum(p.numel() for p in wr_params)}, "
        f"lr={WRITER_READER_LR}, wd={WRITER_READER_WEIGHT_DECAY}"
    )

    ckpt_root = SCRIPT_DIR / f"ckpt/{DATA_PRESET}/base"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    vis_root = SCRIPT_DIR / f"vis/{DATA_PRESET}/base"
    vis_root.mkdir(parents=True, exist_ok=True)
    prev_best_path = None
    resume_ckpt_obj = None
    resume_stage_msb = None
    resume_prev_stage_msb = None
    resume_epoch = 0
    resume_new_bits_w = None

    if len(RESUME_CKPT_PATH.strip()) > 0:
        resume_p = Path(RESUME_CKPT_PATH.strip())
        if not resume_p.is_absolute():
            resume_p = SCRIPT_DIR / resume_p
        if resume_p.exists():
            prev_best_path = str(resume_p)
            resume_ckpt_obj = safe_torch_load(prev_best_path, map_location="cpu")
            resume_stage_msb = int(
                resume_ckpt_obj.get("stage_msb", resume_ckpt_obj.get("payload_msb_bits", 0)) or 0
            )
            resume_prev_stage_msb = int(resume_ckpt_obj.get("prev_stage_msb", 0) or 0)
            resume_epoch = int(resume_ckpt_obj.get("epoch", 0) or 0)
            resume_new_bits_w = float(resume_ckpt_obj.get("new_bits_weight", NEW_BITS_WEIGHT))
            print(f"[Resume] Using explicit checkpoint: {prev_best_path}")
            if bool(RESUME_MID_STAGE):
                if int(resume_stage_msb) <= 0:
                    raise RuntimeError(
                        "RESUME_MID_STAGE=True requires stage_msb / payload_msb_bits metadata in RESUME_CKPT_PATH."
                    )
                msb_schedule = [int(resume_stage_msb)]
                first_stage_msb = int(resume_stage_msb)
                print(
                    f"[Resume] Mid-stage continue: stage={resume_stage_msb}, prev={resume_prev_stage_msb}, "
                    f"epoch={resume_epoch}, target_end={RESUME_MID_STAGE_END_EPOCH if int(RESUME_MID_STAGE_END_EPOCH) > 0 else 'default'}"
                )
        else:
            print(f"[Resume] checkpoint not found, ignore: {resume_p}")
    elif bool(AUTO_RESUME_FIRST_STAGE_BEST):
        auto_p = ckpt_root / f"msb_{first_stage_msb:02d}" / "best.pt"
        if auto_p.exists():
            prev_best_path = str(auto_p)
            print(f"[Resume] Auto-loaded first stage best: {prev_best_path}")

    for stage_idx, stage_msb in enumerate(msb_schedule, start=1):
        stage_msb = int(stage_msb)
        is_mid_stage_resume = bool(
            RESUME_MID_STAGE
            and stage_idx == 1
            and resume_ckpt_obj is not None
            and int(stage_msb) == int(resume_stage_msb or -1)
        )
        if is_mid_stage_resume:
            prev_stage_msb = int(resume_prev_stage_msb or 0)
            stage_new_bits_w = float(resume_new_bits_w if resume_new_bits_w is not None else NEW_BITS_WEIGHT)
        else:
            prev_stage_msb = int(msb_schedule[stage_idx - 2]) if stage_idx > 1 else 0
            stage_new_bits_w = float(NEW_BITS_WEIGHT) if prev_stage_msb < stage_msb else 1.0
        stage_dir = ckpt_root / f"msb_{stage_msb:02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        stage_bits = int(Nt * stage_msb)
        print(f"[Stage {stage_idx}] MSB={stage_msb}: payload_bits={stage_bits}, mode=conditioner_reader")

        if prev_best_path is not None and Path(prev_best_path).exists():
            prev = resume_ckpt_obj if (is_mid_stage_resume and resume_ckpt_obj is not None) else safe_torch_load(prev_best_path, map_location="cpu")
            load_lora_state(lora_modules, prev)
            load_writer_reader_state(writer, reader, prev)
            print(f"[Stage {stage_idx}] Loaded previous best: {prev_best_path}")

        optim_groups = [{"params": lora_params, "lr": float(LORA_LR)}]
        if len(wr_params) > 0:
            optim_groups.append(
                {
                    "params": wr_params,
                    "lr": float(WRITER_READER_LR),
                    "weight_decay": float(WRITER_READER_WEIGHT_DECAY),
                }
            )
        optimizer = torch.optim.Adam(optim_groups)

        best_path = stage_dir / "best.pt"
        best_val = 1e9
        if best_path.exists():
            best_obj = safe_torch_load(str(best_path), map_location="cpu")
            best_stats = best_obj.get("val_stats", {}) if isinstance(best_obj, dict) else {}
            best_val = float(best_stats.get("loss", best_val))
        stage_epochs = get_stage_epochs(stage_msb)
        default_end_epoch = int(stage_epochs + STAGE_MAX_EXTRA_EPOCHS)
        start_epoch = 1
        loop_end_epoch = int(default_end_epoch)
        finish_gate_epoch = int(stage_epochs)
        if is_mid_stage_resume:
            start_epoch = int(resume_epoch) + 1
            if int(RESUME_MID_STAGE_END_EPOCH) > 0:
                loop_end_epoch = int(RESUME_MID_STAGE_END_EPOCH)
                finish_gate_epoch = int(RESUME_MID_STAGE_END_EPOCH)
        gate_key = "ber_all"
        stage_passed = False

        print(
            f"\n===== Stage {stage_idx}/{len(msb_schedule)} | "
            f"MSB={stage_msb} | prev={prev_stage_msb} | epochs={stage_epochs}(+{STAGE_MAX_EXTRA_EPOCHS}) | "
            f"new_bits_w={stage_new_bits_w:.1f} | gate={gate_key}<={BER_NEW_ADVANCE_THRESHOLD:.3f} ====="
        )
        if is_mid_stage_resume:
            print(
                f"  [Resume] continue current stage from epoch {start_epoch} to {loop_end_epoch} "
                f"(ckpt epoch={resume_epoch}, best_val={best_val:.4f})"
            )
        if int(start_epoch) > int(loop_end_epoch):
            print(
                f"  [Resume] nothing to run for MSB={stage_msb}: start_epoch={start_epoch} > end_epoch={loop_end_epoch}."
            )
            prev_best_path = str(best_path) if best_path.exists() else prev_best_path
            continue
        for epoch in range(int(start_epoch), int(loop_end_epoch) + 1):
            train_stats = train_one_epoch(
                sd_vae=sd_vae,
                proj_head=proj_head,
                codebook_size=codebook_size,
                vq_codec=vq_codec,
                writer=writer,
                reader=reader,
                loader=train_loader,
                optimizer=optimizer,
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
                loader=val_loader,
                device=device,
                epoch=epoch,
                visual_dir=str(vis_root / f"msb_{stage_msb:02d}"),
                num_visual_batches=3,
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
                f"rber={train_stats.get('ber_res', 0.0):.4f}, tok={train_stats['token_acc']:.4f}, tokc={train_stats.get('token_acc_coarse', 0.0):.4f} || "
                f"val loss={val_stats['loss']:.4f}, ber={val_stats['ber']:.4f}, old={val_stats['ber_old']:.4f}, new={val_stats['ber_new']:.4f}, "
                f"rber={val_stats.get('ber_res', 0.0):.4f}, tok={val_stats['token_acc']:.4f}, tokc={val_stats.get('token_acc_coarse', 0.0):.4f}"
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
                "epoch": epoch,
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
                "writer_hidden": int(WRITER_HIDDEN),
                "reader_hidden": int(READER_HIDDEN),
                "writer_reader_lr": float(WRITER_READER_LR),
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

            if val_stats["loss"] < best_val:
                best_val = val_stats["loss"]
                torch.save(save_obj, best_path)

            gate_val = float(
                max(
                    val_stats.get("ber", 1.0),
                    val_stats.get("ber_old", 0.0 if prev_stage_msb == 0 else 1.0),
                    val_stats.get("ber_new", 1.0),
                )
            )
            if gate_val <= float(BER_NEW_ADVANCE_THRESHOLD):
                stage_passed = True
                if bool(STAGE_EARLY_ADVANCE):
                    print(f"  [Gate] pass at epoch {epoch}: {gate_key}={gate_val:.4f}, advance to next stage.")
                    break
            if epoch >= int(finish_gate_epoch) and stage_passed:
                print(f"  [Gate] pass at epoch {epoch}: {gate_key}={gate_val:.4f}, finish stage.")
                break

            if (
                epoch == int(stage_epochs)
                and not stage_passed
                and int(STAGE_MAX_EXTRA_EPOCHS) > 0
                and int(loop_end_epoch) > int(stage_epochs)
            ):
                print(
                    f"  [Gate] not passed after base epochs: {gate_key}={gate_val:.4f}, "
                    f"ber={val_stats.get('ber', 1.0):.4f}, old={val_stats.get('ber_old', 1.0):.4f}, new={val_stats.get('ber_new', 1.0):.4f}. "
                    f"Continue extra epochs up to {max(0, int(loop_end_epoch) - int(stage_epochs))}."
                )

        prev_best_path = str(best_path)

        if bool(STAGE_STRICT_ADVANCE) and not stage_passed:
            print(
                f"[Gate] stop escalation at MSB={stage_msb}: "
                f"target {gate_key}<={BER_NEW_ADVANCE_THRESHOLD:.3f} not reached."
            )
            break


if __name__ == "__main__":
    main()
