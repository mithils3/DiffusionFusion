"""Decoder loading and FID utilities for diffusion evaluation.

Provides helpers to load a trained JiT decoder checkpoint, decode
generated latent + DINO features into RGB images, and compute FID
using pytorch-fid.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from JiT.decoder import Decoder


def load_decoder_for_eval(
    checkpoint_path: str,
    device: torch.device,
    checkpoint_key: str = "auto",
) -> Decoder:
    """Load a trained decoder from a checkpoint file.

    Returns the decoder model in eval mode on *device*.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Decoder checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu")

    args_dict = payload.get("args")
    if not isinstance(args_dict, dict):
        raise KeyError("Decoder checkpoint missing `args` payload.")
    ckpt_args = argparse.Namespace(**args_dict)

    model = Decoder(
        input_size=int(ckpt_args.latent_size),
        patch_size=int(ckpt_args.decoder_patch_size),
        latent_patch_size=int(ckpt_args.decoder_latent_patch_size),
        in_channels=int(ckpt_args.latent_in_channels),
        bottleneck_dim=int(ckpt_args.bottleneck_dim),
        dino_hidden_size=int(ckpt_args.dino_hidden_size),
        hidden_size=int(ckpt_args.decoder_hidden_size),
        out_channels=int(ckpt_args.image_out_channels),
        depth=int(ckpt_args.decoder_depth),
        attn_drop=float(ckpt_args.attn_dropout),
        proj_drop=float(ckpt_args.proj_dropout),
        num_heads=int(ckpt_args.decoder_num_heads),
        mlp_ratio=float(ckpt_args.decoder_mlp_ratio),
        output_image_size=int(ckpt_args.decoder_output_image_size),
    )

    if checkpoint_key == "auto":
        key = "model_ema" if "model_ema" in payload else "model"
    else:
        key = checkpoint_key
    if key not in payload:
        raise KeyError(f"Checkpoint key `{key}` not found in {path}.")

    state_dict = _resolve_state_dict(payload[key], model)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _resolve_state_dict(
    checkpoint_state: dict,
    model: torch.nn.Module,
) -> dict:
    """Strip common prefixes from a checkpoint state dict to match model keys."""
    expected_keys = set(model.state_dict().keys())

    if set(checkpoint_state.keys()) == expected_keys:
        return dict(checkpoint_state)

    candidate = dict(checkpoint_state)
    for _ in range(4):
        if set(candidate.keys()) == expected_keys:
            return candidate
        if not candidate or any("." not in k for k in candidate):
            break
        heads = {k.split(".", 1)[0] for k in candidate}
        if len(heads) != 1:
            break
        candidate = {k.split(".", 1)[1]: v for k, v in candidate.items()}

    for prefix in ("decoder.", "module.decoder.", "_orig_mod.decoder."):
        filtered = {
            k[len(prefix):]: v
            for k, v in checkpoint_state.items()
            if k.startswith(prefix)
        }
        if set(filtered.keys()) == expected_keys:
            return filtered

    raise RuntimeError("Decoder checkpoint state dict is incompatible with the model.")


def decode_with_decoder(
    decoder: Decoder,
    latent: torch.Tensor,
    dino: torch.Tensor,
) -> np.ndarray:
    """Decode latent + DINO features into uint8 images using the custom decoder.

    Args:
        decoder: Trained decoder model in eval mode.
        latent: Latent features ``[B, C, H, W]`` from the denoiser.
        dino: DINO features ``[B, D, H, W]`` from the denoiser.

    Returns:
        uint8 numpy array of shape ``[B, H, W, 3]``.
    """
    device = latent.device
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with ctx:
        images = decoder.generate(latent, dino)
    images = torch.clamp(127.5 * images + 128.0, 0, 255)
    return images.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
