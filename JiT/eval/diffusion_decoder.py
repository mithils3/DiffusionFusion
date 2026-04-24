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
import timm

from JiT.decoder import Decoder
from JiT.eval.utils import images_to_uint8, resolve_strict_state_dict


_DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)


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
    output_mean, output_std = _resolve_decoder_output_stats(ckpt_args)
    model._output_mean = output_mean
    model._output_std = output_std
    model.to(device)
    model.eval()
    return model


def _resolve_decoder_output_stats(
    ckpt_args: argparse.Namespace,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    image_mean = getattr(ckpt_args, "image_mean", None)
    image_std = getattr(ckpt_args, "image_std", None)
    if image_mean is not None and image_std is not None:
        return tuple(float(value) for value in image_mean), tuple(float(value) for value in image_std)

    model_name = getattr(ckpt_args, "image_model_name", None)
    if model_name:
        transform_model = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True,
        )
        try:
            data_config = timm.data.resolve_model_data_config(transform_model)
        finally:
            del transform_model

        mean = data_config.get("mean")
        std = data_config.get("std")
        if mean is not None and std is not None:
            return (
                tuple(float(value) for value in mean),
                tuple(float(value) for value in std),
            )

    # Older checkpoints may not record the image model config. Fall back to the
    # ImageNet normalization used by the current decoder training recipe.
    return _DEFAULT_IMAGE_MEAN, _DEFAULT_IMAGE_STD


def _resolve_state_dict(
    checkpoint_state: dict,
    model: torch.nn.Module,
) -> dict:
    return resolve_strict_state_dict(checkpoint_state, model, label="Decoder")


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
    return _decoder_images_to_uint8(images, decoder)


def _decoder_images_to_uint8(
    images: torch.Tensor,
    decoder: Decoder,
) -> np.ndarray:
    mean = torch.as_tensor(
        getattr(decoder, "_output_mean", _DEFAULT_IMAGE_MEAN),
        dtype=torch.float32,
    ).view(1, -1, 1, 1)
    std = torch.as_tensor(
        getattr(decoder, "_output_std", _DEFAULT_IMAGE_STD),
        dtype=torch.float32,
    ).view(1, -1, 1, 1)
    return images_to_uint8(images, mean, std)
