from __future__ import annotations

import argparse
import datetime
import os
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image


def init_distributed(
    device_arg: str,
    timeout_sec: int | None = None,
) -> tuple[torch.device, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed evaluation requires CUDA.")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        kwargs = {"device_id": local_rank}
        if timeout_sec is not None:
            kwargs["timeout"] = datetime.timedelta(seconds=int(timeout_sec))
        dist.init_process_group("nccl", **kwargs)
        return torch.device("cuda", local_rank), rank, world_size, True

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return device, 0, 1, False


def barrier_if_distributed(
    is_distributed: bool,
    device: torch.device | None = None,
) -> None:
    if is_distributed:
        device_ids = None
        if device is not None and device.type == "cuda" and device.index is not None:
            device_ids = [device.index]
        dist.barrier(device_ids=device_ids)


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def log_rank0(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def images_to_uint8(
    images: torch.Tensor,
    mean: torch.Tensor | tuple[float, ...] | list[float],
    std: torch.Tensor | tuple[float, ...] | list[float],
) -> np.ndarray:
    images = images.detach().float().cpu()
    mean_tensor = torch.as_tensor(mean, dtype=images.dtype).view(1, -1, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=images.dtype).view(1, -1, 1, 1)
    images = images * std_tensor + mean_tensor
    images = images.clamp_(0.0, 1.0)
    images = images.mul(255.0).round().to(torch.uint8)
    return images.permute(0, 2, 3, 1).numpy()


def save_uint8_pngs(images: np.ndarray, ids: np.ndarray, output_dir: Path, *, width: int) -> None:
    for image_array, image_id in zip(images, ids.tolist(), strict=True):
        Image.fromarray(image_array).save(
            output_dir / f"{image_id:0{width}d}.png",
            format="PNG",
            compress_level=0,
        )


def run_pytorch_fid(
    *,
    reference_dir: Path,
    recon_dir: Path,
    device: torch.device,
    batch_size: int,
    dims: int,
    num_workers: int,
) -> float:
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError as exc:
        raise ImportError(
            "pytorch-fid is required for decoder evaluation. Install it with "
            "`pip install pytorch-fid` or from requirements.txt."
        ) from exc

    return float(
        calculate_fid_given_paths(
            [str(reference_dir), str(recon_dir)],
            batch_size=batch_size,
            device=device,
            dims=dims,
            num_workers=num_workers,
        )
    )


def run_torch_fidelity_metrics(
    image_dir: Path,
    fid_stats_path: Path,
    device: torch.device,
) -> dict:
    try:
        import torch_fidelity
    except ImportError as exc:
        raise ImportError(
            "torch-fidelity is required for generation evaluation. "
            "Install it from requirements.txt."
        ) from exc

    return torch_fidelity.calculate_metrics(
        input1=str(image_dir),
        input2=None,
        fid_statistics_file=str(fid_stats_path),
        cuda=device.type == "cuda",
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=True,
    )


def load_checkpoint_payload(checkpoint_path: Path) -> dict:
    if hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals([argparse.Namespace]):
            return torch.load(checkpoint_path, map_location="cpu")
    return torch.load(checkpoint_path, map_location="cpu")


def load_checkpoint_args(checkpoint_payload: Mapping[str, object], *, label: str) -> argparse.Namespace:
    args_payload = checkpoint_payload.get("args")
    if isinstance(args_payload, argparse.Namespace):
        return deepcopy(args_payload)
    if isinstance(args_payload, dict):
        return argparse.Namespace(**args_payload)
    raise KeyError(
        f"{label} checkpoint does not contain a supported `args` payload. "
        "Expected argparse.Namespace or dict."
    )


def select_checkpoint_key(
    requested_key: str,
    checkpoint_payload: Mapping[str, object],
    *,
    auto_key: str,
    label: str,
) -> str:
    if requested_key == "auto":
        return auto_key if auto_key in checkpoint_payload else "model"
    if requested_key not in checkpoint_payload:
        raise KeyError(f"Checkpoint key `{requested_key}` not found in the {label} checkpoint.")
    return requested_key


def resolve_strict_state_dict(
    checkpoint_state: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
    *,
    label: str,
) -> dict:
    if not isinstance(checkpoint_state, dict):
        raise TypeError(
            f"{label} checkpoint entry must be a state dict mapping, "
            f"got {type(checkpoint_state).__name__}."
        )

    checkpoint_keys = set(checkpoint_state.keys())
    expected_keys = set(model.state_dict().keys())
    missing_keys = sorted(expected_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - expected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"{label} checkpoint state dict does not match the current model exactly. "
            f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}; "
            f"unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}."
        )
    return dict(checkpoint_state)
