#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_fidelity
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

from diffusers.models import AutoencoderKL

from JiT.util.image_transforms import build_center_crop_normalize_transform


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_DEFAULT_FID_STATS_PATH = Path("/work/nvme/betw/msalunkhe/data/jit_in256_stats.npz")


def collate_fn(batch):
    images = torch.stack([sample["image"] for sample in batch], dim=0)
    return {"image": images}


def init_distributed(device_arg: str) -> tuple[torch.device, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed evaluation requires CUDA.")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return torch.device("cuda", local_rank), rank, world_size, True

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return device, 0, 1, False


def barrier_if_distributed(is_distributed: bool) -> None:
    if is_distributed:
        dist.barrier()


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple SDXL-VAE FID/IS eval following JiT's ImageNet loading setup."
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help='Dataset split to reconstruct. Defaults to "validation".',
    )
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-images", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--vae_pretrained_path",
        type=str,
        default="stabilityai/sdxl-vae",
    )
    parser.add_argument(
        "--fid-statistics-file",
        type=str,
        default=str(_DEFAULT_FID_STATS_PATH),
        help="Defaults to the same .npz path used by JiT decoder training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Single-process device. Ignored under torchrun, where LOCAL_RANK selects the GPU.',
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def resolve_fid_stats_path(args: argparse.Namespace) -> Path:
    return Path(args.fid_statistics_file).expanduser().resolve()


def validate_fid_stats_file(path: Path) -> None:
    try:
        stats = np.load(path)
    except Exception as exc:
        raise ValueError(f"Failed to load FID statistics file {path}: {exc}") from exc

    keys = set(stats.files)
    if not {"mu", "sigma"}.issubset(keys):
        raise ValueError(
            f"FID statistics file {path} must contain 'mu' and 'sigma' arrays, found keys: {sorted(keys)}"
        )

    mu = stats["mu"]
    sigma = stats["sigma"]
    if mu.shape != (2048,) or sigma.shape != (2048, 2048):
        raise ValueError(
            "FID statistics shape mismatch. "
            f"Expected mu=(2048,) and sigma=(2048, 2048) for torch-fidelity's "
            f'inception-v3-compat extractor, but got mu={mu.shape} and sigma={sigma.shape} from "{path}".'
        )


def save_reconstructions(images: torch.Tensor, sample_ids: torch.Tensor, output_dir: Path) -> None:
    for image_tensor, sample_id in zip(images, sample_ids.tolist(), strict=True):
        image = TF.to_pil_image(image_tensor.clamp(0.0, 1.0))
        image.save(output_dir / f"{sample_id:08d}.png", format="PNG", compress_level=0)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device, rank, world_size, is_distributed = init_distributed(args.device)

    if args.image_size % 8 != 0:
        cleanup_distributed(is_distributed)
        raise ValueError("Image size must be divisible by 8.")

    fid_statistics_file = resolve_fid_stats_path(args)
    if not fid_statistics_file.is_file():
        cleanup_distributed(is_distributed)
        raise FileNotFoundError(f"FID statistics file not found: {fid_statistics_file}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    recon_dir = output_dir / "reconstructions"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        recon_dir.mkdir(parents=True, exist_ok=True)
    barrier_if_distributed(is_distributed)

    transform = build_center_crop_normalize_transform(
        args.image_size,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    dataset = load_dataset(args.data_path, split=args.split)
    end_index = args.start_index + args.num_images
    if args.start_index < 0:
        cleanup_distributed(is_distributed)
        raise ValueError("--start-index must be non-negative.")
    if end_index > len(dataset):
        cleanup_distributed(is_distributed)
        raise ValueError(
            f"Requested samples [{args.start_index}, {end_index}) exceed dataset size {len(dataset)}."
        )
    selected_indices = list(range(args.start_index, end_index))
    local_indices = selected_indices[rank::world_size]
    dataset = dataset.select(local_indices)
    dataset = dataset.with_format("torch")
    dataset = dataset.with_transform(
        lambda examples: {
            "image": [transform(image.convert("RGB")) for image in examples["image"]],
            "label": examples["label"],
        }
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )

    vae = AutoencoderKL.from_pretrained(
        args.vae_pretrained_path,
        local_files_only=args.local_files_only,
    ).to(device)
    vae.eval()
    vae.enable_slicing()

    local_offset = 0
    progress = tqdm(loader, total=len(loader), desc=f"Rank {rank}", disable=rank != 0)
    for batch in progress:
        images = batch["image"].to(device)
        latents = vae.encode(images).latent_dist.sample().mul_(vae.config.scaling_factor)
        reconstructions = vae.decode(latents / vae.config.scaling_factor).sample
        reconstructions = reconstructions.clamp(-1.0, 1.0).add(1.0).div(2.0).cpu()

        batch_size = reconstructions.shape[0]
        sample_ids = torch.tensor(local_indices[local_offset:local_offset + batch_size], dtype=torch.long)
        save_reconstructions(reconstructions, sample_ids, recon_dir)
        local_offset += batch_size

    barrier_if_distributed(is_distributed)

    if rank == 0:
        metrics = torch_fidelity.calculate_metrics(
            input1=str(recon_dir),
            input2=None,
            fid_statistics_file=str(fid_statistics_file),
            cuda=device.type == "cuda",
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=True,
        )
        summary = {
            "fid": float(metrics["frechet_inception_distance"]),
            "inception_score_mean": float(metrics["inception_score_mean"]),
            "inception_score_std": float(metrics["inception_score_std"]),
            "num_images": args.num_images,
            "start_index": args.start_index,
            "split": args.split,
            "image_size": args.image_size,
            "data_path": args.data_path,
            "vae_pretrained_path": args.vae_pretrained_path,
            "fid_statistics_file": str(fid_statistics_file),
            "reconstructions_dir": str(recon_dir),
            "world_size": world_size,
        }
        if str(fid_statistics_file) == str(_DEFAULT_FID_STATS_PATH) and args.split != "train":
            summary["note"] = (
                "Using validation images with the decoder's default FID stats file. "
                "That default stats file is likely train-reference stats because JiT/prepare_ref.py defaults to train."
            )
        print(json.dumps(summary, indent=2))
        (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    barrier_if_distributed(is_distributed)
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
