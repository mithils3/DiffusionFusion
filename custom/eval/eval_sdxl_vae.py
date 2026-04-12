#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusers.models import AutoencoderKL

from custom.util.image_transforms import build_center_crop_normalize_transform


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def collate_fn(batch):
    images = torch.stack([sample["image"] for sample in batch], dim=0)
    image_uint8 = torch.stack([sample["image_uint8"]
                              for sample in batch], dim=0)
    return {
        "image": images,
        "image_uint8": image_uint8,
    }


def init_distributed(device_arg: str) -> tuple[torch.device, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed evaluation requires CUDA.")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank %
                         torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return torch.device("cuda", local_rank), rank, world_size, True

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but no CUDA device is available.")
    return device, 0, 1, False


def barrier_if_distributed(is_distributed: bool) -> None:
    if is_distributed:
        dist.barrier()


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct ImageNet images with SDXL-VAE and compute FID using pytorch-fid "
            "between saved validation images and reconstructions."
        )
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help='Dataset split to reconstruct. Defaults to "validation".',
    )
    parser.add_argument("--image-size", type=int,
                        choices=[224, 256, 512], default=256)
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
        "--fid-batch-size",
        type=int,
        default=256,
        help="Batch size passed to pytorch-fid feature extraction.",
    )
    parser.add_argument(
        "--fid-dims",
        type=int,
        default=2048,
        choices=[64, 192, 768, 2048],
        help="Feature dimensionality passed to pytorch-fid.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Single-process device. Ignored under torchrun.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def save_uint8_pngs(images: np.ndarray, sample_ids: np.ndarray, output_dir: Path) -> None:
    for image_array, sample_id in zip(images, sample_ids.tolist(), strict=True):
        Image.fromarray(image_array).save(
            output_dir / f"{sample_id:08d}.png",
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
            "pytorch-fid is required for SD-VAE evaluation. Install it with "
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


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device, rank, world_size, is_distributed = init_distributed(args.device)

    if args.image_size % 8 != 0:
        cleanup_distributed(is_distributed)
        raise ValueError("Image size must be divisible by 8.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    subset_tag = f"{args.split}_{args.start_index:06d}_{args.num_images}"
    reference_dir = output_dir / f"reference_images_{subset_tag}"
    recon_dir = output_dir / f"reconstructions_{subset_tag}"

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        reference_dir.mkdir(parents=True, exist_ok=True)
        recon_dir.mkdir(parents=True, exist_ok=True)
    barrier_if_distributed(is_distributed)

    input_transform = build_center_crop_normalize_transform(
        args.image_size,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    save_transform = build_center_crop_normalize_transform(args.image_size)

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
            "image": [input_transform(image.convert("RGB")) for image in examples["image"]],
            "image_uint8": [
                save_transform(image.convert("RGB"))
                .mul(255.0)
                .clamp(0.0, 255.0)
                .to(torch.uint8)
                for image in examples["image"]
            ],
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
    progress = tqdm(loader, total=len(loader),
                    desc=f"Rank {rank}", disable=rank != 0)
    for batch in progress:
        images = batch["image"].to(device)
        batch_size = images.shape[0]
        sample_ids = np.asarray(
            local_indices[local_offset:local_offset + batch_size],
            dtype=np.int64,
        )

        reference_images = batch["image_uint8"].permute(
            0, 2, 3, 1).cpu().numpy()
        save_uint8_pngs(reference_images, sample_ids, reference_dir)

        latents = vae.encode(images).latent_dist.sample().mul_(
            vae.config.scaling_factor)
        reconstructions = vae.decode(
            latents / vae.config.scaling_factor).sample
        reconstructions = (
            (127.5 * reconstructions + 128.0)
            .clamp(0.0, 255.0)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )
        save_uint8_pngs(reconstructions, sample_ids, recon_dir)
        local_offset += batch_size

    barrier_if_distributed(is_distributed)

    if rank == 0:
        fid = run_pytorch_fid(
            reference_dir=reference_dir,
            recon_dir=recon_dir,
            device=device,
            batch_size=args.fid_batch_size,
            dims=args.fid_dims,
            num_workers=args.num_workers,
        )
        summary = {
            "fid": fid,
            "num_images": args.num_images,
            "start_index": args.start_index,
            "split": args.split,
            "image_size": args.image_size,
            "data_path": args.data_path,
            "vae_pretrained_path": args.vae_pretrained_path,
            "reference_dir": str(reference_dir),
            "reconstructions_dir": str(recon_dir),
            "fid_batch_size": args.fid_batch_size,
            "fid_dims": args.fid_dims,
            "world_size": world_size,
        }
        print(json.dumps(summary, indent=2))
        (output_dir / "metrics.json").write_text(json.dumps(summary,
                                                            indent=2) + "\n", encoding="utf-8")

    barrier_if_distributed(is_distributed)
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
