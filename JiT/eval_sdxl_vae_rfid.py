#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch_fidelity
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

from diffusers.models import AutoencoderKL

from JiT.util.image_transforms import build_center_crop_normalize_transform


_DEFAULT_FID_STATS_PATH = Path("/work/nvme/betw/msalunkhe/data/jit_in256_stats.npz")


def collate_fn(batch):
    images = torch.stack([sample["image"] for sample in batch], dim=0)
    return {"image": images}


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
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def resolve_fid_stats_path(args: argparse.Namespace) -> Path:
    return Path(args.fid_statistics_file).expanduser().resolve()


def save_reconstructions(images: torch.Tensor, sample_ids: torch.Tensor, output_dir: Path) -> None:
    for image_tensor, sample_id in zip(images, sample_ids.tolist(), strict=True):
        image = TF.to_pil_image(image_tensor.clamp(0.0, 1.0))
        image.save(output_dir / f"{sample_id:08d}.png", format="PNG", compress_level=0)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")

    if args.image_size % 8 != 0:
        raise ValueError("Image size must be divisible by 8.")

    fid_statistics_file = resolve_fid_stats_path(args)
    if not fid_statistics_file.is_file():
        raise FileNotFoundError(f"FID statistics file not found: {fid_statistics_file}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    recon_dir = output_dir / "reconstructions"
    output_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    transform = build_center_crop_normalize_transform(
        args.image_size,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    dataset = load_dataset(args.data_path, split=args.split)
    end_index = args.start_index + args.num_images
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if end_index > len(dataset):
        raise ValueError(
            f"Requested samples [{args.start_index}, {end_index}) exceed dataset size {len(dataset)}."
        )
    dataset = dataset.select(range(args.start_index, end_index))
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

    global_offset = args.start_index
    for batch in tqdm(loader, total=len(loader), desc="Reconstructing"):
        images = batch["image"].to(device)
        latents = vae.encode(images).latent_dist.sample().mul_(vae.config.scaling_factor)
        reconstructions = vae.decode(latents / vae.config.scaling_factor).sample
        reconstructions = reconstructions.clamp(-1.0, 1.0).add(1.0).div(2.0).cpu()

        batch_size = reconstructions.shape[0]
        sample_ids = torch.arange(global_offset, global_offset + batch_size, dtype=torch.long)
        save_reconstructions(reconstructions, sample_ids, recon_dir)
        global_offset += batch_size

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
    }
    if str(fid_statistics_file) == str(_DEFAULT_FID_STATS_PATH) and args.split != "train":
        summary["note"] = (
            "Using validation images with the decoder's default FID stats file. "
            "That default stats file is likely train-reference stats because JiT/prepare_ref.py defaults to train."
        )
    print(json.dumps(summary, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
