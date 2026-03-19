#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from diffusers.models import AutoencoderKL

from JiT.util.image_transforms import build_center_crop_normalize_transform


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
        description=(
            "Reconstruct ImageNet images with SDXL-VAE, save ADM-compatible sample batches "
            "as a .npz, and optionally run guided-diffusion evaluator.py."
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
        "--reference-npz",
        type=str,
        default=None,
        help=(
            "Optional ADM reference batch .npz such as "
            "VIRTUAL_imagenet256_labeled.npz. Required only if you also pass "
            "--guided-diffusion-evaluator."
        ),
    )
    parser.add_argument(
        "--guided-diffusion-evaluator",
        type=str,
        default=None,
        help=(
            "Optional path to guided-diffusion/evaluations/evaluator.py. If provided, "
            "the script will run ADM-style metrics on the saved samples .npz."
        ),
    )
    parser.add_argument(
        "--samples-npz-name",
        type=str,
        default="reconstructions_adm.npz",
        help="Filename for the ADM-compatible samples .npz inside --output-dir.",
    )
    parser.add_argument(
        "--save-pngs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally also save per-image PNG reconstructions alongside the ADM .npz.",
    )
    parser.add_argument(
        "--keep-temp-chunks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep temporary per-rank .npz chunks used to assemble the final ADM batch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Single-process device. Ignored under torchrun, where LOCAL_RANK selects the GPU.',
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.guided_diffusion_evaluator and not args.reference_npz:
        parser.error("--reference-npz is required when --guided-diffusion-evaluator is set.")
    return args


def resolve_optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def save_uint8_pngs(images: np.ndarray, sample_positions: np.ndarray, output_dir: Path) -> None:
    for image_array, sample_position in zip(images, sample_positions.tolist(), strict=True):
        Image.fromarray(image_array).save(
            output_dir / f"{sample_position:06d}.png",
            format="PNG",
            compress_level=0,
        )


def save_chunk(
    *,
    chunk_dir: Path,
    rank: int,
    chunk_index: int,
    sample_positions: np.ndarray,
    samples: np.ndarray,
) -> Path:
    chunk_path = chunk_dir / f"rank{rank:02d}_chunk{chunk_index:06d}.npz"
    np.savez(chunk_path, positions=sample_positions, samples=samples)
    return chunk_path


def build_adm_npz_from_chunks(
    *,
    chunk_dir: Path,
    output_path: Path,
    num_images: int,
) -> tuple[Path, tuple[int, int, int, int]]:
    chunk_paths = sorted(chunk_dir.glob("rank*_chunk*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}.")

    samples: np.ndarray | None = None
    for chunk_path in tqdm(chunk_paths, desc="Merging ADM chunks"):
        with np.load(chunk_path) as payload:
            sample_positions = payload["positions"]
            chunk_samples = payload["samples"]
        if samples is None:
            height, width = chunk_samples.shape[1:3]
            samples = np.empty((num_images, height, width, 3), dtype=np.uint8)
        samples[sample_positions] = chunk_samples

    assert samples is not None
    np.savez(output_path, arr_0=samples)
    return output_path, samples.shape


def run_guided_diffusion_evaluator(
    *,
    evaluator_path: Path,
    reference_npz: Path,
    samples_npz: Path,
    output_dir: Path,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(evaluator_path),
        str(reference_npz),
        str(samples_npz),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout_path = output_dir / "guided_diffusion_eval_stdout.txt"
    stderr_path = output_dir / "guided_diffusion_eval_stderr.txt"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device, rank, world_size, is_distributed = init_distributed(args.device)

    if args.image_size % 8 != 0:
        cleanup_distributed(is_distributed)
        raise ValueError("Image size must be divisible by 8.")

    reference_npz = resolve_optional_path(args.reference_npz)
    if reference_npz is not None and not reference_npz.is_file():
        cleanup_distributed(is_distributed)
        raise FileNotFoundError(f"Reference ADM .npz file not found: {reference_npz}")

    guided_diffusion_evaluator = resolve_optional_path(args.guided_diffusion_evaluator)
    if guided_diffusion_evaluator is not None and not guided_diffusion_evaluator.is_file():
        cleanup_distributed(is_distributed)
        raise FileNotFoundError(
            f"guided-diffusion evaluator.py not found: {guided_diffusion_evaluator}"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    recon_dir = output_dir / "reconstructions"
    chunk_dir = output_dir / "adm_chunks"
    samples_npz_path = output_dir / args.samples_npz_name
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        if args.save_pngs:
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
    local_positions = list(range(rank, len(selected_indices), world_size))
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
    chunk_index = 0
    progress = tqdm(loader, total=len(loader), desc=f"Rank {rank}", disable=rank != 0)
    for batch in progress:
        images = batch["image"].to(device)
        latents = vae.encode(images).latent_dist.sample().mul_(vae.config.scaling_factor)
        reconstructions = vae.decode(latents / vae.config.scaling_factor).sample
        reconstructions = (
            (127.5 * reconstructions + 128.0)
            .clamp(0.0, 255.0)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )

        batch_size = reconstructions.shape[0]
        sample_positions = np.asarray(
            local_positions[local_offset:local_offset + batch_size],
            dtype=np.int64,
        )
        save_chunk(
            chunk_dir=chunk_dir,
            rank=rank,
            chunk_index=chunk_index,
            sample_positions=sample_positions,
            samples=reconstructions,
        )
        if args.save_pngs:
            save_uint8_pngs(reconstructions, sample_positions, recon_dir)
        local_offset += batch_size
        chunk_index += 1

    barrier_if_distributed(is_distributed)

    if rank == 0:
        samples_npz_path, sample_shape = build_adm_npz_from_chunks(
            chunk_dir=chunk_dir,
            output_path=samples_npz_path,
            num_images=args.num_images,
        )
        summary = {
            "num_images": args.num_images,
            "start_index": args.start_index,
            "split": args.split,
            "image_size": args.image_size,
            "data_path": args.data_path,
            "vae_pretrained_path": args.vae_pretrained_path,
            "samples_npz_path": str(samples_npz_path),
            "samples_npz_shape": list(sample_shape),
            "reconstructions_dir": str(recon_dir) if args.save_pngs else None,
            "temp_chunk_dir": str(chunk_dir),
            "world_size": world_size,
            "reference_npz": str(reference_npz) if reference_npz is not None else None,
            "guided_diffusion_evaluator": (
                str(guided_diffusion_evaluator)
                if guided_diffusion_evaluator is not None
                else None
            ),
        }
        if guided_diffusion_evaluator is not None and reference_npz is not None:
            adm_eval = run_guided_diffusion_evaluator(
                evaluator_path=guided_diffusion_evaluator,
                reference_npz=reference_npz,
                samples_npz=samples_npz_path,
                output_dir=output_dir,
            )
            summary["guided_diffusion_eval"] = adm_eval
        elif reference_npz is not None:
            summary["guided_diffusion_eval_command"] = [
                sys.executable,
                "path/to/guided-diffusion/evaluations/evaluator.py",
                str(reference_npz),
                str(samples_npz_path),
            ]
        if not args.keep_temp_chunks:
            shutil.rmtree(chunk_dir)
            summary["temp_chunk_dir_removed"] = True
        else:
            summary["temp_chunk_dir_removed"] = False
        print(json.dumps(summary, indent=2))
        (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    barrier_if_distributed(is_distributed)
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
