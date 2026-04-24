#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from JiT.decoder import Decoder
from JiT.decoder.dataset import (
    RamLoadedShardDataset,
    inspect_feature_shards,
    resolve_feature_dir_name as _resolve_feature_dir_name,
)
from JiT.eval.utils import (
    autocast_context,
    barrier_if_distributed,
    cleanup_distributed,
    images_to_uint8,
    init_distributed,
    load_checkpoint_args as _load_checkpoint_args,
    log_rank0,
    resolve_strict_state_dict,
    run_pytorch_fid,
    save_uint8_pngs as _save_uint8_pngs,
    select_checkpoint_key as _select_checkpoint_key,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained JiT decoder on saved latent and DINO feature shards and "
            "compute FID against reconstructed validation images."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--feature-root", type=str, required=True)
    parser.add_argument("--image-data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help='Raw image split aligned with the evaluation feature shards. Defaults to "validation".',
    )
    parser.add_argument(
        "--latent-dir-name",
        type=str,
        default=None,
        help="Override the latent feature shard directory name.",
    )
    parser.add_argument(
        "--dino-dir-name",
        type=str,
        default=None,
        help="Override the DINO feature shard directory name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-process decoder eval batch size. Defaults to the checkpoint batch size.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50000,
        help="Maximum number of images to evaluate. Defaults to 50000.",
    )
    parser.add_argument(
        "--checkpoint-key",
        type=str,
        default="auto",
        choices=["auto", "model", "model_ema"],
        help="Checkpoint state dict to load. Defaults to model_ema when present, otherwise model.",
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
        "--fid-num-workers",
        type=int,
        default=4,
        help="Worker count passed to pytorch-fid.",
    )
    parser.add_argument("--pin-mem", action="store_true", dest="pin_mem")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Single-process device. Ignored under torchrun.",
    )
    return parser.parse_args()


def maybe_append_split_suffix(dataset_name: str, split: str) -> str:
    from JiT.util.feature_shards import maybe_append_split_suffix as helper

    return helper(dataset_name, split)


def resolve_feature_dir_name(
    explicit_name: str | None,
    checkpoint_name: str | None,
    default_name: str,
    split: str,
) -> str:
    return _resolve_feature_dir_name(
        explicit_name,
        checkpoint_name or default_name,
        split,
    )


def require_checkpoint_arg(checkpoint_args: argparse.Namespace, name: str):
    if not hasattr(checkpoint_args, name):
        raise KeyError(
            f"Checkpoint args are missing `{name}`. Pass an updated decoder checkpoint or "
            "regenerate it with the current training code."
        )
    return getattr(checkpoint_args, name)


def load_checkpoint_args(checkpoint_payload: dict) -> argparse.Namespace:
    return _load_checkpoint_args(checkpoint_payload, label="Decoder")


def select_checkpoint_key(args: argparse.Namespace, checkpoint_payload: dict) -> str:
    return _select_checkpoint_key(
        args.checkpoint_key,
        checkpoint_payload,
        auto_key="model_ema",
        label="Decoder",
    )


def resolve_decoder_state_dict(
    checkpoint_state: dict,
    model: torch.nn.Module,
) -> tuple[dict, str | None]:
    return resolve_strict_state_dict(checkpoint_state, model, label="Decoder"), None


def build_decoder_model_from_args(checkpoint_args: argparse.Namespace) -> Decoder:
    return Decoder(
        input_size=int(require_checkpoint_arg(checkpoint_args, "latent_size")),
        patch_size=int(require_checkpoint_arg(checkpoint_args, "decoder_patch_size")),
        latent_patch_size=int(require_checkpoint_arg(checkpoint_args, "decoder_latent_patch_size")),
        in_channels=int(require_checkpoint_arg(checkpoint_args, "latent_in_channels")),
        bottleneck_dim=int(require_checkpoint_arg(checkpoint_args, "bottleneck_dim")),
        dino_hidden_size=int(require_checkpoint_arg(checkpoint_args, "dino_hidden_size")),
        hidden_size=int(require_checkpoint_arg(checkpoint_args, "decoder_hidden_size")),
        out_channels=int(require_checkpoint_arg(checkpoint_args, "image_out_channels")),
        depth=int(require_checkpoint_arg(checkpoint_args, "decoder_depth")),
        attn_drop=float(require_checkpoint_arg(checkpoint_args, "attn_dropout")),
        proj_drop=float(require_checkpoint_arg(checkpoint_args, "proj_dropout")),
        num_heads=int(require_checkpoint_arg(checkpoint_args, "decoder_num_heads")),
        mlp_ratio=float(require_checkpoint_arg(checkpoint_args, "decoder_mlp_ratio")),
        output_image_size=int(require_checkpoint_arg(checkpoint_args, "decoder_output_image_size")),
    )


def resolve_batch_size(
    cli_batch_size: int | None,
    checkpoint_args: argparse.Namespace,
    world_size: int,
) -> int:
    if cli_batch_size is not None:
        return cli_batch_size

    checkpoint_batch_size = getattr(checkpoint_args, "batch_size", None)
    if checkpoint_batch_size is not None:
        return int(checkpoint_batch_size)

    checkpoint_global_batch_size = getattr(checkpoint_args, "global_batch_size", None)
    if checkpoint_global_batch_size is None:
        raise ValueError(
            "Could not infer the per-process batch size from the checkpoint. Pass --batch-size explicitly."
        )
    if int(checkpoint_global_batch_size) % world_size != 0:
        raise ValueError(
            f"Checkpoint global_batch_size={checkpoint_global_batch_size} is not divisible by world_size={world_size}."
        )
    return int(checkpoint_global_batch_size) // world_size


def extract_image_normalization(data_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        transform_steps = data_loader.dataset.image_store.transform.transforms
    except AttributeError as exc:
        raise AttributeError(
            "Decoder eval dataloader must expose dataset.image_store.transform.transforms."
        ) from exc

    for step in reversed(transform_steps):
        mean = getattr(step, "mean", None)
        std = getattr(step, "std", None)
        if mean is not None and std is not None:
            return (
                torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1),
                torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1),
            )

    raise ValueError(
        "Decoder eval transform must include a normalization step with mean/std."
    )


def save_uint8_pngs(images: np.ndarray, sample_ids: np.ndarray, output_dir: Path) -> None:
    _save_uint8_pngs(images, sample_ids, output_dir, width=8)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device, rank, world_size, is_distributed = init_distributed(args.device)
    try:
        log_rank0(rank, f"Starting decoder eval on device={device}, world_size={world_size}, split={args.split}.")

        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Decoder checkpoint not found: {checkpoint_path}")

        log_rank0(rank, f"Loading checkpoint from {checkpoint_path}")
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_args = load_checkpoint_args(checkpoint_payload)
        checkpoint_key = select_checkpoint_key(args, checkpoint_payload)
        checkpoint_epoch = int(checkpoint_payload.get("epoch", -1))

        image_size = int(require_checkpoint_arg(checkpoint_args, "decoder_output_image_size"))
        latent_dir_name = resolve_feature_dir_name(
            args.latent_dir_name,
            getattr(checkpoint_args, "latent_dir_name", None),
            "imagenet256_latents",
            args.split,
        )
        dino_dir_name = resolve_feature_dir_name(
            args.dino_dir_name,
            getattr(checkpoint_args, "dino_dir_name", None),
            "imagenet256_dinov3_features",
            args.split,
        )

        batch_size = resolve_batch_size(args.batch_size, checkpoint_args, world_size)
        pin_mem = getattr(checkpoint_args, "pin_mem", True) if args.pin_mem is None else args.pin_mem
        decoder_batch_prefetch = bool(getattr(checkpoint_args, "decoder_batch_prefetch", True))
        image_model_name = getattr(
            checkpoint_args,
            "image_model_name",
            "vit_base_patch16_dinov3.lvd1689m",
        )

        log_rank0(
            rank,
            "Inspecting feature shards: "
            f"latent={latent_dir_name}, dino={dino_dir_name}, feature_root={args.feature_root}",
        )

        latent_store = inspect_feature_shards(args.feature_root, latent_dir_name)
        dino_store = inspect_feature_shards(args.feature_root, dino_dir_name)
        log_rank0(
            rank,
            "Loaded shard metadata: "
            f"latent={latent_store.total_size} samples across {len(latent_store.shard_spans)} shards, "
            f"dino={dino_store.total_size} samples across {len(dino_store.shard_spans)} shards.",
        )
        dataset = RamLoadedShardDataset(
            latent_store=latent_store,
            dino_store=dino_store,
            batch_size=batch_size,
            num_replicas=world_size,
            rank=rank,
            shuffle_shards=False,
            seed=0,
            preload_next_shard=False,
            preload_next_batch=decoder_batch_prefetch,
            image_data_path=args.image_data_path,
            image_data_split=args.split,
            image_model_name=image_model_name,
            image_size=image_size,
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=pin_mem and device.type == "cuda",
        )
        log_rank0(rank, f"Built dataloader with {len(data_loader)} batches per rank and batch_size={batch_size}.")

        total_available_images = len(data_loader) * batch_size * world_size
        raw_dataset_size = data_loader.dataset.image_store.dataset_size
        max_evaluable_images = min(total_available_images, raw_dataset_size)
        target_num_images = int(args.num_images or 0)
        if target_num_images <= 0:
            target_num_images = max_evaluable_images
        if target_num_images > max_evaluable_images and rank == 0:
            print(
                f"Requested {target_num_images} decoder eval images, but only "
                f"{max_evaluable_images} unique images are available. Evaluating on the available images."
            )
        target_num_images = min(target_num_images, max_evaluable_images)
        if target_num_images <= 0:
            raise ValueError("No decoder eval images are available.")

        output_dir = Path(args.output_dir).expanduser().resolve()
        subset_tag = f"{args.split}_{target_num_images}"
        reference_dir = output_dir / f"reference_images_{subset_tag}"
        recon_dir = output_dir / f"reconstructions_{subset_tag}"
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            reference_dir.mkdir(parents=True, exist_ok=True)
            recon_dir.mkdir(parents=True, exist_ok=True)
        barrier_if_distributed(is_distributed, device)
        log_rank0(
            rank,
            f"Writing reference images to {reference_dir} and reconstructions to {recon_dir}.",
        )

        model = build_decoder_model_from_args(checkpoint_args).to(device)
        checkpoint_state, stripped_prefix = resolve_decoder_state_dict(
            checkpoint_payload[checkpoint_key],
            model,
        )
        model.load_state_dict(checkpoint_state, strict=True)
        model.eval()
        prefix_message = ""
        if stripped_prefix is not None:
            prefix_message = f" after stripping `{stripped_prefix}.`"
        log_rank0(
            rank,
            f"Model loaded with checkpoint key `{checkpoint_key}` from epoch {checkpoint_epoch}{prefix_message}.",
        )

        image_mean, image_std = extract_image_normalization(data_loader)

        local_mse_sum = 0.0
        local_mse_count = 0
        log_rank0(rank, f"Starting reconstruction loop for up to {target_num_images} images.")
        progress = tqdm(data_loader, total=len(data_loader), desc=f"Rank {rank}", disable=rank != 0)
        for step_idx, batch in enumerate(progress):
            latent = batch["latent"].to(device, non_blocking=True)
            dino = batch["dino"].to(device, non_blocking=True)
            target_image = batch["image"].to(device, non_blocking=True)
            sample_ids = batch["sample_id"].cpu().numpy().astype(np.int64, copy=False)

            with autocast_context(device):
                reconstructed = model.generate(latent, dino)

            batch_indices = (
                step_idx * world_size * batch_size
                + rank * batch_size
                + np.arange(reconstructed.shape[0], dtype=np.int64)
            )
            keep_mask = batch_indices < target_num_images
            if not np.any(keep_mask):
                break

            keep_tensor = torch.as_tensor(keep_mask, device=reconstructed.device)
            per_image_mse = (reconstructed.float() - target_image.float()).square().flatten(1).mean(dim=1)
            local_mse_sum += float(per_image_mse[keep_tensor].sum().item())
            local_mse_count += int(keep_mask.sum())

            reference_images = images_to_uint8(target_image, image_mean, image_std)[keep_mask]
            reconstructed_images = images_to_uint8(reconstructed, image_mean, image_std)[keep_mask]
            kept_sample_ids = sample_ids[keep_mask]
            save_uint8_pngs(reference_images, kept_sample_ids, reference_dir)
            save_uint8_pngs(reconstructed_images, kept_sample_ids, recon_dir)

        barrier_if_distributed(is_distributed, device)

        mse_stats = torch.tensor(
            [local_mse_sum, float(local_mse_count)],
            dtype=torch.float64,
            device=device,
        )
        if is_distributed:
            dist.all_reduce(mse_stats)
        recon_mse = float(mse_stats[0].item() / max(mse_stats[1].item(), 1.0))

        if rank == 0:
            log_rank0(rank, "Reconstruction loop finished. Computing FID.")
            fid = run_pytorch_fid(
                reference_dir=reference_dir,
                recon_dir=recon_dir,
                device=device,
                batch_size=args.fid_batch_size,
                dims=args.fid_dims,
                num_workers=args.fid_num_workers,
            )
            summary = {
                "fid": fid,
                "recon_mse": recon_mse,
                "num_images": target_num_images,
                "split": args.split,
                "feature_root": args.feature_root,
                "latent_dir_name": latent_dir_name,
                "dino_dir_name": dino_dir_name,
                "image_data_path": args.image_data_path,
                "image_size": image_size,
                "batch_size": batch_size,
                "world_size": world_size,
                "checkpoint": str(checkpoint_path),
                "checkpoint_key": checkpoint_key,
                "checkpoint_epoch": checkpoint_epoch,
                "reference_dir": str(reference_dir),
                "reconstructions_dir": str(recon_dir),
                "fid_batch_size": args.fid_batch_size,
                "fid_dims": args.fid_dims,
            }
            print(json.dumps(summary, indent=2))
            (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        barrier_if_distributed(is_distributed, device)
    finally:
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
