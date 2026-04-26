#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from JiT.decoder.dataset import RamLoadedShardDataset, inspect_feature_shards
from JiT.eval.eval_decoder import (
    build_decoder_model_from_args,
    extract_image_normalization,
    load_checkpoint_args,
    resolve_batch_size,
    resolve_decoder_state_dict,
    resolve_feature_dir_name,
    select_checkpoint_key,
)
from JiT.eval.utils import images_to_uint8, load_checkpoint_payload


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_patch(value: str) -> tuple[int, int]:
    try:
        row_text, col_text = value.split(",", maxsplit=1)
        row = int(row_text)
        col = int(col_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected patch as row,col, got {value!r}."
        ) from exc
    if row < 0 or col < 0:
        raise argparse.ArgumentTypeError("Patch row and col must be non-negative.")
    return row, col


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Grad-CAM-style patch attribution on JiT.decoder.model.Decoder context "
            "tokens, split into DINO and latent token heatmaps."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--feature-root", type=str, required=True)
    parser.add_argument("--image-data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--latent-dir-name", type=str, default=None)
    parser.add_argument("--dino-dir-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of samples to process. Defaults to 100.",
    )
    parser.add_argument(
        "--patch",
        type=parse_patch,
        action="append",
        default=None,
        help=(
            "Output patch as row,col. Repeat for multiple patches per sample. "
            "If omitted, one high-variance decoded patch is selected per sample."
        ),
    )
    parser.add_argument(
        "--target",
        type=str,
        default="mean",
        choices=["mean", "square", "red", "green", "blue"],
        help="Scalar target computed from the selected output patch.",
    )
    parser.add_argument(
        "--checkpoint-key",
        type=str,
        default="auto",
        choices=["auto", "model", "model_ema"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        dest="pin_mem",
        help="Pin dataloader memory when using CUDA.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=None)
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Save raw signed and absolute attribution maps beside each figure.",
    )
    parser.add_argument(
        "--upload-to-hf",
        action="store_true",
        help="Upload the output directory to Hugging Face after generation.",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="Mithilss",
        help="Hugging Face repo id used with --upload-to-hf.",
    )
    parser.add_argument(
        "--hf-repo-type",
        type=str,
        default="dataset",
        choices=["dataset", "model", "space"],
    )
    parser.add_argument(
        "--hf-path-in-repo",
        type=str,
        default="decoder_patch_attribution",
    )
    return parser.parse_args()


def tensor_to_uint8_image(
    image: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Image.Image:
    array = images_to_uint8(image.unsqueeze(0), mean, std)[0]
    return Image.fromarray(array, mode="RGB")


def normalize_map(values: np.ndarray, vmax: float | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if vmax is None:
        vmax = float(arr.max()) if arr.size else 0.0
    if vmax <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / vmax, 0.0, 1.0)


def _interpolate_colormap(x: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    out = np.zeros((*x.shape, 3), dtype=np.float32)
    for idx in range(len(stops) - 1):
        left_x, left_rgb = stops[idx]
        right_x, right_rgb = stops[idx + 1]
        if idx == len(stops) - 2:
            mask = (x >= left_x) & (x <= right_x)
        else:
            mask = (x >= left_x) & (x < right_x)
        if not np.any(mask):
            continue
        denom = max(right_x - left_x, 1.0e-12)
        t = ((x[mask] - left_x) / denom).reshape(-1, 1)
        left = np.asarray(left_rgb, dtype=np.float32)
        right = np.asarray(right_rgb, dtype=np.float32)
        out[mask] = left * (1.0 - t) + right * t
    return np.clip(out, 0, 255).astype(np.uint8)


def heatmap_image(values: np.ndarray, size: int, *, vmax: float | None = None) -> Image.Image:
    normed = normalize_map(values, vmax=vmax)
    colors = _interpolate_colormap(
        normed,
        [
            (0.00, (8, 8, 24)),
            (0.30, (64, 20, 115)),
            (0.60, (210, 72, 66)),
            (0.82, (252, 175, 70)),
            (1.00, (255, 246, 180)),
        ],
    )
    return Image.fromarray(colors, mode="RGB").resize((size, size), Image.Resampling.NEAREST)


def difference_image(values: np.ndarray, size: int) -> Image.Image:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.abs(arr).max()) if arr.size else 0.0
    if scale <= 0.0:
        normed = np.full_like(arr, 0.5, dtype=np.float32)
    else:
        normed = np.clip((arr / scale + 1.0) * 0.5, 0.0, 1.0)
    colors = _interpolate_colormap(
        normed,
        [
            (0.00, (49, 76, 160)),
            (0.50, (245, 245, 245)),
            (1.00, (180, 42, 42)),
        ],
    )
    return Image.fromarray(colors, mode="RGB").resize((size, size), Image.Resampling.NEAREST)


def draw_patch_box(
    image: Image.Image,
    row: int,
    col: int,
    patch_size: int,
    *,
    color: tuple[int, int, int] = (255, 48, 48),
    width: int = 3,
) -> None:
    draw = ImageDraw.Draw(image)
    x0 = col * patch_size
    y0 = row * patch_size
    x1 = (col + 1) * patch_size - 1
    y1 = (row + 1) * patch_size - 1
    max_width = max(1, min(width, (x1 - x0 + 1) // 2, (y1 - y0 + 1) // 2))
    for offset in range(max_width):
        draw.rectangle(
            [x0 + offset, y0 + offset, x1 - offset, y1 - offset],
            outline=color,
        )


def label_panel(image: Image.Image, title: str) -> Image.Image:
    title_h = 34
    panel = Image.new("RGB", (image.width, image.height + title_h), "white")
    panel.paste(image, (0, title_h))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(24, 24, 24), font=font)
    return panel


def save_figure(
    path: Path,
    *,
    recon: Image.Image,
    dino_abs: np.ndarray,
    latent_abs: np.ndarray,
    row: int,
    col: int,
    patch_size: int,
    grid_size: int,
) -> None:
    image_size = recon.width
    recon_panel = recon.copy()
    draw_patch_box(recon_panel, row, col, patch_size)

    shared_vmax = max(float(dino_abs.max()), float(latent_abs.max()), 1.0e-12)
    dino_panel = heatmap_image(dino_abs, image_size, vmax=shared_vmax)
    latent_panel = heatmap_image(latent_abs, image_size, vmax=shared_vmax)
    diff_panel = difference_image(latent_abs - dino_abs, image_size)

    heat_patch = max(image_size // grid_size, 1)
    for panel in (dino_panel, latent_panel, diff_panel):
        draw_patch_box(panel, row, col, heat_patch, color=(0, 255, 255), width=2)

    panels = [
        label_panel(recon_panel, "decoded image"),
        label_panel(dino_panel, "DINO abs attribution"),
        label_panel(latent_panel, "latent abs attribution"),
        label_panel(diff_panel, "latent abs - DINO abs"),
    ]
    gap = 12
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels) + gap * (len(panels) - 1), panels[0].height),
        "white",
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    canvas.save(path, format="PNG", compress_level=3)


def select_visible_patch(
    image: torch.Tensor,
    patch_size: int,
    *,
    border: int = 1,
) -> tuple[int, int]:
    _, height, width = image.shape
    rows = height // patch_size
    cols = width // patch_size
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid image shape for patch selection: {tuple(image.shape)}")

    patches = image[:, : rows * patch_size, : cols * patch_size]
    patches = patches.reshape(3, rows, patch_size, cols, patch_size)
    patches = patches.permute(1, 3, 0, 2, 4).reshape(rows, cols, -1)
    scores = patches.float().std(dim=-1)

    if rows > border * 2 and cols > border * 2:
        masked = torch.full_like(scores, -1.0)
        masked[border : rows - border, border : cols - border] = scores[
            border : rows - border, border : cols - border
        ]
        scores = masked

    flat_idx = int(torch.argmax(scores).item())
    return flat_idx // cols, flat_idx % cols


def patch_target(
    image: torch.Tensor,
    row: int,
    col: int,
    patch_size: int,
    mode: str,
) -> torch.Tensor:
    y0 = row * patch_size
    y1 = (row + 1) * patch_size
    x0 = col * patch_size
    x1 = (col + 1) * patch_size
    patch = image[0, :, y0:y1, x0:x1]
    if patch.numel() == 0:
        raise ValueError(
            f"Patch ({row}, {col}) is outside decoded image shape {tuple(image.shape)}."
        )
    if mode == "mean":
        return patch.mean()
    if mode == "square":
        return patch.square().mean()
    channel = {"red": 0, "green": 1, "blue": 2}[mode]
    return patch[channel].mean()


def decoder_forward_with_context(
    decoder: torch.nn.Module,
    dino: torch.Tensor,
    latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_tokens = decoder.latent_tokenizer(latent)
    latent_tokens = latent_tokens + decoder.pos_embed
    dino_tokens = decoder._prepare_dino_tokens(dino)

    x = decoder.query_tokens.expand(latent.shape[0], -1, -1)
    x = x + decoder.query_pos_embed
    ctx_tokens = torch.cat([dino_tokens, latent_tokens], dim=1)
    ctx_tokens.retain_grad()
    for block in decoder.blocks:
        x = block(x, ctx_tokens)
    return decoder.tokens_to_image(x), ctx_tokens


def compute_attribution(
    decoder: torch.nn.Module,
    dino: torch.Tensor,
    latent: torch.Tensor,
    *,
    row: int,
    col: int,
    target_mode: str,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    decoder.zero_grad(set_to_none=True)
    image, ctx_tokens = decoder_forward_with_context(decoder, dino, latent)
    target = patch_target(image, row, col, decoder.patch_size, target_mode)
    target.backward()

    if ctx_tokens.grad is None:
        raise RuntimeError("Context token gradients were not retained.")
    token_attr = (ctx_tokens.grad * ctx_tokens).sum(dim=-1)[0].detach().float().cpu()
    num_patches = int(decoder.num_patches)
    grid_size = int(num_patches**0.5)
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Decoder num_patches={num_patches} is not a square grid.")

    dino_signed = token_attr[:num_patches].reshape(grid_size, grid_size).numpy()
    latent_signed = token_attr[num_patches:].reshape(grid_size, grid_size).numpy()
    dino_abs = np.abs(dino_signed)
    latent_abs = np.abs(latent_signed)
    return image.detach(), dino_signed, latent_signed, dino_abs, latent_abs


def attribution_metrics(
    *,
    sample_id: int,
    row: int,
    col: int,
    dino_signed: np.ndarray,
    latent_signed: np.ndarray,
    dino_abs: np.ndarray,
    latent_abs: np.ndarray,
) -> dict[str, float | int]:
    dino_abs_sum = float(dino_abs.sum())
    latent_abs_sum = float(latent_abs.sum())
    total = dino_abs_sum + latent_abs_sum
    dino_aligned_abs = float(dino_abs[row, col])
    latent_aligned_abs = float(latent_abs[row, col])
    return {
        "sample_id": sample_id,
        "patch_row": row,
        "patch_col": col,
        "dino_abs_sum": dino_abs_sum,
        "latent_abs_sum": latent_abs_sum,
        "dino_share": float(dino_abs_sum / total) if total > 0.0 else 0.0,
        "dino_aligned_abs": dino_aligned_abs,
        "latent_aligned_abs": latent_aligned_abs,
        "dino_nonlocal_abs": float(dino_abs_sum - dino_aligned_abs),
        "latent_nonlocal_abs": float(latent_abs_sum - latent_aligned_abs),
        "dino_aligned_signed": float(dino_signed[row, col]),
        "latent_aligned_signed": float(latent_signed[row, col]),
    }


def iter_requested_patches(
    explicit_patches: Iterable[tuple[int, int]] | None,
    image: torch.Tensor,
    patch_size: int,
) -> list[tuple[int, int]]:
    if explicit_patches:
        return list(explicit_patches)
    return [select_visible_patch(image[0].detach().cpu(), patch_size)]


def upload_to_huggingface(args: argparse.Namespace, output_dir: Path) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for --upload-to-hf. Install it or rerun without upload."
        ) from exc

    api = HfApi()
    api.create_repo(
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
        folder_path=str(output_dir),
        path_in_repo=args.hf_path_in_repo.strip("/") or ".",
    )


def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Decoder checkpoint not found: {checkpoint_path}")

    checkpoint_payload = load_checkpoint_payload(checkpoint_path)
    checkpoint_args = load_checkpoint_args(checkpoint_payload)
    checkpoint_key = select_checkpoint_key(args, checkpoint_payload)
    image_size = int(getattr(checkpoint_args, "decoder_output_image_size"))
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
    batch_size = resolve_batch_size(args.batch_size, checkpoint_args, world_size=1)
    pin_mem = bool(getattr(checkpoint_args, "pin_mem", True)) if args.pin_mem is None else args.pin_mem
    decoder_batch_prefetch = bool(getattr(checkpoint_args, "decoder_batch_prefetch", True))
    image_model_name = getattr(
        checkpoint_args,
        "image_model_name",
        "vit_base_patch16_dinov3.lvd1689m",
    )

    latent_store = inspect_feature_shards(args.feature_root, latent_dir_name)
    dino_store = inspect_feature_shards(args.feature_root, dino_dir_name)
    dataset = RamLoadedShardDataset(
        latent_store=latent_store,
        dino_store=dino_store,
        batch_size=batch_size,
        num_replicas=1,
        rank=0,
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
    image_mean, image_std = extract_image_normalization(data_loader)

    decoder = build_decoder_model_from_args(checkpoint_args).to(device)
    checkpoint_state, _stripped_prefix = resolve_decoder_state_dict(
        checkpoint_payload[checkpoint_key],
        decoder,
    )
    decoder.load_state_dict(checkpoint_state, strict=True)
    decoder.eval()

    output_dir = Path(args.output_dir).expanduser().resolve()
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    arrays_dir = output_dir / "arrays"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if args.save_npz:
        arrays_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.jsonl"
    all_metrics: list[dict[str, float | int | str]] = []
    processed_samples = 0
    progress = tqdm(data_loader, desc="decoder attribution")
    with summary_path.open("w", encoding="utf-8") as summary_file:
        for batch in progress:
            latent_batch = batch["latent"].to(device, non_blocking=True)
            dino_batch = batch["dino"].to(device, non_blocking=True)
            sample_ids = batch["sample_id"].cpu().numpy().astype(np.int64, copy=False)

            for item_idx, sample_id in enumerate(sample_ids.tolist()):
                if processed_samples >= args.num_images:
                    break

                latent = latent_batch[item_idx : item_idx + 1]
                dino = dino_batch[item_idx : item_idx + 1]

                with torch.no_grad():
                    preview = decoder.generate(latent, dino)
                patches = iter_requested_patches(args.patch, preview, decoder.patch_size)

                for row, col in patches:
                    if row >= int(decoder.num_patches**0.5) or col >= int(decoder.num_patches**0.5):
                        raise ValueError(
                            f"Patch ({row}, {col}) exceeds decoder grid for num_patches={decoder.num_patches}."
                        )

                    image, dino_signed, latent_signed, dino_abs, latent_abs = compute_attribution(
                        decoder,
                        dino,
                        latent,
                        row=row,
                        col=col,
                        target_mode=args.target,
                    )
                    recon = tensor_to_uint8_image(image[0].cpu(), image_mean, image_std)
                    stem = f"decoder_attr_sample{sample_id:08d}_patch{row:02d}_{col:02d}"
                    figure_path = figures_dir / f"{stem}.png"
                    save_figure(
                        figure_path,
                        recon=recon,
                        dino_abs=dino_abs,
                        latent_abs=latent_abs,
                        row=row,
                        col=col,
                        patch_size=decoder.patch_size,
                        grid_size=int(decoder.num_patches**0.5),
                    )

                    metrics = attribution_metrics(
                        sample_id=int(sample_id),
                        row=row,
                        col=col,
                        dino_signed=dino_signed,
                        latent_signed=latent_signed,
                        dino_abs=dino_abs,
                        latent_abs=latent_abs,
                    )
                    metrics.update(
                        {
                            "figure": str(figure_path.relative_to(output_dir)),
                            "target": args.target,
                            "checkpoint": str(checkpoint_path),
                            "checkpoint_key": checkpoint_key,
                            "split": args.split,
                            "latent_dir_name": latent_dir_name,
                            "dino_dir_name": dino_dir_name,
                        }
                    )
                    metrics_path = metrics_dir / f"{stem}.json"
                    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
                    summary_file.write(json.dumps(metrics) + "\n")
                    summary_file.flush()
                    all_metrics.append(metrics)

                    if args.save_npz:
                        np.savez_compressed(
                            arrays_dir / f"{stem}.npz",
                            dino_signed=dino_signed,
                            latent_signed=latent_signed,
                            dino_abs=dino_abs,
                            latent_abs=latent_abs,
                        )

                processed_samples += 1
                progress.set_postfix(samples=processed_samples)

            if processed_samples >= args.num_images:
                break

    manifest = {
        "num_samples": processed_samples,
        "num_figures": len(all_metrics),
        "target": args.target,
        "checkpoint": str(checkpoint_path),
        "checkpoint_key": checkpoint_key,
        "split": args.split,
        "feature_root": args.feature_root,
        "latent_dir_name": latent_dir_name,
        "dino_dir_name": dino_dir_name,
        "image_data_path": args.image_data_path,
        "figures_dir": str(figures_dir),
        "summary_jsonl": str(summary_path),
    }
    if all_metrics:
        dino_shares = np.asarray([float(item["dino_share"]) for item in all_metrics])
        manifest["mean_dino_share"] = float(dino_shares.mean())
        manifest["median_dino_share"] = float(np.median(dino_shares))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    if args.upload_to_hf:
        upload_to_huggingface(args, output_dir)
        print(
            json.dumps(
                {
                    "uploaded_to_hf": args.hf_repo_id,
                    "repo_type": args.hf_repo_type,
                    "path_in_repo": args.hf_path_in_repo,
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
