#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import shutil
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from JiT.denoiser import Denoiser
from JiT.eval.diffusion_decoder import decode_with_decoder, load_decoder_for_eval


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


_FID_STATS_DIR = Path(__file__).resolve().parents[1] / "fid_stats"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained JiT checkpoint by decoding generated samples with either "
            "an SDXL VAE or the trained JiT decoder and computing FID/IS."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a JiT checkpoint. Overrides --resume-dir when both are provided.",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="JiT training output directory containing checkpoint-last.pth.",
    )
    parser.add_argument(
        "--checkpoint-key",
        type=str,
        default="auto",
        choices=["auto", "model", "model_ema1", "model_ema2"],
        help="Checkpoint weights to evaluate. Defaults to model_ema1 when present.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--fid-stats-path",
        type=str,
        default=None,
        help="Path to torch-fidelity FID stats. Defaults to JiT/fid_stats/jit_in{image_size}_stats.npz.",
    )
    parser.add_argument(
        "--decode-backend",
        type=str,
        default="vae",
        choices=["vae", "decoder"],
        help="Image decoder used for generation evaluation.",
    )
    parser.add_argument(
        "--vae-pretrained-path",
        type=str,
        default="stabilityai/sdxl-vae",
        help="Diffusers SDXL VAE model id or local path.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load the SDXL VAE from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        type=str,
        default=None,
        help="Path to a trained JiT decoder checkpoint used when --decode-backend=decoder.",
    )
    parser.add_argument(
        "--decoder-checkpoint-key",
        type=str,
        default="auto",
        choices=["auto", "model", "model_ema"],
        help="Decoder checkpoint state dict key to load when --decode-backend=decoder.",
    )
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument(
        "--gen-bsz",
        type=int,
        default=128,
        help="Per-process generation batch size.",
    )
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--interval-min", type=float, default=None)
    parser.add_argument("--interval-max", type=float, default=None)
    parser.add_argument("--num-sampling-steps", type=int, default=None)
    parser.add_argument("--sampling-method", type=str, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--inference-t-eps", type=float, default=None)
    parser.add_argument("--class-num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Keep the generated PNGs after metric computation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Single-process device. Ignored under torchrun.",
    )
    parser.add_argument(
        "--dist-timeout-sec",
        type=int,
        default=7200,
        help="Distributed process group timeout in seconds.",
    )
    return parser.parse_args()


def resolve_default_fid_stats_path(latent_size: int) -> str | None:
    image_size = latent_size * 8
    candidate = _FID_STATS_DIR / f"jit_in{image_size}_stats.npz"
    if candidate.is_file():
        return str(candidate)
    return None


def init_distributed(
    device_arg: str,
    timeout_sec: int,
) -> tuple[torch.device, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed evaluation requires CUDA.")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            "nccl",
            timeout=datetime.timedelta(seconds=int(timeout_sec)),
            device_id=local_rank,
        )
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


def resolve_checkpoint_path(
    checkpoint: str | None,
    resume_dir: str | None,
) -> Path:
    if checkpoint:
        return Path(checkpoint).expanduser().resolve()
    if resume_dir:
        return (Path(resume_dir).expanduser().resolve() / "checkpoint-last.pth")
    raise ValueError("Pass either --checkpoint or --resume-dir.")


def load_checkpoint_payload(checkpoint_path: Path) -> dict:
    if hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals([argparse.Namespace]):
            return torch.load(checkpoint_path, map_location="cpu")
    return torch.load(checkpoint_path, map_location="cpu")


def load_checkpoint_args(checkpoint_payload: dict) -> argparse.Namespace:
    args_payload = checkpoint_payload.get("args")
    if isinstance(args_payload, argparse.Namespace):
        return deepcopy(args_payload)
    if isinstance(args_payload, dict):
        return argparse.Namespace(**args_payload)
    raise KeyError(
        "JiT checkpoint does not contain a supported `args` payload. "
        "Expected argparse.Namespace or dict."
    )


def select_checkpoint_key(args: argparse.Namespace, checkpoint_payload: dict) -> str:
    if args.checkpoint_key == "auto":
        if "model_ema1" in checkpoint_payload:
            return "model_ema1"
        return "model"
    if args.checkpoint_key not in checkpoint_payload:
        raise KeyError(
            f"Checkpoint key `{args.checkpoint_key}` not found in the JiT checkpoint."
        )
    return args.checkpoint_key


def apply_generation_overrides(
    checkpoint_args: argparse.Namespace,
    cli_args: argparse.Namespace,
) -> argparse.Namespace:
    eval_args = deepcopy(checkpoint_args)
    overrides = {
        "cfg": cli_args.cfg,
        "interval_min": cli_args.interval_min,
        "interval_max": cli_args.interval_max,
        "num_sampling_steps": cli_args.num_sampling_steps,
        "sampling_method": cli_args.sampling_method,
        "noise_scale": cli_args.noise_scale,
        "inference_t_eps": cli_args.inference_t_eps,
        "class_num": cli_args.class_num,
        "seed": cli_args.seed,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(eval_args, name, value)
    return eval_args


def resolve_denoiser_state_dict(
    checkpoint_state: dict,
    model: torch.nn.Module,
) -> tuple[dict, str | None]:
    if not isinstance(checkpoint_state, dict):
        raise TypeError(
            "JiT checkpoint entry must be a state dict mapping, "
            f"got {type(checkpoint_state).__name__}."
        )

    expected_keys = set(model.state_dict().keys())
    candidate_state = dict(checkpoint_state)
    stripped_segments: list[str] = []

    for _ in range(4):
        if set(candidate_state.keys()) == expected_keys:
            stripped_prefix = ".".join(stripped_segments) if stripped_segments else None
            return candidate_state, stripped_prefix
        if not candidate_state or any("." not in key for key in candidate_state):
            break

        head_segments = {key.split(".", 1)[0] for key in candidate_state}
        if len(head_segments) != 1:
            break

        segment = next(iter(head_segments))
        stripped_segments.append(segment)
        candidate_state = {
            key.split(".", 1)[1]: value
            for key, value in candidate_state.items()
        }

    for prefix in ("module.", "_orig_mod.", "model.", "module.model.", "_orig_mod.model."):
        filtered_state = {
            key[len(prefix):]: value
            for key, value in checkpoint_state.items()
            if key.startswith(prefix)
        }
        if set(filtered_state.keys()) == expected_keys:
            return filtered_state, prefix.rstrip(".")

    raise RuntimeError("JiT checkpoint state dict is incompatible with the evaluation model.")


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_vae_decoder(
    pretrained_path: str,
    device: torch.device,
    local_files_only: bool,
):
    try:
        from diffusers.models import AutoencoderKL
    except ImportError as exc:
        raise ImportError(
            "diffusers is required for latent-only VAE evaluation. "
            "Install it from requirements.txt."
        ) from exc

    vae = AutoencoderKL.from_pretrained(
        pretrained_path,
        local_files_only=local_files_only,
    ).to(device)
    vae.eval()
    vae.enable_slicing()
    return vae


def decode_latents_with_vae(vae, latents: torch.Tensor) -> np.ndarray:
    device = latents.device
    with autocast_context(device):
        images = vae.decode(latents / vae.config.scaling_factor).sample
    images = torch.clamp(127.5 * images + 128.0, 0.0, 255.0)
    return images.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


def load_generation_decoder(
    args: argparse.Namespace,
    device: torch.device,
):
    if args.decode_backend == "vae":
        return load_vae_decoder(
            args.vae_pretrained_path,
            device,
            args.local_files_only,
        )

    if args.decoder_checkpoint is None:
        raise ValueError(
            "Decoder-backed generation evaluation requires --decoder-checkpoint."
        )
    decoder_checkpoint_path = str(Path(args.decoder_checkpoint).expanduser().resolve())
    args.decoder_checkpoint = decoder_checkpoint_path
    return load_decoder_for_eval(
        decoder_checkpoint_path,
        device,
        args.decoder_checkpoint_key,
    )


def decode_generated_images(
    args: argparse.Namespace,
    image_decoder,
    sampled_latents: torch.Tensor,
    sampled_dino: torch.Tensor,
) -> np.ndarray:
    if args.decode_backend == "vae":
        return decode_latents_with_vae(image_decoder, sampled_latents)
    return decode_with_decoder(image_decoder, sampled_latents, sampled_dino)


def save_uint8_pngs(
    images: np.ndarray,
    indices: np.ndarray,
    output_dir: Path,
) -> None:
    for image_array, sample_idx in zip(images, indices.tolist(), strict=True):
        Image.fromarray(image_array).save(
            output_dir / f"{sample_idx:05d}.png",
            format="PNG",
            compress_level=0,
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


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device, rank, world_size, is_distributed = init_distributed(
        args.device,
        args.dist_timeout_sec,
    )
    try:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.resume_dir)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"JiT checkpoint not found: {checkpoint_path}")

        output_dir = Path(args.output_dir).expanduser().resolve()
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        barrier_if_distributed(is_distributed, device)

        log_rank0(rank, f"Loading JiT checkpoint from {checkpoint_path}")
        checkpoint_payload = load_checkpoint_payload(checkpoint_path)
        checkpoint_args = load_checkpoint_args(checkpoint_payload)
        eval_args = apply_generation_overrides(checkpoint_args, args)
        checkpoint_key = select_checkpoint_key(args, checkpoint_payload)
        checkpoint_epoch = int(checkpoint_payload.get("epoch", -1))

        if args.fid_stats_path is None:
            args.fid_stats_path = resolve_default_fid_stats_path(int(eval_args.latent_size))
        if args.fid_stats_path is None:
            raise FileNotFoundError(
                "No FID statistics file was provided and no built-in JiT stats file matches "
                f"latent_size={eval_args.latent_size}."
            )
        fid_stats_path = Path(args.fid_stats_path).expanduser().resolve()
        if not fid_stats_path.is_file():
            raise FileNotFoundError(f"FID statistics file not found: {fid_stats_path}")

        base_seed = int(getattr(eval_args, "seed", 0))
        torch.manual_seed(base_seed + rank)
        np.random.seed(base_seed + rank)

        class_num = int(eval_args.class_num)
        if args.num_images <= 0:
            raise ValueError("--num-images must be positive.")
        if args.gen_bsz <= 0:
            raise ValueError("--gen-bsz must be positive.")
        if args.num_images % class_num != 0:
            raise ValueError(
                f"--num-images ({args.num_images}) must be divisible by class_num ({class_num})."
            )

        model = Denoiser(eval_args).to(device)
        checkpoint_state, stripped_prefix = resolve_denoiser_state_dict(
            checkpoint_payload[checkpoint_key],
            model,
        )
        model.load_state_dict(checkpoint_state, strict=True)
        model.eval()

        image_decoder = load_generation_decoder(args, device)

        run_tag = (
            f"{model.method}-steps{model.steps}-cfg{model.cfg_scale}"
            f"-interval{model.cfg_interval[0]}-{model.cfg_interval[1]}"
            f"-image{args.num_images}-res{eval_args.latent_size}-{args.decode_backend}"
        )
        save_folder = output_dir / run_tag
        if rank == 0:
            save_folder.mkdir(parents=True, exist_ok=True)
        barrier_if_distributed(is_distributed, device)

        prefix_message = ""
        if stripped_prefix is not None:
            prefix_message = f" after stripping `{stripped_prefix}.`"
        log_rank0(
            rank,
            f"Evaluating checkpoint key `{checkpoint_key}` from epoch {checkpoint_epoch}{prefix_message}.",
        )
        if args.decode_backend == "vae":
            log_rank0(rank, f"Decoding JiT latents with VAE `{args.vae_pretrained_path}`")
        else:
            log_rank0(
                rank,
                f"Decoding JiT latents + DINO with decoder `{args.decoder_checkpoint}` "
                f"(key `{args.decoder_checkpoint_key}`)",
            )
        log_rank0(rank, f"Writing temporary samples to {save_folder}")

        global_batch_size = args.gen_bsz * world_size
        num_steps = math.ceil(args.num_images / global_batch_size)
        class_label_gen_world = np.arange(
            0,
            class_num,
            dtype=np.int64,
        ).repeat(args.num_images // class_num)

        for step_idx in range(num_steps):
            if rank == 0:
                print(f"Generation step {step_idx + 1}/{num_steps}", flush=True)

            start_idx = global_batch_size * step_idx + rank * args.gen_bsz
            end_idx = start_idx + args.gen_bsz
            labels_gen_np = class_label_gen_world[start_idx:min(end_idx, args.num_images)].copy()
            if labels_gen_np.size == 0:
                continue

            labels_gen = np.zeros(args.gen_bsz, dtype=np.int64)
            labels_gen[:labels_gen_np.shape[0]] = labels_gen_np
            labels_gen_tensor = torch.from_numpy(labels_gen).to(device=device, dtype=torch.long)

            with autocast_context(device):
                sampled_latents, sampled_dino = model.generate(labels_gen_tensor)

            sampled_images = decode_generated_images(
                args,
                image_decoder,
                sampled_latents,
                sampled_dino,
            )
            batch_indices = np.arange(
                start_idx,
                start_idx + sampled_images.shape[0],
                dtype=np.int64,
            )
            keep_mask = batch_indices < args.num_images
            if np.any(keep_mask):
                save_uint8_pngs(
                    sampled_images[keep_mask],
                    batch_indices[keep_mask],
                    save_folder,
                )

        barrier_if_distributed(is_distributed, device)

        if rank == 0:
            metrics_dict = run_torch_fidelity_metrics(
                save_folder,
                fid_stats_path,
                device,
            )
            summary = {
                "fid": float(metrics_dict["frechet_inception_distance"]),
                "inception_score": float(metrics_dict["inception_score_mean"]),
                "num_images": args.num_images,
                "gen_bsz": args.gen_bsz,
                "world_size": world_size,
                "checkpoint": str(checkpoint_path),
                "checkpoint_key": checkpoint_key,
                "checkpoint_epoch": checkpoint_epoch,
                "fid_stats_path": str(fid_stats_path),
                "decode_backend": args.decode_backend,
                "vae_pretrained_path": (
                    args.vae_pretrained_path if args.decode_backend == "vae" else None
                ),
                "decoder_checkpoint": (
                    args.decoder_checkpoint if args.decode_backend == "decoder" else None
                ),
                "decoder_checkpoint_key": (
                    args.decoder_checkpoint_key if args.decode_backend == "decoder" else None
                ),
                "sampling_method": model.method,
                "num_sampling_steps": model.steps,
                "cfg": float(model.cfg_scale),
                "interval_min": float(model.cfg_interval[0]),
                "interval_max": float(model.cfg_interval[1]),
                "class_num": class_num,
                "latent_size": int(eval_args.latent_size),
                "seed": base_seed,
                "image_dir": str(save_folder) if args.keep_images else None,
                "temporary_image_dir": str(save_folder),
                "keep_images": bool(args.keep_images),
            }
            metrics_path = output_dir / f"{run_tag}-metrics.json"
            metrics_path.write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, indent=2), flush=True)
            print(f"Saved metrics to {metrics_path}", flush=True)

            if not args.keep_images:
                shutil.rmtree(save_folder)
                print(f"Removed temporary sample directory {save_folder}", flush=True)

        barrier_if_distributed(is_distributed, device)
    finally:
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
