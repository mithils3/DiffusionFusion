#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

from JiT.eval.vae_eval import (
    load_checkpoint_args,
    load_checkpoint_payload,
    resolve_checkpoint_path,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATION_EVAL_SCRIPT = _REPO_ROOT / "JiT" / "eval" / "vae_eval.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a staged JiT CFG sweep with image decoding, select the best 5k-image "
            "FID, then rerun that CFG at 50k images."
        )
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument(
        "--checkpoint-key",
        type=str,
        default="auto",
        choices=["auto", "model", "model_ema1", "model_ema2"],
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--fid-stats-path", type=str, default=None)
    parser.add_argument(
        "--decode-backend",
        type=str,
        default="vae",
        choices=["vae", "decoder"],
    )
    parser.add_argument(
        "--vae-pretrained-path",
        type=str,
        default="stabilityai/sdxl-vae",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--decoder-checkpoint", type=str, default=None)
    parser.add_argument(
        "--decoder-checkpoint-key",
        type=str,
        default="auto",
        choices=["auto", "model", "model_ema"],
    )
    parser.add_argument("--keep-images", action="store_true")
    parser.add_argument("--gen-bsz", type=int, default=128)
    parser.add_argument("--sweep-num-images", type=int, default=5000)
    parser.add_argument("--final-num-images", type=int, default=50000)
    parser.add_argument(
        "--cfg-values",
        type=str,
        default=None,
        help="Optional comma/space-separated explicit CFG values. Skips smart search.",
    )
    parser.add_argument(
        "--base-cfg",
        type=float,
        default=None,
        help="Center CFG for the smart search. Defaults to the checkpoint CFG.",
    )
    parser.add_argument(
        "--coarse-offsets",
        type=str,
        default="-0.8,-0.4,0.0,0.4,0.8",
        help="Comma/space-separated offsets added to --base-cfg for the coarse sweep.",
    )
    parser.add_argument("--refine-step", type=float, default=0.1)
    parser.add_argument("--refine-count", type=int, default=2)
    parser.add_argument("--cfg-min", type=float, default=1.0)
    parser.add_argument("--cfg-max", type=float, default=4.5)
    parser.add_argument("--interval-min", type=float, default=None)
    parser.add_argument("--interval-max", type=float, default=None)
    parser.add_argument("--num-sampling-steps", type=int, default=None)
    parser.add_argument("--sampling-method", type=str, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--inference-t-eps", type=float, default=None)
    parser.add_argument("--class-num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--nproc-per-node", type=str, default="auto")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    return parser.parse_args()


def parse_float_list(raw_value: str | None) -> list[float]:
    if raw_value is None:
        return []
    cleaned = raw_value.replace(",", " ")
    return [float(token) for token in cleaned.split()]


def normalize_cfg(value: float) -> float:
    return round(float(value), 3)


def cfg_tag(value: float) -> str:
    normalized = f"{normalize_cfg(value):.3f}"
    return normalized.replace("-", "m").replace(".", "p")


def dedupe_preserve_order(values: list[float]) -> list[float]:
    deduped: list[float] = []
    seen: set[float] = set()
    for value in values:
        normalized = normalize_cfg(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def build_initial_cfg_grid(
    base_cfg: float,
    cfg_min: float,
    cfg_max: float,
    coarse_offsets: list[float],
) -> list[float]:
    candidates = [
        min(cfg_max, max(cfg_min, base_cfg + offset))
        for offset in coarse_offsets
    ]
    return dedupe_preserve_order(candidates)


def build_refine_cfg_grid(
    best_cfg: float,
    evaluated_cfgs: set[float],
    cfg_min: float,
    cfg_max: float,
    refine_step: float,
    refine_count: int,
) -> list[float]:
    candidates: list[float] = []
    for distance in range(refine_count, 0, -1):
        candidates.append(best_cfg - distance * refine_step)
    for distance in range(1, refine_count + 1):
        candidates.append(best_cfg + distance * refine_step)

    refined: list[float] = []
    for candidate in candidates:
        bounded = normalize_cfg(min(cfg_max, max(cfg_min, candidate)))
        if bounded in evaluated_cfgs or bounded in refined:
            continue
        refined.append(bounded)
    return refined


def choose_best_result(results: list[dict], center_cfg: float) -> dict:
    if not results:
        raise ValueError("At least one sweep result is required to choose the best CFG.")

    return min(
        results,
        key=lambda result: (
            float(result["fid"]),
            -float(result.get("inception_score", 0.0)),
            abs(float(result["cfg"]) - center_cfg),
        ),
    )


def load_checkpoint_defaults(args: argparse.Namespace) -> tuple[Path, argparse.Namespace]:
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.resume_dir)
    checkpoint_payload = load_checkpoint_payload(checkpoint_path)
    checkpoint_args = load_checkpoint_args(checkpoint_payload)
    return checkpoint_path, checkpoint_args


def get_effective_class_num(
    cli_args: argparse.Namespace,
    checkpoint_args: argparse.Namespace,
) -> int:
    if cli_args.class_num is not None:
        return int(cli_args.class_num)
    if hasattr(checkpoint_args, "class_num"):
        return int(checkpoint_args.class_num)
    return 1000


def validate_image_count(num_images: int, class_num: int, label: str) -> None:
    if num_images <= 0:
        raise ValueError(f"{label} must be positive.")
    if num_images % class_num != 0:
        raise ValueError(
            f"{label} ({num_images}) must be divisible by class_num ({class_num})."
        )


def discover_metrics_path(run_dir: Path) -> Path | None:
    metric_files = sorted(run_dir.glob("*-metrics.json"))
    if not metric_files:
        return None
    if len(metric_files) == 1:
        return metric_files[0]
    return max(metric_files, key=lambda path: path.stat().st_mtime_ns)


def load_metrics_if_compatible(
    cli_args: argparse.Namespace,
    run_dir: Path,
    cfg: float,
    num_images: int,
) -> dict | None:
    metrics_path = discover_metrics_path(run_dir)
    if metrics_path is None:
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if normalize_cfg(float(metrics["cfg"])) != normalize_cfg(cfg):
        return None
    if int(metrics["num_images"]) != int(num_images):
        return None
    metrics_backend = str(metrics.get("decode_backend", "vae"))
    if metrics_backend != cli_args.decode_backend:
        return None
    if cli_args.decode_backend == "decoder":
        if cli_args.decoder_checkpoint is None:
            return None
        expected_decoder_checkpoint = str(
            Path(cli_args.decoder_checkpoint).expanduser().resolve()
        )
        if metrics.get("decoder_checkpoint") != expected_decoder_checkpoint:
            return None
        if metrics.get("decoder_checkpoint_key") != cli_args.decoder_checkpoint_key:
            return None
    elif metrics.get("vae_pretrained_path", cli_args.vae_pretrained_path) != cli_args.vae_pretrained_path:
        return None

    metrics["metrics_path"] = str(metrics_path)
    metrics["run_dir"] = str(run_dir)
    return metrics


def build_eval_command(
    cli_args: argparse.Namespace,
    run_dir: Path,
    cfg: float,
    num_images: int,
) -> list[str]:
    command = [
        "torchrun",
        "--nproc_per_node",
        str(cli_args.nproc_per_node),
        "--nnodes",
        str(cli_args.nnodes),
        "--node_rank",
        str(cli_args.node_rank),
        str(_GENERATION_EVAL_SCRIPT),
        "--output-dir",
        str(run_dir),
        "--checkpoint-key",
        cli_args.checkpoint_key,
        "--decode-backend",
        cli_args.decode_backend,
        "--vae-pretrained-path",
        cli_args.vae_pretrained_path,
        "--gen-bsz",
        str(cli_args.gen_bsz),
        "--num-images",
        str(num_images),
        "--cfg",
        str(cfg),
    ]

    if cli_args.checkpoint is not None:
        command.extend(["--checkpoint", cli_args.checkpoint])
    elif cli_args.resume_dir is not None:
        command.extend(["--resume-dir", cli_args.resume_dir])
    else:
        raise ValueError("Pass either --checkpoint or --resume-dir.")

    optional_pairs = [
        ("--fid-stats-path", cli_args.fid_stats_path),
        ("--decoder-checkpoint", cli_args.decoder_checkpoint),
        ("--decoder-checkpoint-key", cli_args.decoder_checkpoint_key),
        ("--interval-min", cli_args.interval_min),
        ("--interval-max", cli_args.interval_max),
        ("--num-sampling-steps", cli_args.num_sampling_steps),
        ("--sampling-method", cli_args.sampling_method),
        ("--noise-scale", cli_args.noise_scale),
        ("--inference-t-eps", cli_args.inference_t_eps),
        ("--class-num", cli_args.class_num),
        ("--seed", cli_args.seed),
    ]
    for flag, value in optional_pairs:
        if value is not None:
            command.extend([flag, str(value)])

    if cli_args.local_files_only:
        command.append("--local-files-only")
    if cli_args.keep_images:
        command.append("--keep-images")

    return command


def log(message: str) -> None:
    print(message, flush=True)


def write_summary(
    summary_path: Path,
    checkpoint_path: Path,
    base_cfg: float,
    sweep_results: list[dict],
    final_result: dict | None,
    cli_args: argparse.Namespace,
) -> None:
    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "resume_dir": cli_args.resume_dir,
        "output_dir": cli_args.output_dir,
        "decode_backend": cli_args.decode_backend,
        "base_cfg": normalize_cfg(base_cfg),
        "sweep_num_images": cli_args.sweep_num_images,
        "final_num_images": cli_args.final_num_images,
        "search_mode": "explicit" if cli_args.cfg_values else "smart",
        "cfg_values": [normalize_cfg(float(item["cfg"])) for item in sweep_results],
        "sweep_results": sweep_results,
        "best_sweep_result": choose_best_result(sweep_results, base_cfg) if sweep_results else None,
        "final_result": final_result,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_eval(
    cli_args: argparse.Namespace,
    stage_dir: Path,
    cfg: float,
    num_images: int,
) -> dict:
    run_dir = stage_dir / f"cfg_{cfg_tag(cfg)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not cli_args.force:
        existing_metrics = load_metrics_if_compatible(cli_args, run_dir, cfg, num_images)
        if existing_metrics is not None:
            log(
                f"Reusing existing metrics for cfg={normalize_cfg(cfg)} "
                f"at {num_images} images from {existing_metrics['metrics_path']}"
            )
            return existing_metrics

    command = build_eval_command(cli_args, run_dir, cfg, num_images)
    log(
        f"Launching eval for cfg={normalize_cfg(cfg)} with {num_images} images "
        f"into {run_dir}"
    )
    subprocess.run(command, cwd=_REPO_ROOT, check=True)

    metrics = load_metrics_if_compatible(cli_args, run_dir, cfg, num_images)
    if metrics is None:
        raise FileNotFoundError(
            f"Expected metrics JSON was not produced for cfg={cfg} in {run_dir}"
        )
    return metrics


def main() -> None:
    args = parse_args()
    if args.decode_backend == "decoder" and args.decoder_checkpoint is None:
        raise ValueError(
            "Decoder-backed CFG sweeps require --decoder-checkpoint."
        )
    checkpoint_path, checkpoint_args = load_checkpoint_defaults(args)

    base_cfg = (
        float(args.base_cfg)
        if args.base_cfg is not None
        else float(getattr(checkpoint_args, "cfg"))
    )
    class_num = get_effective_class_num(args, checkpoint_args)
    validate_image_count(args.sweep_num_images, class_num, "--sweep-num-images")
    validate_image_count(args.final_num_images, class_num, "--final-num-images")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "cfg_sweep_summary.json"

    explicit_cfg_values = dedupe_preserve_order(parse_float_list(args.cfg_values))
    sweep_results: list[dict] = []
    final_result: dict | None = None

    if explicit_cfg_values:
        planned_cfgs = explicit_cfg_values
        log(f"Running explicit CFG sweep: {planned_cfgs}")
        sweep_stage_dir = output_dir / f"sweep_{args.sweep_num_images}"
        for cfg in planned_cfgs:
            metrics = run_eval(args, sweep_stage_dir, cfg, args.sweep_num_images)
            metrics["search_stage"] = "explicit"
            sweep_results.append(metrics)
            write_summary(summary_path, checkpoint_path, base_cfg, sweep_results, final_result, args)
    else:
        coarse_offsets = parse_float_list(args.coarse_offsets)
        if not coarse_offsets:
            raise ValueError("Smart search requires at least one coarse offset.")

        coarse_cfgs = build_initial_cfg_grid(
            base_cfg,
            args.cfg_min,
            args.cfg_max,
            coarse_offsets,
        )
        coarse_stage_dir = output_dir / f"sweep_{args.sweep_num_images}" / "coarse"
        log(f"Running coarse CFG sweep around {normalize_cfg(base_cfg)}: {coarse_cfgs}")
        for cfg in coarse_cfgs:
            metrics = run_eval(args, coarse_stage_dir, cfg, args.sweep_num_images)
            metrics["search_stage"] = "coarse"
            sweep_results.append(metrics)
            write_summary(summary_path, checkpoint_path, base_cfg, sweep_results, final_result, args)

        best_after_coarse = choose_best_result(sweep_results, base_cfg)
        evaluated_cfgs = {normalize_cfg(float(item["cfg"])) for item in sweep_results}
        refine_cfgs = build_refine_cfg_grid(
            float(best_after_coarse["cfg"]),
            evaluated_cfgs,
            args.cfg_min,
            args.cfg_max,
            args.refine_step,
            args.refine_count,
        )
        if refine_cfgs:
            refine_stage_dir = output_dir / f"sweep_{args.sweep_num_images}" / "refine"
            log(
                f"Running refine CFG sweep around {normalize_cfg(float(best_after_coarse['cfg']))}: "
                f"{refine_cfgs}"
            )
            for cfg in refine_cfgs:
                metrics = run_eval(args, refine_stage_dir, cfg, args.sweep_num_images)
                metrics["search_stage"] = "refine"
                sweep_results.append(metrics)
                write_summary(summary_path, checkpoint_path, base_cfg, sweep_results, final_result, args)

    best_result = choose_best_result(sweep_results, base_cfg)
    best_cfg = float(best_result["cfg"])
    log(
        f"Best CFG from {args.sweep_num_images}-image sweep: "
        f"{normalize_cfg(best_cfg)} with FID {float(best_result['fid']):.4f}"
    )

    if not args.skip_final:
        final_stage_dir = output_dir / f"final_{args.final_num_images}"
        final_result = run_eval(args, final_stage_dir, best_cfg, args.final_num_images)
        log(
            f"Final {args.final_num_images}-image eval completed with cfg={normalize_cfg(best_cfg)} "
            f"and FID {float(final_result['fid']):.4f}"
        )
        final_result_path = output_dir / "cfg_sweep_final_result.json"
        final_result_path.write_text(json.dumps(final_result, indent=2) + "\n", encoding="utf-8")

    write_summary(summary_path, checkpoint_path, base_cfg, sweep_results, final_result, args)
    log(f"Saved sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
