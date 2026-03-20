import json
import math
import os
import shutil
import time
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, nullcontext
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import JiT.util.misc as misc
from JiT.decoder.gan import (
    DecoderGanTrainingState,
    apply_noise_augmentation,
    build_decoder_gan_training_state,
    calculate_adaptive_weight,
    get_decoder_last_layer,
    images_to_minus_one_to_one,
    set_requires_grad,
)
from JiT.decoder.losses import (
    build_decoder_loss_breakdown,
    hinge_discriminator_loss,
    r1_gradient_penalty,
    vanilla_generator_loss,
)

def _dist_barrier():
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@contextmanager
def _discriminator_forward_context(
    device: torch.device,
    *,
    needs_second_order: bool,
):
    """Use an unfused SDPA path when discriminator training needs second-order grads."""
    with ExitStack() as stack:
        if needs_second_order and device.type == "cuda":
            stack.enter_context(
                torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
            )
        else:
            stack.enter_context(_autocast_context(device))
        yield


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _forward_reconstruction(model, eva: torch.Tensor, dino: torch.Tensor) -> torch.Tensor:
    reconstructed = model(eva, dino)
    if not isinstance(reconstructed, torch.Tensor):
        raise TypeError(
            "Decoder training expects model(eva, dino) to return a reconstruction tensor."
        )
    return reconstructed


def _adjust_optimizer_learning_rate(
    optimizer,
    epoch_progress: float,
    *,
    lr: float,
    min_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    lr_schedule: str,
):
    if warmup_epochs > 0 and epoch_progress < warmup_epochs:
        current_lr = lr * epoch_progress / warmup_epochs
    else:
        if lr_schedule == "constant":
            current_lr = lr
        elif lr_schedule == "cosine":
            decay_span = max(total_epochs - warmup_epochs, 1)
            current_lr = min_lr + (lr - min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * (epoch_progress - warmup_epochs) / decay_span)
            )
        else:
            raise NotImplementedError(f"Unsupported lr schedule: {lr_schedule}")

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = current_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = current_lr
    return current_lr


def _resolve_gan_state(
    model,
    args,
    device: torch.device,
    gan_state: DecoderGanTrainingState | None,
) -> DecoderGanTrainingState | None:
    if gan_state is not None:
        return gan_state
    if not args.decoder_use_gan:
        return None

    base_model = _unwrap_model(model)
    cached_state = getattr(base_model, "_decoder_gan_state", None)
    if cached_state is None:
        cached_state = build_decoder_gan_training_state(args, device)
        setattr(base_model, "_decoder_gan_state", cached_state)
    return cached_state


def _extract_image_normalization(data_loader):
    try:
        transform_steps = data_loader.dataset.image_store.transform.transforms
    except AttributeError as exc:
        raise AttributeError(
            "Decoder dataloader must expose dataset.image_store.transform.transforms."
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
        "Decoder dataloader transform must include a normalization step with mean/std."
    )


def _images_to_uint8(images, mean, std):
    images = images.detach().float().cpu()
    images = images * std + mean
    images = images.clamp_(0.0, 1.0)
    images = images.mul(255.0).round().to(torch.uint8)
    return images.permute(0, 2, 3, 1).numpy()


def _raise_if_not_finite(loss_value: float, label: str):
    if not math.isfinite(loss_value):
        raise FloatingPointError(f"{label} is not finite: {loss_value}")


def _reduce_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {name: misc.all_reduce_mean(value) for name, value in metrics.items()}


def _require_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is not installed. Install it with `pip install wandb` or disable --use_wandb."
        ) from exc
    return wandb


def _require_pytorch_fid():
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError as exc:
        raise ImportError(
            "pytorch-fid is required for decoder evaluation. Install it with "
            "`pip install pytorch-fid` or from requirements.txt."
        ) from exc
    return calculate_fid_given_paths


def _save_uint8_pngs(images: np.ndarray, sample_ids: np.ndarray, output_dir: Path) -> None:
    for image_array, sample_id in zip(images, sample_ids.tolist(), strict=True):
        Image.fromarray(image_array).save(
            output_dir / f"{sample_id:08d}.png",
            format="PNG",
            compress_level=0,
        )


def _run_pytorch_fid(
    *,
    reference_dir: Path,
    recon_dir: Path,
    device: torch.device,
    batch_size: int,
    dims: int,
    num_workers: int,
) -> float:
    calculate_fid_given_paths = _require_pytorch_fid()
    return float(
        calculate_fid_given_paths(
            [str(reference_dir), str(recon_dir)],
            batch_size=batch_size,
            device=device,
            dims=dims,
            num_workers=num_workers,
        )
    )


def _apply_discriminator_augment(images: torch.Tensor, gan_state: DecoderGanTrainingState) -> torch.Tensor:
    gan_state.discriminator_augment.train(True)
    return gan_state.discriminator_augment(images)


def _discriminator_step(
    *,
    gan_state: DecoderGanTrainingState,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    device: torch.device,
    apply_r1: bool,
) -> dict[str, float]:
    if gan_state.loss_config.disc_loss != "hinge":
        raise NotImplementedError(
            f"Unsupported decoder discriminator loss: {gan_state.loss_config.disc_loss}"
        )

    discriminator = gan_state.discriminator
    discriminator.train(True)
    set_requires_grad(discriminator, True)
    gan_state.discriminator_optimizer.zero_grad(set_to_none=True)

    real_disc_images = images_to_minus_one_to_one(real_images.detach(), image_mean, image_std)
    fake_disc_images = images_to_minus_one_to_one(fake_images.detach(), image_mean, image_std)
    real_disc_images = _apply_discriminator_augment(real_disc_images, gan_state)
    fake_disc_images = _apply_discriminator_augment(fake_disc_images, gan_state)

    r1_weight = gan_state.loss_config.r1_weight if apply_r1 else 0.0
    if r1_weight > 0.0:
        real_disc_images = real_disc_images.float().requires_grad_(True)

    with _discriminator_forward_context(
        device,
        needs_second_order=r1_weight > 0.0,
    ):
        fake_logits, real_logits = discriminator(fake_disc_images, real_disc_images)
        disc_hinge_loss = hinge_discriminator_loss(real_logits, fake_logits)
        disc_loss = disc_hinge_loss

    r1_penalty_value = 0.0
    raw_r1_penalty_value = 0.0
    r1_loss_value = 0.0
    if r1_weight > 0.0:
        raw_r1_penalty = r1_gradient_penalty(real_logits.float(), real_disc_images)
        raw_r1_penalty_value = float(raw_r1_penalty.item())
        r1_loss = r1_weight * 0.5 * raw_r1_penalty
        disc_loss = disc_loss + r1_loss
        r1_penalty_value = raw_r1_penalty_value
        r1_loss_value = float(r1_loss.item())

    disc_loss_value = float(disc_loss.item())
    _raise_if_not_finite(disc_loss_value, "Discriminator loss")
    disc_loss.backward()
    gan_state.discriminator_optimizer.step()

    metrics = {
        "disc_loss": disc_loss_value,
        "disc_hinge_loss": float(disc_hinge_loss.item()),
        "disc_real_logit": float(real_logits.detach().mean().item()),
        "disc_fake_logit": float(fake_logits.detach().mean().item()),
        "r1_applied": 1.0 if r1_weight > 0.0 else 0.0,
        "r1_loss": r1_loss_value,
        "r1_penalty": r1_penalty_value,
        "r1_penalty_raw": raw_r1_penalty_value,
    }
    return metrics


def train_epoch(
    model,
    optimizer,
    log_writer,
    epoch,
    args,
    steps_per_epoch,
    wandb_run,
    data_loader,
    device,
    gan_state: DecoderGanTrainingState | None = None,
    post_step_callback: Callable[[], None] | None = None,
):
    model.train(True)
    model_without_ddp = _unwrap_model(model)
    gan_state = _resolve_gan_state(model, args, device, gan_state)
    if gan_state is not None:
        if gan_state.perceptual_loss is not None:
            gan_state.perceptual_loss.eval()
            set_requires_grad(gan_state.perceptual_loss, False)
        gan_state.discriminator.train(True)
        set_requires_grad(gan_state.discriminator, True)

    image_mean, image_std = _extract_image_normalization(data_loader)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    if gan_state is not None:
        metric_logger.add_meter("disc_lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, steps_per_epoch)
    ):
        epoch_progress = data_iter_step / steps_per_epoch + epoch
        lr = _adjust_optimizer_learning_rate(
            optimizer,
            epoch_progress,
            lr=float(args.lr),
            min_lr=float(args.min_lr),
            warmup_epochs=int(args.warmup_epochs),
            total_epochs=int(args.epochs),
            lr_schedule=str(args.lr_schedule),
        )

        disc_lr = None
        if gan_state is not None:
            disc_epoch_offset = gan_state.disc_lr_epoch_offset
            disc_epoch_progress = max(0.0, epoch_progress - disc_epoch_offset)
            disc_lr = _adjust_optimizer_learning_rate(
                gan_state.discriminator_optimizer,
                disc_epoch_progress,
                lr=gan_state.disc_lr,
                min_lr=gan_state.disc_min_lr,
                warmup_epochs=gan_state.disc_warmup_epochs,
                total_epochs=max(1, gan_state.disc_total_epochs - disc_epoch_offset),
                lr_schedule=gan_state.disc_lr_schedule,
            )

        eva = batch["eva"].to(device, non_blocking=True)
        dino = batch["dino"].to(device, non_blocking=True)
        target_image = batch["image"].to(device, non_blocking=True)
        eva_input, dino_input = eva, dino
        if gan_state is not None:
            eva_input, dino_input = apply_noise_augmentation(
                eva_input,
                dino_input,
                gan_state.noise_tau,
            )

        step_metrics: dict[str, float] = {}
        if gan_state is not None and gan_state.loss_config.discriminator_updates_enabled(epoch):
            for _ in range(gan_state.loss_config.disc_updates):
                with torch.no_grad():
                    with _autocast_context(device):
                        reconstructed_for_disc = _forward_reconstruction(
                            model,
                            eva_input,
                            dino_input,
                        )
                apply_r1 = gan_state.loss_config.r1_enabled_for_step(
                    gan_state.discriminator_step
                )
                disc_metrics = _discriminator_step(
                    gan_state=gan_state,
                    real_images=target_image,
                    fake_images=reconstructed_for_disc,
                    image_mean=image_mean,
                    image_std=image_std,
                    device=device,
                    apply_r1=apply_r1,
                )
                gan_state.discriminator_step += 1
                step_metrics.update(disc_metrics)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device):
            reconstructed = _forward_reconstruction(model, eva_input, dino_input)

        use_perceptual = gan_state is not None and gan_state.loss_config.perceptual_enabled(epoch)
        perceptual_module = gan_state.perceptual_loss if use_perceptual else None
        loss_breakdown = build_decoder_loss_breakdown(
            reconstructed,
            target_image,
            perceptual_loss_module=perceptual_module,
            image_mean=image_mean,
            image_std=image_std,
            perceptual_weight=(
                gan_state.loss_config.perceptual_weight if gan_state is not None else 0.0
            ),
            adversarial_weight=0.0,
            use_perceptual=use_perceptual,
            use_adversarial=False,
        )
        total_loss = loss_breakdown.total
        adversarial_loss = loss_breakdown.total.new_zeros(())
        disc_weight_value = 0.0

        if gan_state is not None and gan_state.loss_config.adversarial_enabled(epoch):
            if gan_state.loss_config.gen_loss != "vanilla":
                raise NotImplementedError(
                    f"Unsupported decoder generator loss: {gan_state.loss_config.gen_loss}"
                )

            discriminator = gan_state.discriminator
            discriminator.eval()
            set_requires_grad(discriminator, False)
            fake_disc_images = images_to_minus_one_to_one(reconstructed, image_mean, image_std)
            fake_disc_images = _apply_discriminator_augment(fake_disc_images, gan_state)
            with _autocast_context(device):
                fake_logits = discriminator(fake_disc_images)
                adversarial_loss = vanilla_generator_loss(fake_logits)

            disc_weight = loss_breakdown.total.new_tensor(gan_state.loss_config.disc_weight)
            if gan_state.loss_config.adaptive_weight:
                disc_weight = disc_weight * calculate_adaptive_weight(
                    loss_breakdown.total,
                    adversarial_loss,
                    get_decoder_last_layer(model_without_ddp),
                    gan_state.loss_config.max_d_weight,
                )
            disc_weight = disc_weight * gan_state.loss_config.adversarial_scale(epoch_progress)
            total_loss = total_loss + disc_weight * adversarial_loss
            disc_weight_value = float(disc_weight.detach().item())
            discriminator.train(True)
            set_requires_grad(discriminator, True)

        total_loss_value = float(total_loss.item())
        _raise_if_not_finite(total_loss_value, "Decoder loss")
        total_loss.backward()
        optimizer.step()
        if post_step_callback is not None:
            post_step_callback()

        metric_logger.update(
            loss=total_loss_value,
            l1=float(loss_breakdown.reconstruction.item()),
            mse=float(loss_breakdown.mse.item()),
            lr=lr,
        )
        step_metrics.update(
            {
                "loss": total_loss_value,
                "l1": float(loss_breakdown.reconstruction.item()),
                "mse": float(loss_breakdown.mse.item()),
            }
        )
        if use_perceptual:
            perceptual_value = float(loss_breakdown.perceptual.item())
            metric_logger.update(perceptual_loss=perceptual_value)
            step_metrics["perceptual_loss"] = perceptual_value
        if gan_state is not None:
            metric_logger.update(disc_lr=disc_lr)
            step_metrics["disc_lr"] = float(disc_lr)
            if gan_state.loss_config.adversarial_enabled(epoch):
                adv_value = float(adversarial_loss.item())
                metric_logger.update(generator_adv_loss=adv_value, disc_weight=disc_weight_value)
                step_metrics["generator_adv_loss"] = adv_value
                step_metrics["disc_weight"] = disc_weight_value
        for key in (
            "disc_loss",
            "disc_hinge_loss",
            "disc_real_logit",
            "disc_fake_logit",
            "r1_applied",
            "r1_loss",
            "r1_penalty",
            "r1_penalty_raw",
        ):
            if key in step_metrics:
                metric_logger.update(**{key: step_metrics[key]})

        # In DDP, every rank must enter the same reductions even though only
        # rank 0 owns the logger integrations.
        if misc.is_dist_avail_and_initialized():
            reduced_metrics = _reduce_metrics(step_metrics)
        else:
            reduced_metrics = step_metrics

        if log_writer is not None and data_iter_step % args.log_freq == 0:
            epoch_1000x = int(epoch_progress * 1000)
            for name, value in reduced_metrics.items():
                log_writer.add_scalar(f"train/{name}", value, epoch_1000x)

        if wandb_run is not None and data_iter_step % args.log_freq == 0:
            global_step = epoch * steps_per_epoch + data_iter_step
            payload = {
                f"train/{name}": value for name, value in reduced_metrics.items()
            }
            payload["train/epoch_progress"] = epoch_progress
            wandb_run.log(payload, step=global_step)

    print("Finished")


def evaluate(
    model_without_ddp,
    args,
    epoch,
    *,
    data_loader,
    log_writer=None,
    wandb_run=None,
    wandb_step=None,
):
    print("Start evaluation at epoch {}".format(epoch))
    model_without_ddp.eval()
    if not hasattr(model_without_ddp, "generate"):
        raise AttributeError(
            "Decoder evaluation expects model_without_ddp.generate(eva, dino) "
            "to return reconstructed images."
        )
    device = next(model_without_ddp.parameters()).device
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    eval_batch_size = data_loader.dataset.batch_size
    requested_num_images = int(args.num_images or 0)
    total_available_steps = len(data_loader)
    total_available_images = total_available_steps * eval_batch_size * world_size
    raw_dataset_size = data_loader.dataset.image_store.dataset_size
    max_evaluable_images = min(total_available_images, raw_dataset_size)
    target_num_images = requested_num_images if requested_num_images > 0 else max_evaluable_images
    if target_num_images > max_evaluable_images and misc.is_main_process():
        print(
            f"Requested {target_num_images} decoder eval images, but only "
            f"{max_evaluable_images} unique images are available. Evaluating on the available images."
        )
    target_num_images = min(target_num_images, max_evaluable_images)
    if target_num_images <= 0:
        print("No decoder eval images available; skipping evaluation.")
        return

    num_steps = min(
        total_available_steps,
        math.ceil(target_num_images / (eval_batch_size * world_size)),
    )
    eval_image_interval = max(1, args.wandb_eval_image_interval)
    wandb_table = None
    wandb_module = None
    if misc.is_main_process() and wandb_run is not None:
        wandb_module = _require_wandb()
        wandb_table = wandb_module.Table(
            columns=[
                "epoch",
                "global_index",
                "sample_id",
                "class_id",
                "target_image",
                "reconstruction",
            ]
        )

    eval_root = Path(args.output_dir if args.output_dir else ".").expanduser().resolve()
    eval_output_dir = eval_root / f"decoder-eval-epoch{int(epoch):04d}-images{target_num_images}"
    subset_tag = f"{getattr(args, 'image_data_split', 'train')}_{target_num_images}"
    reference_dir = eval_output_dir / f"reference_images_{subset_tag}"
    recon_dir = eval_output_dir / f"reconstructions_{subset_tag}"
    print(f"Save references to: {reference_dir}")
    print(f"Save reconstructions to: {recon_dir}")
    if misc.is_main_process():
        if eval_output_dir.is_dir():
            shutil.rmtree(eval_output_dir)
        reference_dir.mkdir(parents=True, exist_ok=True)
        recon_dir.mkdir(parents=True, exist_ok=True)
    _dist_barrier()

    image_mean, image_std = _extract_image_normalization(data_loader)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("mse", misc.SmoothedValue(
        window_size=1, fmt="{value:.6f}"))
    header = "Eval: [{}]".format(epoch)
    print_freq = 20

    eval_iterable = islice(data_loader, num_steps)
    local_mse_sum = 0.0
    local_mse_count = 0
    with torch.no_grad():
        for step_idx, batch in enumerate(metric_logger.log_every(eval_iterable, print_freq, header, num_steps)):
            eva = batch["eva"].to(device, non_blocking=True)
            dino = batch["dino"].to(device, non_blocking=True)
            target_image = batch["image"].to(device, non_blocking=True)
            labels = batch["y"]
            sample_ids = batch["sample_id"].cpu().numpy().astype(np.int64, copy=False)

            with _autocast_context(device):
                reconstructed = model_without_ddp.generate(eva, dino)
            batch_indices = (
                step_idx * world_size * eval_batch_size
                + local_rank * eval_batch_size
                + np.arange(reconstructed.shape[0], dtype=np.int64)
            )
            keep_mask = batch_indices < target_num_images
            if not np.any(keep_mask):
                break

            keep_tensor = torch.as_tensor(keep_mask, device=reconstructed.device)
            per_image_mse = (reconstructed.float() - target_image.float()).square().flatten(1).mean(dim=1)
            local_mse_sum += float(per_image_mse[keep_tensor].sum().item())
            local_mse_count += int(keep_mask.sum())
            metric_logger.update(mse=float(per_image_mse[keep_tensor].mean().item()))

            reconstructed_uint8 = _images_to_uint8(reconstructed, image_mean, image_std)[keep_mask]
            target_uint8 = _images_to_uint8(target_image, image_mean, image_std)[keep_mask]
            kept_sample_ids = sample_ids[keep_mask]
            _save_uint8_pngs(target_uint8, kept_sample_ids, reference_dir)
            _save_uint8_pngs(reconstructed_uint8, kept_sample_ids, recon_dir)

            if wandb_table is not None:
                kept_positions = np.flatnonzero(keep_mask)
                kept_global_indices = batch_indices[keep_mask]
                for kept_offset, sample_idx in enumerate(kept_positions.tolist()):
                    global_index = int(kept_global_indices[kept_offset])
                    if global_index % eval_image_interval != 0:
                        continue
                    class_id = int(labels[sample_idx])
                    sample_id = int(sample_ids[sample_idx])
                    wandb_table.add_data(
                        epoch,
                        global_index,
                        sample_id,
                        class_id,
                        wandb_module.Image(
                            target_uint8[kept_offset],
                            caption=f"target class={class_id}, sample_id={sample_id}",
                        ),
                        wandb_module.Image(
                            reconstructed_uint8[kept_offset],
                            caption=f"recon class={class_id}, sample_id={sample_id}",
                        ),
                    )

    _dist_barrier()

    mse_stats = torch.tensor(
        [local_mse_sum, float(local_mse_count)],
        dtype=torch.float64,
        device=device,
    )
    if misc.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(mse_stats)
    metric_logger.synchronize_between_processes()
    recon_mse = float(mse_stats[0].item() / max(mse_stats[1].item(), 1.0))
    saved_images_total = int(mse_stats[1].item())

    if misc.is_main_process():
        print(
            f"Saved {saved_images_total} decoder eval image pairs. "
            f"Mean reconstruction MSE: {recon_mse:.6f}"
        )

    postfix = f"_decoder_res{int(args.decoder_output_image_size)}"
    if log_writer is not None:
        log_writer.add_scalar("recon_mse{}".format(postfix), recon_mse, epoch)
    log_payload = {
        "eval/recon_mse{}".format(postfix): recon_mse,
    }
    if wandb_table is not None and len(wandb_table.data) > 0:
        log_payload["eval/samples{}".format(postfix)] = wandb_table

    distribution_metrics_requested = bool(args.decoder_eval_metrics)
    metrics_done_path = eval_root / f".decoder_eval_metrics_done_epoch_{int(epoch):05d}"
    if distribution_metrics_requested and misc.is_main_process() and metrics_done_path.exists():
        metrics_done_path.unlink()
    if distribution_metrics_requested:
        _dist_barrier()

    if distribution_metrics_requested and misc.is_main_process():
        fid = _run_pytorch_fid(
            reference_dir=reference_dir,
            recon_dir=recon_dir,
            device=device,
            batch_size=int(args.decoder_eval_fid_batch_size),
            dims=int(args.decoder_eval_fid_dims),
            num_workers=int(args.decoder_eval_fid_num_workers),
        )
        if log_writer is not None:
            log_writer.add_scalar("fid{}".format(postfix), fid, epoch)
        log_payload["eval/fid{}".format(postfix)] = fid
        metrics_summary = {
            "epoch": int(epoch),
            "split": getattr(args, "image_data_split", "train"),
            "num_images": int(target_num_images),
            "recon_mse": recon_mse,
            "fid": fid,
            "image_size": int(args.decoder_output_image_size),
            "reference_dir": str(reference_dir),
            "reconstructions_dir": str(recon_dir),
            "fid_batch_size": int(args.decoder_eval_fid_batch_size),
            "fid_dims": int(args.decoder_eval_fid_dims),
            "fid_num_workers": int(args.decoder_eval_fid_num_workers),
        }
        (eval_output_dir / "metrics.json").write_text(
            json.dumps(metrics_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "Decoder eval metrics | Recon MSE: {:.6f}, FID: {:.4f}".format(recon_mse, fid)
        )
        metrics_done_path.write_text(
            json.dumps(metrics_summary) + "\n",
            encoding="utf-8",
        )
    elif distribution_metrics_requested:
        wait_start = time.time()
        wait_timeout = float(args.dist_timeout_sec)
        while not metrics_done_path.exists():
            if time.time() - wait_start > wait_timeout:
                raise RuntimeError(
                    f"Timed out waiting for decoder eval metrics sync file: {metrics_done_path}"
                )
            time.sleep(1.0)
    elif misc.is_main_process():
        print("Decoder eval metrics | Recon MSE: {:.6f}".format(recon_mse))

    if wandb_run is not None:
        if wandb_step is None:
            wandb_run.log(log_payload)
        else:
            wandb_run.log(log_payload, step=wandb_step)

    _dist_barrier()
    if misc.is_main_process() and distribution_metrics_requested and metrics_done_path.exists():
        metrics_done_path.unlink()
