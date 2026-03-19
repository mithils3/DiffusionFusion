from contextlib import nullcontext
from itertools import islice
import math
import os
from pathlib import Path
import shutil
import sys
import time

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

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch_fidelity
except ImportError:
    torch_fidelity = None

_DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)
_JIT_FID_STATS_DIR = Path(__file__).resolve().parents[1] / "fid_stats"


def _dist_barrier():
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _forward_reconstruction(model, latent: torch.Tensor, dino: torch.Tensor) -> torch.Tensor:
    reconstructed = model(latent, dino)
    if not isinstance(reconstructed, torch.Tensor):
        raise TypeError(
            "Decoder training expects model(latent, dino) to return a reconstruction tensor."
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
    if not bool(getattr(args, "decoder_use_gan", True)):
        return None

    base_model = _unwrap_model(model)
    cached_state = getattr(base_model, "_decoder_gan_state", None)
    if cached_state is None:
        cached_state = build_decoder_gan_training_state(args, device)
        setattr(base_model, "_decoder_gan_state", cached_state)
    return cached_state


def _extract_image_normalization(data_loader):
    dataset = getattr(data_loader, "dataset", None)
    image_store = getattr(dataset, "image_store", None)
    transform = getattr(image_store, "transform", None)
    transform_steps = getattr(transform, "transforms", None)

    if transform_steps is not None:
        for step in reversed(transform_steps):
            mean = getattr(step, "mean", None)
            std = getattr(step, "std", None)
            if mean is not None and std is not None:
                return (
                    torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1),
                    torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1),
                )

    if misc.is_main_process():
        print(
            "Decoder eval could not infer image normalization from the data loader; "
            "falling back to ImageNet mean/std."
        )
    return (
        torch.tensor(_DEFAULT_IMAGE_MEAN, dtype=torch.float32).view(1, -1, 1, 1),
        torch.tensor(_DEFAULT_IMAGE_STD, dtype=torch.float32).view(1, -1, 1, 1),
    )


def _images_to_uint8(images, mean, std):
    images = images.detach().float().cpu()
    images = images * std + mean
    images = images.clamp_(0.0, 1.0)
    images = images.mul(255.0).round().to(torch.uint8)
    return images.permute(0, 2, 3, 1).numpy()


def _resolve_eval_data_loader(vae, data_loader):
    if data_loader is not None:
        return data_loader

    if vae is not None and hasattr(vae, "__iter__") and not hasattr(vae, "decode"):
        return vae

    raise ValueError(
        "Decoder evaluation requires an eval data_loader. Pass it via data_loader=..."
    )


def _resolve_decoder_reference_dir(args):
    for attr_name in (
        "decoder_eval_reference_dir",
        "eval_reference_dir",
        "reference_dir",
        "fid_ref_dir",
        "ref_dir",
    ):
        reference_dir = getattr(args, attr_name, None)
        if reference_dir:
            return reference_dir
    return None


def _resolve_decoder_fid_stats_file(args):
    for attr_name in (
        "decoder_eval_fid_stats",
        "eval_fid_stats",
        "fid_statistics_file",
    ):
        fid_stats_path = getattr(args, attr_name, None)
        if fid_stats_path:
            return fid_stats_path

    image_size = int(getattr(args, "decoder_output_image_size", 256))
    for candidate in (
        Path(f"/work/nvme/betw/msalunkhe/data/jit_in{image_size}_stats.npz"),
        _JIT_FID_STATS_DIR / f"jit_in{image_size}_stats.npz",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _raise_if_not_finite(loss_value: float, label: str):
    if not math.isfinite(loss_value):
        print(f"{label} is {loss_value}, stopping training")
        sys.exit(1)


def _reduce_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {name: misc.all_reduce_mean(value) for name, value in metrics.items()}


def _apply_discriminator_augment(images: torch.Tensor, gan_state: DecoderGanTrainingState) -> torch.Tensor:
    if gan_state.discriminator_augment is None:
        return images
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

    r1_weight = gan_state.loss_config.r1_weight
    if r1_weight > 0.0:
        real_disc_images = real_disc_images.float().requires_grad_(True)

    with _autocast_context(device):
        fake_logits, real_logits = discriminator(fake_disc_images, real_disc_images)
        disc_loss = hinge_discriminator_loss(real_logits, fake_logits)

    r1_penalty_value = 0.0
    if r1_weight > 0.0:
        r1_penalty = r1_gradient_penalty(real_logits.float(), real_disc_images)
        disc_loss = disc_loss + r1_weight * 0.5 * r1_penalty
        r1_penalty_value = float(r1_penalty.item())

    disc_loss_value = float(disc_loss.item())
    _raise_if_not_finite(disc_loss_value, "Discriminator loss")
    disc_loss.backward()
    gan_state.discriminator_optimizer.step()

    metrics = {
        "disc_loss": disc_loss_value,
        "disc_real_logit": float(real_logits.detach().mean().item()),
        "disc_fake_logit": float(fake_logits.detach().mean().item()),
    }
    if r1_weight > 0.0:
        metrics["r1_penalty"] = r1_penalty_value
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

        latent = batch["latent"].to(device, non_blocking=True)
        dino = batch["dino"].to(device, non_blocking=True)
        target_image = batch["image"].to(device, non_blocking=True)
        latent_input, dino_input = latent, dino
        if gan_state is not None:
            latent_input, dino_input = apply_noise_augmentation(
                latent_input,
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
                            latent_input,
                            dino_input,
                        )
                disc_metrics = _discriminator_step(
                    gan_state=gan_state,
                    real_images=target_image,
                    fake_images=reconstructed_for_disc,
                    image_mean=image_mean,
                    image_std=image_std,
                    device=device,
                )
                step_metrics.update(disc_metrics)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device):
            reconstructed = _forward_reconstruction(model, latent_input, dino_input)

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
        for key in ("disc_loss", "disc_real_logit", "disc_fake_logit", "r1_penalty"):
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


def evaluate(model_without_ddp, args, epoch, vae=None, batch_size=64, log_writer=None, wandb_run=None, wandb_step=None, data_loader=None):
    print("Start evaluation at epoch {}".format(epoch))
    model_without_ddp.eval()
    if not hasattr(model_without_ddp, "generate"):
        raise AttributeError(
            "Decoder evaluation expects model_without_ddp.generate(latent, dino) "
            "to return reconstructed images."
        )
    data_loader = _resolve_eval_data_loader(vae, data_loader)
    device = next(model_without_ddp.parameters()).device
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    eval_batch_size = getattr(getattr(data_loader, "dataset", None), "batch_size", batch_size)
    requested_num_images = int(getattr(args, "num_images", 0) or 0)
    total_available_steps = len(data_loader)
    total_available_images = total_available_steps * eval_batch_size * world_size
    target_num_images = requested_num_images if requested_num_images > 0 else total_available_images
    if target_num_images > total_available_images and misc.is_main_process():
        print(
            f"Requested {target_num_images} decoder eval images, but only "
            f"{total_available_images} are available. Evaluating on the available images."
        )
    target_num_images = min(target_num_images, total_available_images)
    if target_num_images <= 0:
        print("No decoder eval images available; skipping evaluation.")
        return

    num_steps = min(
        total_available_steps,
        math.ceil(target_num_images / (eval_batch_size * world_size)),
    )
    eval_image_interval = max(1, getattr(
        args, "wandb_eval_image_interval", 10))
    wandb_table = None
    if misc.is_main_process() and wandb_run is not None:
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Install it with `pip install wandb` or disable --use_wandb."
            )

        wandb_table = wandb.Table(
            columns=[
                "epoch",
                "global_index",
                "sample_id",
                "class_id",
                "target_image",
                "reconstruction",
            ]
        )

    eval_root = args.output_dir if getattr(args, "output_dir", None) else "."
    eval_output_dir = os.path.join(
        eval_root,
        f"decoder-eval-reconstructions-epoch{int(epoch):04d}-images{target_num_images}",
    )
    print("Save to:", eval_output_dir)
    if misc.is_main_process():
        if os.path.isdir(eval_output_dir):
            shutil.rmtree(eval_output_dir)
        os.makedirs(eval_output_dir, exist_ok=True)
    _dist_barrier()

    image_mean, image_std = _extract_image_normalization(data_loader)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("mse", misc.SmoothedValue(
        window_size=1, fmt="{value:.6f}"))
    header = "Eval: [{}]".format(epoch)
    print_freq = 20

    eval_iterable = islice(data_loader, num_steps)
    saved_images_local = 0
    with torch.no_grad():
        for step_idx, batch in enumerate(metric_logger.log_every(eval_iterable, print_freq, header, num_steps)):
            latent = batch["latent"].to(device, non_blocking=True)
            dino = batch["dino"].to(device, non_blocking=True)
            target_image = batch["image"].to(device, non_blocking=True)
            labels = batch.get("y")
            sample_ids = batch.get("sample_id")

            with _autocast_context(device):
                reconstructed = model_without_ddp.generate(latent, dino)
            batch_mse = torch.mean(
                (reconstructed.float() - target_image.float()) ** 2
            )
            metric_logger.update(mse=batch_mse.item())

            reconstructed_uint8 = _images_to_uint8(
                reconstructed, image_mean, image_std)
            target_uint8 = _images_to_uint8(target_image, image_mean, image_std)

            for sample_idx, reconstructed_image in enumerate(reconstructed_uint8):
                global_index = (
                    step_idx * world_size * eval_batch_size
                    + local_rank * eval_batch_size
                    + sample_idx
                )
                if global_index >= target_num_images:
                    continue

                Image.fromarray(reconstructed_image).save(
                    os.path.join(eval_output_dir, f"{str(global_index).zfill(5)}.png")
                )
                saved_images_local += 1

                if wandb_table is not None and global_index % eval_image_interval == 0:
                    class_id = int(labels[sample_idx]) if labels is not None else -1
                    sample_id = int(sample_ids[sample_idx]) if sample_ids is not None else -1
                    wandb_table.add_data(
                        epoch,
                        global_index,
                        sample_id,
                        class_id,
                        wandb.Image(
                            target_uint8[sample_idx],
                            caption=f"target class={class_id}, sample_id={sample_id}",
                        ),
                        wandb.Image(
                            reconstructed_image,
                            caption=f"recon class={class_id}, sample_id={sample_id}",
                        ),
                    )

    _dist_barrier()

    metric_logger.synchronize_between_processes()
    recon_mse = metric_logger.meters["mse"].global_avg

    saved_images_total = saved_images_local
    if misc.is_dist_avail_and_initialized():
        saved_images_tensor = torch.tensor(saved_images_local, device=device)
        torch.distributed.all_reduce(saved_images_tensor)
        saved_images_total = int(saved_images_tensor.item())

    if misc.is_main_process():
        print(
            f"Saved {saved_images_total} decoder eval reconstructions. "
            f"Mean reconstruction MSE: {recon_mse:.6f}"
        )

    postfix = "_decoder_res256"
    if log_writer is not None:
        log_writer.add_scalar("recon_mse{}".format(postfix), recon_mse, epoch)
    log_payload = {
        "eval/recon_mse{}".format(postfix): recon_mse,
    }
    if wandb_table is not None and len(wandb_table.data) > 0:
        log_payload["eval/samples{}".format(postfix)] = wandb_table

    reference_dir = _resolve_decoder_reference_dir(args)
    fid_statistics_file = _resolve_decoder_fid_stats_file(args)
    distribution_metrics_requested = bool(reference_dir or fid_statistics_file)
    metrics_done_path = os.path.join(
        eval_root,
        f".decoder_eval_metrics_done_epoch_{int(epoch):05d}",
    )
    if distribution_metrics_requested and misc.is_main_process() and os.path.exists(metrics_done_path):
        os.remove(metrics_done_path)
    if distribution_metrics_requested:
        _dist_barrier()

    if distribution_metrics_requested and misc.is_main_process():
        if torch_fidelity is None:
            raise ImportError(
                "torch_fidelity is not installed. Install it to run decoder evaluation metrics."
            )
        if reference_dir and not os.path.isdir(reference_dir):
            raise FileNotFoundError(
                f"Decoder eval reference directory not found: {reference_dir}"
            )
        if fid_statistics_file and not os.path.isfile(fid_statistics_file):
            raise FileNotFoundError(
                f"Decoder eval FID statistics file not found: {fid_statistics_file}"
            )

        metrics_kwargs = dict(
            input1=eval_output_dir,
            cuda=device.type == "cuda",
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=True,
        )
        if reference_dir:
            metrics_kwargs["input2"] = reference_dir
        else:
            metrics_kwargs["input2"] = None
            metrics_kwargs["fid_statistics_file"] = fid_statistics_file
        metrics_dict = torch_fidelity.calculate_metrics(**metrics_kwargs)
        fid = metrics_dict["frechet_inception_distance"]
        inception_score = metrics_dict["inception_score_mean"]
        if log_writer is not None:
            log_writer.add_scalar("fid{}".format(postfix), fid, epoch)
            log_writer.add_scalar("is{}".format(postfix), inception_score, epoch)
        log_payload["eval/fid{}".format(postfix)] = fid
        log_payload["eval/is{}".format(postfix)] = inception_score
        print(
            "Decoder eval metrics | Recon MSE: {:.6f}, FID: {:.4f}, Inception Score: {:.4f}".format(
                recon_mse, fid, inception_score
            )
        )
        with open(metrics_done_path, "w", encoding="utf-8") as f:
            f.write(f"{recon_mse},{fid},{inception_score}\n")
    elif distribution_metrics_requested:
        wait_start = time.time()
        wait_timeout = float(getattr(args, "dist_timeout_sec", 7200))
        while not os.path.exists(metrics_done_path):
            if time.time() - wait_start > wait_timeout:
                raise RuntimeError(
                    f"Timed out waiting for decoder eval metrics sync file: {metrics_done_path}"
                )
            time.sleep(1.0)
    elif misc.is_main_process():
        print("Decoder eval metrics | Recon MSE: {:.6f}".format(recon_mse))
        if getattr(args, "decoder_eval_metrics", True):
            print(
                "Skipping FID/IS because no decoder eval reference directory or FID stats file was provided. "
                "Set args.decoder_eval_reference_dir or args.decoder_eval_fid_stats."
            )

    if wandb_run is not None:
        if wandb_step is None:
            wandb_run.log(log_payload)
        else:
            wandb_run.log(log_payload, step=wandb_step)

    _dist_barrier()
    if misc.is_main_process():
        if os.path.isdir(eval_output_dir):
            shutil.rmtree(eval_output_dir)
        if distribution_metrics_requested and os.path.exists(metrics_done_path):
            os.remove(metrics_done_path)
