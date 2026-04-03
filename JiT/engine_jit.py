import math
import sys
import os
import shutil
import time

import torch
import numpy as np

import JiT.util.misc as misc
import JiT.util.lr_sched as lr_sched
from JiT.eval.diffusion_decoder import decode_with_decoder
import torch_fidelity
import copy
from PIL import Image
try:
    import wandb
except ImportError:
    wandb = None


def _iter_accumulation_groups(iterable, accum_iter: int, total_micro_batches: int):
    if accum_iter < 1:
        raise ValueError("accum_iter must be at least 1.")
    if total_micro_batches < 0:
        raise ValueError("total_micro_batches must be non-negative.")
    iterator = iter(iterable)
    emitted_micro_batches = 0

    while emitted_micro_batches < total_micro_batches:
        micro_batches_in_update = min(
            accum_iter, total_micro_batches - emitted_micro_batches
        )
        try:
            first_batch = next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"Expected {total_micro_batches} micro-batches, consumed {emitted_micro_batches}."
            ) from exc

        def group_batches(
            first_batch=first_batch,
            remaining=micro_batches_in_update - 1,
            emitted_before_group=emitted_micro_batches,
        ):
            yield first_batch
            for consumed_in_group in range(remaining):
                try:
                    yield next(iterator)
                except StopIteration as exc:
                    raise RuntimeError(
                        f"Expected {total_micro_batches} micro-batches, consumed "
                        f"{emitted_before_group + 1 + consumed_in_group}."
                    ) from exc

        emitted_micro_batches += micro_batches_in_update
        yield group_batches(), micro_batches_in_update

    try:
        next(iterator)
    except StopIteration:
        return
    raise RuntimeError(
        f"Expected {total_micro_batches} micro-batches, but data loader yielded extra batches."
    )


def train_one_epoch(
    model,
    model_without_ddp,
    data_loader,
    optimizer,
    device,
    epoch,
    log_writer=None,
    args=None,
    steps_per_epoch: int = None,
    optimizer_steps_per_epoch: int = None,
    wandb_run=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = max(1, int(getattr(args, "accum_iter", 1)))
    if steps_per_epoch is None or steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be a positive integer.")
    if optimizer_steps_per_epoch is None:
        optimizer_steps_per_epoch = (
            steps_per_epoch + accum_iter - 1
        ) // accum_iter
    if optimizer_steps_per_epoch <= 0:
        raise ValueError("optimizer_steps_per_epoch must be a positive integer.")

    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    optimizer_steps_completed = 0

    for optimizer_step, (micro_batches, micro_batches_in_update) in enumerate(
        metric_logger.log_every(
            _iter_accumulation_groups(data_loader, accum_iter, steps_per_epoch),
            print_freq,
            header,
            optimizer_steps_per_epoch,
        )
    ):
        # Per optimizer step (instead of per micro-batch) lr scheduler so accumulation
        # matches the reference large-batch JiT training recipe.
        lr = lr_sched.adjust_learning_rate(
            optimizer,
            optimizer_step / optimizer_steps_per_epoch + epoch,
            args,
        )
        accum_loss = 0.0

        for batch in micro_batches:
            # normalize image to [-1, 1]
            latent = batch["latent"].to(device, non_blocking=True)
            dino = batch["dino"].to(device, non_blocking=True)
            labels = batch["y"].to(device, non_blocking=True).view(-1).long()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(latent, dino, labels)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            accum_loss += loss_value
            (loss / micro_batches_in_update).backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model_without_ddp.update_ema()

        step_loss = accum_loss / micro_batches_in_update
        metric_logger.update(loss=step_loss)
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(step_loss)
        completed_optimizer_steps = optimizer_step + 1
        optimizer_steps_completed = completed_optimizer_steps

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int(
                (epoch + completed_optimizer_steps / optimizer_steps_per_epoch) * 1000
            )
            if optimizer_step % args.log_freq == 0:
                log_writer.add_scalar(
                    'train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
        if wandb_run is not None and optimizer_step % args.log_freq == 0:
            global_step = epoch * optimizer_steps_per_epoch + optimizer_step
            wandb_run.log({
                "train/loss": loss_value_reduce,
                "train/lr": lr,
                "train/epoch_progress": epoch + completed_optimizer_steps / optimizer_steps_per_epoch,
            }, step=global_step)
    if optimizer_steps_completed != optimizer_steps_per_epoch:
        raise RuntimeError(
            f"Expected {optimizer_steps_per_epoch} optimizer steps, got {optimizer_steps_completed}."
        )
    print(f"Finished ")


def evaluate(model_without_ddp, args, epoch, decoder, batch_size=64, log_writer=None, wandb_run=None, wandb_step=None):
    print("Start evaluation at epoch {}".format(epoch))
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    global_batch_size = batch_size * world_size
    num_steps = math.ceil(args.num_images / global_batch_size)
    eval_image_interval = max(1, getattr(
        args, "wandb_eval_image_interval", 10))
    wandb_table = None
    if misc.is_main_process() and wandb_run is not None:
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Install it with `pip install wandb` or disable --use_wandb."
            )
        wandb_table = wandb.Table(
            columns=["epoch", "global_index", "class_id", "image"]
        )

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.latent_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(
        0, class_num, dtype=np.int64).repeat(args.num_images // class_num)

    for step_idx in range(num_steps):
        print("Generation step {}/{}".format(step_idx, num_steps))

        start_idx = world_size * batch_size * step_idx + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen_np = class_label_gen_world[start_idx:min(end_idx, args.num_images)].copy()
        labels_gen = np.zeros(batch_size, dtype=np.int64)
        labels_gen[:labels_gen_np.shape[0]] = labels_gen_np
        labels_gen = torch.from_numpy(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_latents, sampled_dino = model_without_ddp.generate(
                labels_gen)

        torch.distributed.barrier()

        print("Decoding step {}/{}".format(step_idx, num_steps))
        sampled_images = decode_with_decoder(decoder, sampled_latents, sampled_dino)
        for sample_idx, sample in enumerate(sampled_images):
            index = sample_idx + world_size * batch_size * \
                step_idx + local_rank * batch_size
            if index >= args.num_images:
                continue

            Image.fromarray(sample).save(os.path.join(save_folder, '{}.png'.format(
                str(index).zfill(5))))
            if wandb_table is not None and index % eval_image_interval == 0:
                class_id = int(labels_gen_np[sample_idx])
                wandb_table.add_data(
                    epoch,
                    index,
                    class_id,
                    wandb.Image(
                        sample, caption=f"class={class_id}, idx={index}")
                )

    torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS on rank 0, while other ranks wait outside NCCL collectives
    metrics_requested = bool(getattr(args, "output_dir", None)) or bool(
        getattr(args, "use_wandb", False)
    )
    metrics_done_path = os.path.join(
        args.output_dir if getattr(args, "output_dir", None) else ".",
        f".eval_metrics_done_epoch_{int(epoch):05d}",
    )
    if metrics_requested and misc.is_main_process() and os.path.exists(metrics_done_path):
        os.remove(metrics_done_path)
    if metrics_requested:
        torch.distributed.barrier()

    if metrics_requested and misc.is_main_process():
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=args.fid_stats_path,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=True,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(
            model_without_ddp.cfg_scale, args.latent_size)
        if log_writer is not None:
            log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
            log_writer.add_scalar('is{}'.format(postfix),
                                  inception_score, epoch)
        if misc.is_main_process() and wandb_run is not None:
            log_payload = {
                'eval/fid{}'.format(postfix): fid,
                'eval/is{}'.format(postfix): inception_score,
            }
            if wandb_table is not None and len(wandb_table.data) > 0:
                log_payload['eval/samples{}'.format(postfix)] = wandb_table
            if wandb_step is None:
                wandb_run.log(log_payload)
            else:
                wandb_run.log(log_payload, step=wandb_step)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(
            fid, inception_score))
        with open(metrics_done_path, "w", encoding="utf-8") as f:
            f.write(f"{fid},{inception_score}\n")
    elif metrics_requested:
        wait_start = time.time()
        wait_timeout = float(getattr(args, "dist_timeout_sec", 7200))
        while not os.path.exists(metrics_done_path):
            if time.time() - wait_start > wait_timeout:
                raise RuntimeError(
                    f"Timed out waiting for eval metrics sync file: {metrics_done_path}"
                )
            time.sleep(1.0)

    torch.distributed.barrier()
    if misc.is_main_process() and metrics_requested:
        if os.path.isdir(save_folder):
            shutil.rmtree(save_folder)
        if os.path.exists(metrics_done_path):
            os.remove(metrics_done_path)
