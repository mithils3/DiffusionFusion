import math
import sys
import os
import shutil
import time
from itertools import islice

import torch
import numpy as np

import JiT.util.misc as misc
import JiT.util.lr_sched as lr_sched
from JiT.eval.diffusion_decoder import decode_with_decoder
import copy
from PIL import Image
try:
    import wandb
except ImportError:
    wandb = None
try:
    import torch_fidelity
except ImportError:
    torch_fidelity = None


def _iter_accumulation_groups(iterable, accum_iter: int, total_micro_batches: int):
    iterator = iter(iterable)
    remaining = total_micro_batches
    while remaining > 0:
        micro_batches = list(islice(iterator, min(accum_iter, remaining)))
        if not micro_batches:
            break
        remaining -= len(micro_batches)
        yield micro_batches, len(micro_batches)


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
    steps_per_epoch = steps_per_epoch or len(data_loader)
    optimizer_steps_per_epoch = optimizer_steps_per_epoch or math.ceil(
        steps_per_epoch / accum_iter
    )

    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

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
            payload = {
                "train/loss": loss_value_reduce,
                "train/lr": lr,
                "train/epoch_progress": epoch + completed_optimizer_steps / optimizer_steps_per_epoch,
            }
            misc.add_wandb_global_step(payload, global_step)
            wandb_run.log(payload)
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

    class_num = args.class_num
    class_label_gen_world = np.arange(
        0, class_num, dtype=np.int64).repeat(math.ceil(args.num_images / class_num))[:args.num_images]

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
        if torch_fidelity is None:
            raise ImportError("torch_fidelity is required for JiT generation metrics.")
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
            misc.add_wandb_global_step(log_payload, wandb_step)
            wandb_run.log(log_payload)
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
