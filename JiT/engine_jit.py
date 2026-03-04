import math
import sys
import os
import shutil
import time

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy
from PIL import Image
try:
    import wandb
except ImportError:
    wandb = None


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None, steps_per_epoch: int = None, wandb_run=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header, steps_per_epoch)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / steps_per_epoch + epoch, args)

        # normalize image to [-1, 1]
        latent = batch["latent"].to(device, non_blocking=True)
        dino = batch["dino"].to(device, non_blocking=True)
        labels = batch["y"].to(device, non_blocking=True).view(-1).long()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(latent, dino, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int(
                (data_iter_step / steps_per_epoch + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar(
                    'train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
        if wandb_run is not None and data_iter_step % args.log_freq == 0:
            global_step = epoch * steps_per_epoch + data_iter_step
            wandb_run.log({
                "train/loss": loss_value_reduce,
                "train/lr": lr,
                "train/epoch_progress": epoch + data_iter_step / steps_per_epoch,
            }, step=global_step)
    print(f"Finished ")


def evaluate(model_without_ddp, args, epoch, vae, batch_size=64, log_writer=None, wandb_run=None, wandb_step=None):
    print("Start evaluation at epoch {}".format(epoch))
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1
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
        0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for step_idx in range(num_steps):
        print("Generation step {}/{}".format(step_idx, num_steps))

        start_idx = world_size * batch_size * step_idx + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen_np = labels_gen.copy()
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_latents, sampled_dino = model_without_ddp.generate(
                labels_gen)

        torch.distributed.barrier()

        print("Decoding step {}/{}".format(step_idx, num_steps))
        sampled_latents = vae.decode(
            sampled_latents / vae.config.scaling_factor).sample
        sampled_latents = torch.clamp(127.5 * sampled_latents + 128.0, 0, 255).permute(
            0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        for sample_idx, sample in enumerate(sampled_latents):
            index = sample_idx + world_size * batch_size * \
                step_idx + local_rank * batch_size
            if index >= args.num_images:
                continue

            Image.fromarray(sample).save(os.path.join(save_folder, '{}.jpg'.format(
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
        
        fid_statistics_file = '/work/nvme/betw/msalunkhe/data/jit_in256_stats.npz'
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
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
