import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb

from util.crop import center_crop_arr, transform
import util.misc as misc
from util.dataset import CustomDataset
import copy
from engine_jit import train_one_epoch, evaluate
from denoiser import Denoiser
from datasets import load_from_disk

from diffusers.models import AutoencoderKL

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float,
                        default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float,
                        default=0.0, help='Projection dropout rate')
    parser.add_argument('--vae_pretrained_path', type=str,
                        default='stabilityai/sdxl-vae')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=64, type=int)
    parser.add_argument('--prefetch_factor', default=4, type=int,
                        help='Number of batches each worker preloads')
    parser.add_argument('--persistent_workers', action='store_true',
                        help='Keep DataLoader workers alive across epochs')
    parser.add_argument('--no_persistent_workers',
                        action='store_false', dest='persistent_workers')
    parser.set_defaults(persistent_workers=True)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--ddp_bucket_cap_mb', default=100, type=int,
                        help='DDP gradient bucket size in MB')
    parser.add_argument('--ddp_broadcast_buffers', action='store_true',
                        help='Broadcast model buffers from rank 0 each forward')
    parser.add_argument('--no_ddp_broadcast_buffers',
                        action='store_false', dest='ddp_broadcast_buffers')
    parser.set_defaults(ddp_broadcast_buffers=False)
    parser.add_argument('--ddp_gradient_as_bucket_view', action='store_true',
                        help='Use DDP bucket views to reduce gradient memory copies')
    parser.add_argument('--no_ddp_gradient_as_bucket_view',
                        action='store_false', dest='ddp_gradient_as_bucket_view')
    parser.set_defaults(ddp_gradient_as_bucket_view=True)
    parser.add_argument('--ddp_static_graph', action='store_true',
                        help='Enable DDP static graph optimizations')
    parser.add_argument('--no_ddp_static_graph',
                        action='store_false', dest='ddp_static_graph')
    parser.set_defaults(ddp_static_graph=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--no_use_wandb', action='store_false', dest='use_wandb',
                        help='Disable Weights & Biases logging')
    parser.set_defaults(use_wandb=True)
    parser.add_argument('--wandb_project', type=str, default='jit',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity/team name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Optional Weights & Biases run name')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases mode')
    parser.add_argument('--wandb_eval_image_interval', type=int, default=10,
                        help='Log one generated eval image to W&B every N images')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')
    parser.add_argument('--dist_timeout_sec', default=7200, type=int,
                        help='Distributed process group timeout in seconds')

    return parser


def collate_fn(batch):
    # batch is list of dicts like {"image": tensor, "label": int, ...}
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([b.get("label", -1)
                          for b in batch], dtype=torch.long)
    return {"image": images, "label": labels}


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None
    wandb_run = None
    if global_rank == 0 and args.use_wandb:
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Install it with `pip install wandb` or disable --use_wandb."
            )
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
            dir=args.output_dir if args.output_dir else None,
            mode=args.wandb_mode,
        )

    # Data augmentation transforms
    hf_dataset = load_from_disk(args.data_path).select(range(100000))
    dataset_train = CustomDataset(hf_dataset=hf_dataset)
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
        drop_last=True,
    )

    loader_kwargs = dict(
        dataset=dataset_train,
        batch_size=args.batch_size,
        sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers
    data_loader_train = torch.utils.data.DataLoader(**loader_kwargs)

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)
    vae = AutoencoderKL.from_pretrained(args.vae_pretrained_path).to(device)
    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu],
        output_device=args.gpu,
        find_unused_parameters=False,
        bucket_cap_mb=args.ddp_bucket_cap_mb,
        broadcast_buffers=args.ddp_broadcast_buffers,
        gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
        static_graph=args.ddp_static_graph,
    )
    model_without_ddp = model.module

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(
        args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        if hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals([argparse.Namespace]):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda(
        ) for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda(
        ) for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(
            list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(
            list(model_without_ddp.parameters()))
        print("Training from scratch")

    try:
        # Evaluate generation
        if args.evaluate_gen:
            print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                with torch.no_grad():
                    evaluate(
                        model_without_ddp,
                        args,
                        0,
                        batch_size=args.gen_bsz,
                        log_writer=log_writer,
                        vae=vae,
                        wandb_run=wandb_run,
                    )
            return

        # Training loop
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        steps_per_epoch = len(data_loader_train)
        for epoch in range(args.start_epoch, args.epochs):
            sampler_train.set_epoch(epoch)

            train_one_epoch(
                model,
                model_without_ddp,
                data_loader_train,
                optimizer,
                device,
                epoch,
                log_writer=log_writer,
                args=args,
                steps_per_epoch=steps_per_epoch,
                wandb_run=wandb_run,
            )

            # Save checkpoint periodically
            did_save_checkpoint = False
            if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
                misc.save_model(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    epoch=epoch,
                    epoch_name="last"
                )
                did_save_checkpoint = True

            if epoch % 50 == 0 and epoch > 0:
                misc.save_model(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    epoch=epoch
                )
                did_save_checkpoint = True

            # Keep ranks in lockstep after rank-0 checkpoint I/O before eval/next epoch.
            if did_save_checkpoint and args.distributed:
                torch.distributed.barrier()

            # Perform online evaluation at specified intervals
            if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
                torch.cuda.empty_cache()
                with torch.no_grad():
                    evaluate(
                        model_without_ddp,
                        args,
                        epoch,
                        batch_size=args.gen_bsz,
                        log_writer=log_writer,
                        vae=vae,
                        wandb_run=wandb_run,
                        wandb_step=(epoch + 1) * steps_per_epoch,
                    )
                torch.cuda.empty_cache()

            if misc.is_main_process() and log_writer is not None:
                log_writer.flush()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time:', total_time_str)
        if wandb_run is not None:
            wandb_run.log({
                "train/total_time_sec": total_time,
                "train/total_time_hms": total_time_str,
            })
    finally:
        if log_writer is not None:
            log_writer.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
