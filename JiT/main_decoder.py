import argparse
import copy
import datetime
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import JiT.util.misc as misc
from JiT.decoder import Decoder, DecoderReconstructionModel, load_decoder_plan_config
from JiT.decoder.dataset import RamLoadedShardDataset, inspect_feature_shards
from JiT.decoder.train import evaluate, train_epoch

try:
    import wandb
except ImportError:
    wandb = None


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "decoder" / "default_config.yaml"


@torch.no_grad()
def update_ema_model(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float,
) -> None:
    for ema_param, model_param in zip(ema_model.parameters(), model.parameters(), strict=True):
        ema_param.mul_(decay).add_(model_param.detach(), alpha=1.0 - decay)
    for ema_buffer, model_buffer in zip(ema_model.buffers(), model.buffers(), strict=True):
        ema_buffer.copy_(model_buffer.detach())


def build_decoder_ema_model(
    model_without_ddp: torch.nn.Module,
    device: torch.device,
) -> torch.nn.Module:
    ema_model = copy.deepcopy(model_without_ddp).to(device)
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


def get_args_parser() -> argparse.ArgumentParser:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG_PATH),
    )
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    plan = load_decoder_plan_config(bootstrap_args.config)

    decoder_defaults = plan.decoder
    training_defaults = plan.training
    optimizer_defaults = training_defaults.optimizer
    scheduler_defaults = training_defaults.scheduler
    disc_defaults = plan.gan.disc
    loss_defaults = plan.gan.loss

    parser = argparse.ArgumentParser("JiT decoder", add_help=True)
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG_PATH),
        help="Path to decoder YAML config.",
    )

    # decoder architecture
    parser.add_argument("--latent_size", default=32, type=int)
    parser.add_argument("--dino_hidden_size", default=decoder_defaults.dino_hidden_size, type=int)
    parser.add_argument("--latent_in_channels", default=4, type=int)
    parser.add_argument("--image_out_channels", default=3, type=int)
    parser.add_argument("--bottleneck_dim", default=128, type=int)
    parser.add_argument("--attn_dropout", default=0.0, type=float)
    parser.add_argument("--proj_dropout", default=0.0, type=float)
    parser.add_argument("--decoder_hidden_size", default=decoder_defaults.hidden_size, type=int)
    parser.add_argument("--decoder_depth", default=decoder_defaults.depth, type=int)
    parser.add_argument("--decoder_num_heads", default=decoder_defaults.num_heads, type=int)
    parser.add_argument("--decoder_mlp_ratio", default=decoder_defaults.mlp_ratio, type=float)
    parser.add_argument("--decoder_patch_size", default=decoder_defaults.patch_size, type=int)
    parser.add_argument(
        "--decoder_latent_patch_size",
        default=decoder_defaults.latent_patch_size,
        type=int,
    )
    parser.add_argument(
        "--decoder_output_image_size",
        default=decoder_defaults.output_image_size,
        type=int,
    )
    parser.add_argument(
        "--decoder_noise_tau",
        default=decoder_defaults.noise_tau,
        type=float,
    )

    # training
    parser.add_argument("--epochs", default=training_defaults.epochs, type=int)
    parser.add_argument(
        "--batch_size",
        default=None,
        type=int,
        help="Per-GPU batch size. If omitted, derived from global_batch_size / world_size.",
    )
    parser.add_argument(
        "--global_batch_size",
        default=training_defaults.global_batch_size,
        type=int,
        help="Used only when --batch_size is omitted.",
    )
    parser.add_argument("--lr", default=optimizer_defaults.lr, type=float)
    parser.add_argument(
        "--optimizer_betas",
        nargs=2,
        type=float,
        default=list(optimizer_defaults.betas),
        metavar=("BETA1", "BETA2"),
    )
    parser.add_argument("--weight_decay", default=optimizer_defaults.weight_decay, type=float)
    parser.add_argument("--ema_decay", default=training_defaults.ema_decay, type=float)
    parser.add_argument("--min_lr", default=scheduler_defaults.final_lr, type=float)
    parser.add_argument("--warmup_epochs", default=scheduler_defaults.warmup_epochs, type=int)
    parser.add_argument("--lr_schedule", default=scheduler_defaults.type, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--save_last_freq", type=int, default=1)
    parser.add_argument("--log_freq", default=20, type=int)
    parser.add_argument("--eval_freq", default=1, type=int)
    parser.add_argument("--num_images", default=2048, type=int)
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--online_eval", action="store_true")
    parser.add_argument("--no_online_eval", action="store_false", dest="online_eval")
    parser.set_defaults(online_eval=True)

    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--ram_shard_prefetch", action="store_true")
    parser.add_argument("--no_ram_shard_prefetch", action="store_false", dest="ram_shard_prefetch")
    parser.set_defaults(ram_shard_prefetch=True)
    parser.add_argument("--decoder_batch_prefetch", action="store_true")
    parser.add_argument("--no_decoder_batch_prefetch", action="store_false", dest="decoder_batch_prefetch")
    parser.set_defaults(decoder_batch_prefetch=True)

    # data
    parser.add_argument("--data_path", required=True, type=str, help="Feature shard root directory.")
    parser.add_argument(
        "--image_data_path",
        required=True,
        type=str,
        help="HF dataset ID or datasets.load_from_disk directory for raw training images.",
    )
    parser.add_argument(
        "--latent_dir_name",
        default="imagenet256_latents",
        type=str,
    )
    parser.add_argument(
        "--dino_dir_name",
        default="imagenet256_dinov3_features",
        type=str,
    )
    parser.add_argument(
        "--image_model_name",
        default="vit_base_patch16_dinov3.lvd1689m",
        type=str,
    )

    # decoder GAN / perceptual options
    parser.add_argument("--decoder_use_gan", action="store_true")
    parser.add_argument("--no_decoder_use_gan", action="store_false", dest="decoder_use_gan")
    parser.set_defaults(decoder_use_gan=True)
    parser.add_argument("--decoder_disc_loss", default=loss_defaults.disc_loss, type=str)
    parser.add_argument("--decoder_gen_loss", default=loss_defaults.gen_loss, type=str)
    parser.add_argument("--decoder_disc_weight", default=loss_defaults.disc_weight, type=float)
    parser.add_argument(
        "--decoder_perceptual_weight",
        default=loss_defaults.perceptual_weight,
        type=float,
    )
    parser.add_argument("--decoder_adaptive_weight", action="store_true")
    parser.add_argument("--no_decoder_adaptive_weight", action="store_false", dest="decoder_adaptive_weight")
    parser.set_defaults(decoder_adaptive_weight=loss_defaults.adaptive_weight)
    parser.add_argument("--decoder_disc_start", default=loss_defaults.disc_start, type=int)
    parser.add_argument("--decoder_disc_upd_start", default=loss_defaults.disc_upd_start, type=int)
    parser.add_argument(
        "--decoder_adversarial_warmup_epochs",
        default=loss_defaults.adversarial_warmup_epochs,
        type=float,
    )
    parser.add_argument("--decoder_lpips_start", default=loss_defaults.lpips_start, type=int)
    parser.add_argument("--decoder_max_d_weight", default=loss_defaults.max_d_weight, type=float)
    parser.add_argument("--decoder_disc_updates", default=loss_defaults.disc_updates, type=int)
    parser.add_argument("--decoder_lpips_net", default="vgg", type=str)

    parser.add_argument(
        "--decoder_disc_backbone_model_name",
        default=disc_defaults.arch.backbone_model_name,
        type=str,
    )
    parser.add_argument(
        "--decoder_disc_ckpt_path",
        default=disc_defaults.arch.dino_ckpt_path,
        type=str,
    )
    parser.add_argument("--decoder_disc_input_size", default=disc_defaults.arch.input_size, type=int)
    parser.add_argument("--decoder_disc_feature_dim", default=disc_defaults.arch.feature_dim, type=int)
    parser.add_argument("--decoder_disc_kernel_size", default=disc_defaults.arch.ks, type=int)
    parser.add_argument("--decoder_disc_norm_type", default=disc_defaults.arch.norm_type, type=str)
    parser.add_argument("--decoder_disc_pretrained", action="store_true")
    parser.add_argument("--no_decoder_disc_pretrained", action="store_false", dest="decoder_disc_pretrained")
    parser.set_defaults(decoder_disc_pretrained=True)
    parser.add_argument("--decoder_disc_using_spec_norm", action="store_true")
    parser.add_argument(
        "--no_decoder_disc_using_spec_norm",
        action="store_false",
        dest="decoder_disc_using_spec_norm",
    )
    parser.set_defaults(decoder_disc_using_spec_norm=disc_defaults.arch.using_spec_norm)
    parser.add_argument("--decoder_disc_freeze_backbone", action="store_true")
    parser.add_argument(
        "--no_decoder_disc_freeze_backbone",
        action="store_false",
        dest="decoder_disc_freeze_backbone",
    )
    parser.set_defaults(decoder_disc_freeze_backbone=disc_defaults.arch.freeze_backbone)
    parser.add_argument("--decoder_disc_lr", default=disc_defaults.optimizer.lr, type=float)
    parser.add_argument(
        "--decoder_disc_betas",
        nargs=2,
        type=float,
        default=list(disc_defaults.optimizer.betas),
        metavar=("BETA1", "BETA2"),
    )
    parser.add_argument(
        "--decoder_disc_weight_decay",
        default=disc_defaults.optimizer.weight_decay,
        type=float,
    )
    parser.add_argument("--decoder_disc_min_lr", default=None, type=float)
    parser.add_argument("--decoder_disc_warmup_epochs", default=None, type=int)
    parser.add_argument("--decoder_disc_epochs", default=None, type=int)
    parser.add_argument("--decoder_disc_lr_schedule", default=None, type=str)

    # logging / eval
    parser.add_argument("--output_dir", default="./output_dir_decoder", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--decoder_eval_reference_dir", default=None, type=str)
    parser.add_argument(
        "--decoder_eval_fid_stats",
        default=None,
        type=str,
        help="Path to a torch-fidelity FID statistics .npz file used for decoder eval.",
    )
    parser.add_argument("--decoder_eval_metrics", action="store_true")
    parser.add_argument("--no_decoder_eval_metrics", action="store_false", dest="decoder_eval_metrics")
    parser.set_defaults(decoder_eval_metrics=True)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--no_use_wandb", action="store_false", dest="use_wandb")
    parser.set_defaults(use_wandb=True)
    parser.add_argument("--wandb_project", type=str, default="jit-decoder")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--wandb_eval_image_interval", type=int, default=10)
    parser.add_argument("--device", default="cuda")

    # distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--dist_timeout_sec", default=7200, type=int)
    parser.add_argument("--ddp_bucket_cap_mb", default=100, type=int)
    parser.add_argument("--ddp_broadcast_buffers", action="store_true")
    parser.add_argument("--no_ddp_broadcast_buffers", action="store_false", dest="ddp_broadcast_buffers")
    parser.set_defaults(ddp_broadcast_buffers=False)
    parser.add_argument("--ddp_gradient_as_bucket_view", action="store_true")
    parser.add_argument(
        "--no_ddp_gradient_as_bucket_view",
        action="store_false",
        dest="ddp_gradient_as_bucket_view",
    )
    parser.set_defaults(ddp_gradient_as_bucket_view=True)
    parser.add_argument("--ddp_static_graph", action="store_true")
    parser.add_argument("--no_ddp_static_graph", action="store_false", dest="ddp_static_graph")
    parser.set_defaults(ddp_static_graph=True)
    return parser


def parse_args() -> argparse.Namespace:
    parser = get_args_parser()
    args = parser.parse_args()
    args.decoder_plan = load_decoder_plan_config(args.config)
    args.optimizer_betas = tuple(float(beta) for beta in args.optimizer_betas)
    args.decoder_disc_betas = tuple(float(beta) for beta in args.decoder_disc_betas)

    if args.decoder_disc_min_lr is None:
        args.decoder_disc_min_lr = args.min_lr
    if args.decoder_disc_warmup_epochs is None:
        args.decoder_disc_warmup_epochs = args.warmup_epochs
    if args.decoder_disc_epochs is None:
        args.decoder_disc_epochs = args.epochs
    if args.decoder_disc_lr_schedule is None:
        args.decoder_disc_lr_schedule = args.lr_schedule
    if not args.resume:
        args.resume = args.output_dir
    return args


def build_decoder_model(args: argparse.Namespace) -> DecoderReconstructionModel:
    decoder = Decoder(
        input_size=args.latent_size,
        patch_size=args.decoder_patch_size,
        latent_patch_size=args.decoder_latent_patch_size,
        in_channels=args.latent_in_channels,
        bottleneck_dim=args.bottleneck_dim,
        dino_hidden_size=args.dino_hidden_size,
        hidden_size=args.decoder_hidden_size,
        out_channels=args.image_out_channels,
        depth=args.decoder_depth,
        attn_drop=args.attn_dropout,
        proj_drop=args.proj_dropout,
        num_heads=args.decoder_num_heads,
        mlp_ratio=args.decoder_mlp_ratio,
        output_image_size=args.decoder_output_image_size,
    )
    return DecoderReconstructionModel(decoder)


def save_decoder_checkpoint(
    args: argparse.Namespace,
    model_without_ddp: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    epoch_name: str = "last",
    ema_model: torch.nn.Module | None = None,
) -> None:
    if not args.output_dir:
        return

    checkpoint_path = Path(args.output_dir) / f"checkpoint-{epoch_name}.pth"
    payload = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": {key: value for key, value in vars(args).items() if key != "decoder_plan"},
    }
    if ema_model is not None:
        payload["model_ema"] = ema_model.state_dict()
    misc.save_on_master(payload, checkpoint_path)


def maybe_resume_checkpoint(
    args: argparse.Namespace,
    model_without_ddp: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_model: torch.nn.Module | None = None,
) -> None:
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint["model"])
    if ema_model is not None:
        ema_state = checkpoint.get("model_ema", checkpoint["model"])
        ema_model.load_state_dict(ema_state)

    if "optimizer" in checkpoint and "epoch" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = int(checkpoint["epoch"]) + 1
    print("Resumed decoder checkpoint from", checkpoint_path)


def describe_dataset_plan(dataset: RamLoadedShardDataset, latent_store, dino_store, *, prefetch: bool) -> None:
    plan = dataset.describe_current_plan()
    max_shard_samples = max(span.size for span in dataset.logical_shards)
    approx_max_ram_bytes = max_shard_samples * (
        latent_store.bytes_per_sample + dino_store.bytes_per_sample
    )
    approx_peak_ram_bytes = approx_max_ram_bytes * 2 if prefetch else approx_max_ram_bytes
    print(
        "Decoder RAM shard loading enabled using "
        f"{plan['logical_shard_count']} logical shards from {plan['logical_shard_source']}."
    )
    print(
        "Approx max per-rank shard working set: "
        f"{approx_max_ram_bytes / (1024 ** 3):.2f} GiB."
    )
    if prefetch:
        print(
            "RAM shard prefetch enabled: peak per-rank working set can temporarily reach about "
            f"{approx_peak_ram_bytes / (1024 ** 3):.2f} GiB while the next shard is staged."
        )
    if dataset.preload_next_batch:
        print(
            "Decoder batch prefetch enabled: one formatted batch is prepared on CPU while the current step runs."
        )
    print(
        "Epoch 0 steps per rank: "
        f"{plan['num_batches']} "
        f"(samples/rank={plan['num_samples_per_rank']}, "
        f"dropped_tail_per_rank={plan['dropped_samples_per_rank']})."
    )


def build_data_loader(
    *,
    latent_store,
    dino_store,
    batch_size: int,
    num_tasks: int,
    global_rank: int,
    shuffle_shards: bool,
    seed: int,
    preload_next_shard: bool,
    preload_next_batch: bool,
    image_data_path: str,
    image_model_name: str,
    image_size: int,
    pin_mem: bool,
):
    dataset = RamLoadedShardDataset(
        latent_store=latent_store,
        dino_store=dino_store,
        batch_size=batch_size,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle_shards=shuffle_shards,
        seed=seed,
        preload_next_shard=preload_next_shard,
        preload_next_batch=preload_next_batch,
        image_data_path=image_data_path,
        image_model_name=image_model_name,
        image_size=image_size,
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=pin_mem,
    )
    return dataset, loader


def main(args: argparse.Namespace) -> None:
    misc.init_distributed_mode(args)
    print("Job directory:", os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if args.batch_size is None:
        if args.global_batch_size % num_tasks != 0:
            raise ValueError(
                f"global_batch_size={args.global_batch_size} must be divisible by world_size={num_tasks}."
            )
        args.batch_size = args.global_batch_size // num_tasks
    args.global_batch_size = args.batch_size * num_tasks

    log_writer = None
    if global_rank == 0 and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)

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
            config={key: value for key, value in vars(args).items() if key != "decoder_plan"},
            dir=args.output_dir if args.output_dir else None,
            mode=args.wandb_mode,
        )

    latent_store = inspect_feature_shards(args.data_path, args.latent_dir_name)
    dino_store = inspect_feature_shards(args.data_path, args.dino_dir_name)

    dataset_train, data_loader_train = build_data_loader(
        latent_store=latent_store,
        dino_store=dino_store,
        batch_size=args.batch_size,
        num_tasks=num_tasks,
        global_rank=global_rank,
        shuffle_shards=True,
        seed=args.seed,
        preload_next_shard=args.ram_shard_prefetch,
        preload_next_batch=args.decoder_batch_prefetch,
        image_data_path=args.image_data_path,
        image_model_name=args.image_model_name,
        image_size=args.decoder_output_image_size,
        pin_mem=args.pin_mem,
    )
    dataset_eval, data_loader_eval = build_data_loader(
        latent_store=latent_store,
        dino_store=dino_store,
        batch_size=args.batch_size,
        num_tasks=num_tasks,
        global_rank=global_rank,
        shuffle_shards=False,
        seed=args.seed,
        preload_next_shard=False,
        preload_next_batch=args.decoder_batch_prefetch,
        image_data_path=args.image_data_path,
        image_model_name=args.image_model_name,
        image_size=args.decoder_output_image_size,
        pin_mem=args.pin_mem,
    )

    if global_rank == 0:
        describe_dataset_plan(
            dataset_train,
            latent_store,
            dino_store,
            prefetch=args.ram_shard_prefetch,
        )

    model = build_decoder_model(args)
    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))
    print("Per-GPU batch size:", args.batch_size)
    print("Effective global batch size:", args.global_batch_size)

    model.to(device)
    if args.distributed:
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
    model_without_ddp = model.module if hasattr(model, "module") else model
    ema_model = None
    if args.ema_decay > 0.0:
        ema_model = build_decoder_ema_model(model_without_ddp, device)
        print(f"Decoder EMA enabled with decay={args.ema_decay:.6f}")

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.Adam(
        param_groups,
        lr=args.lr,
        betas=args.optimizer_betas,
    )
    print(optimizer)

    maybe_resume_checkpoint(args, model_without_ddp, optimizer, ema_model)

    try:
        if args.evaluate_only:
            print("Evaluating decoder checkpoint at epoch", args.start_epoch)
            with torch.no_grad():
                evaluate(
                    ema_model if ema_model is not None else model_without_ddp,
                    args,
                    args.start_epoch,
                    data_loader=data_loader_eval,
                    log_writer=log_writer,
                    wandb_run=wandb_run,
                )
            return

        print(f"Start decoder training for {args.epochs} epochs")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            dataset_train.set_epoch(epoch)
            steps_per_epoch = len(data_loader_train)
            if steps_per_epoch <= 0:
                raise RuntimeError("Decoder dataloader has zero steps for this epoch.")

            train_epoch(
                model=model,
                optimizer=optimizer,
                log_writer=log_writer,
                epoch=epoch,
                args=args,
                steps_per_epoch=steps_per_epoch,
                wandb_run=wandb_run,
                data_loader=data_loader_train,
                device=device,
                post_step_callback=(
                    lambda: update_ema_model(ema_model, model_without_ddp, args.ema_decay)
                    if ema_model is not None
                    else None
                ),
            )

            did_save_checkpoint = False
            if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
                save_decoder_checkpoint(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    epoch=epoch,
                    epoch_name="last",
                    ema_model=ema_model,
                )
                did_save_checkpoint = True

            if epoch % 10 == 0 and epoch > 0:
                save_decoder_checkpoint(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    epoch=epoch,
                    epoch_name=str(epoch),
                    ema_model=ema_model,
                )
                did_save_checkpoint = True

            if did_save_checkpoint and args.distributed:
                torch.distributed.barrier()

            if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    evaluate(
                        ema_model if ema_model is not None else model_without_ddp,
                        args,
                        epoch,
                        data_loader=data_loader_eval,
                        log_writer=log_writer,
                        wandb_run=wandb_run,
                        wandb_step=(epoch + 1) * steps_per_epoch,
                    )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if misc.is_main_process() and log_writer is not None:
                log_writer.flush()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Decoder training time:", total_time_str)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/total_time_sec": total_time,
                    "train/total_time_hms": total_time_str,
                }
            )
    finally:
        if log_writer is not None:
            log_writer.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main(parse_args())
