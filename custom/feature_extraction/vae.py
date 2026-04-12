


from diffusers.models import AutoencoderKL
from datasets import Array3D, Dataset, Features, Value, load_dataset
from tqdm import tqdm
import os
import logging
import argparse
import shutil
from time import time
from copy import deepcopy
from PIL import Image
from collections import OrderedDict
import numpy as np
import uuid
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch

from custom.util.image_transforms import build_center_crop_normalize_transform
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(
                f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################
def collate_fn(batch):
    # batch is list of dicts like {"image": tensor, "label": int, ...}
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([b.get("label", -1)
                          for b in batch], dtype=torch.long)
    return {"image": images, "label": labels}


def compute_samples_per_shard(shape, max_shard_size_mb):
    bytes_per_sample = int(np.prod(shape)) * np.dtype(np.float16).itemsize
    shard_bytes = max_shard_size_mb * 1024 * 1024
    return max(1, shard_bytes // max(bytes_per_sample, 1))


def save_feature_shard(output_dir, shard_name, features, labels, sample_ids, hf_features):
    shard_dir = os.path.join(output_dir, shard_name)
    tmp_shard_dir = f"{shard_dir}.tmp-{uuid.uuid4().hex}"
    shard_ds = Dataset.from_dict(
        {
            "feature": features,
            "label": labels,
            "sample_id": sample_ids,
        },
        features=hf_features,
    )
    try:
        shard_ds.save_to_disk(tmp_shard_dir)
        if os.path.exists(shard_dir):
            shutil.rmtree(shard_dir)
        os.rename(tmp_shard_dir, shard_dir)
    finally:
        del shard_ds
        if os.path.exists(tmp_shard_dir):
            shutil.rmtree(tmp_shard_dir, ignore_errors=True)


def resolve_output_dataset_name(explicit_name, image_size, split):
    if explicit_name:
        return explicit_name
    base_name = f"imagenet{image_size}_latents"
    if split == "train":
        return base_name
    return f"{base_name}_{split}"


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size(
    ) == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(
        f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
    dist.barrier()

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sdxl-vae").to(device)

    # Setup data:
    local_batch_size = args.global_batch_size // dist.get_world_size()
    transform = build_center_crop_normalize_transform(
        args.image_size,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    dataset = load_dataset(args.data_path, split=args.split)
    dataset = dataset.with_format("torch")
    dataset = dataset.with_transform(
        lambda examples: {
            "image": [transform(image.convert("RGB")) for image in examples["image"]],
            "label": examples["label"],
        }
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=4,
    )

    hf_features = Features(
        {
            "feature": Array3D(shape=(4, latent_size, latent_size), dtype="float16"),
            "label": Value("int64"),
            # used to restore deterministic global ordering after rank-wise writes
            "sample_id": Value("int64"),
        }
    )

    samples_per_shard = compute_samples_per_shard(
        (4, latent_size, latent_size), args.max_shard_size_mb)
    print(f"Rank {rank}: writing approximately {samples_per_shard} samples per shard.")

    output_dataset_name = resolve_output_dataset_name(
        args.hf_dataset_name,
        args.image_size,
        args.split,
    )
    output_dir = os.path.join(args.features_path, output_dataset_name)

    feature_buf = []
    label_buf = []
    sample_id_buf = []
    shard_idx = 0
    local_sample_idx = 0
    rank_sample_offset = rank * len(sampler)

    for batch in tqdm(loader, total=len(loader), desc=f"Rank {rank}"):
        x = batch["image"]
        y = batch["label"]

        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)

        x = x.detach().cpu().numpy()    # (bs, 4, 32, 32)
        y = y.detach().cpu().numpy()    # (bs,)
        for i in range(x.shape[0]):
            sample_id = rank_sample_offset + local_sample_idx
            feature_buf.append(x[i].astype(np.float16, copy=False))
            label_buf.append(int(y[i]))
            sample_id_buf.append(int(sample_id))
            local_sample_idx += 1

            if len(feature_buf) >= samples_per_shard:
                shard_name = f"shard_{rank:05d}_{shard_idx:05d}"
                save_feature_shard(
                    output_dir=output_dir,
                    shard_name=shard_name,
                    features=feature_buf,
                    labels=label_buf,
                    sample_ids=sample_id_buf,
                    hf_features=hf_features,
                )
                feature_buf = []
                label_buf = []
                sample_id_buf = []
                shard_idx += 1

    os.makedirs(output_dir, exist_ok=True)
    if feature_buf:
        shard_name = f"shard_{rank:05d}_{shard_idx:05d}"
        save_feature_shard(
            output_dir=output_dir,
            shard_name=shard_name,
            features=feature_buf,
            labels=label_buf,
            sample_ids=sample_id_buf,
            hf_features=hf_features,
        )
    del feature_buf, label_buf, sample_id_buf

    dist.barrier()
    if rank == 0:
        print(f"Saved {args.split} HF dataset to: {output_dir}")

    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, default="features")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help='Dataset split to encode. Defaults to "train".',
    )
    parser.add_argument("--image-size", type=int,
                        choices=[256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--hf-dataset-name",
        type=str,
        default=None,
        help=(
            "Output HF dataset directory name. Defaults to "
            "`imagenet{image_size}_latents` for train and "
            "`imagenet{image_size}_latents_{split}` for non-train splits."
        ),
    )
    parser.add_argument("--max-shard-size-mb", type=int, default=1024)
    args = parser.parse_args()
    main(args)
