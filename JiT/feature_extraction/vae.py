# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
Reference - https://github.com/chuanyangjin/fast-DiT
"""


from diffusers.models import AutoencoderKL
from datasets import Array3D, Dataset, Features, Value, load_dataset
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm
import os
import logging
import argparse
from time import time
from copy import deepcopy
import shutil
from PIL import Image
from collections import OrderedDict
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch
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


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################
def collate_fn(batch):
    # batch is list of dicts like {"image": tensor, "label": int, ...}
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([b.get("label", -1)
                          for b in batch], dtype=torch.long)
    return {"image": images, "label": labels}


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
        tmp_shard_dir = os.path.join(args.features_path, "_tmp_hf_latent_shards")
        if os.path.exists(tmp_shard_dir):
            shutil.rmtree(tmp_shard_dir)
        os.makedirs(tmp_shard_dir, exist_ok=True)
    dist.barrier()

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sdxl-vae").to(device)

    # Setup data:
    local_batch_size = args.global_batch_size // dist.get_world_size()
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(
            pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[
                             0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = load_dataset(args.data_path, split="train")
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

    tmp_shard_dir = os.path.join(args.features_path, "_tmp_hf_latent_shards")
    rank_shard_path = os.path.join(tmp_shard_dir, f"rank_{rank:05d}.arrow")
    hf_features = Features(
        {
            "feature": Array3D(shape=(4, latent_size, latent_size), dtype="float16"),
            "label": Value("int64"),
            # used to restore deterministic global ordering after rank-wise writes
            "sample_id": Value("int64"),
        }
    )
    shard_writer = ArrowWriter(path=rank_shard_path, features=hf_features)

    train_steps = 0
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
            # save_num = NUM_SAMPLES * rank + train_steps * local_batch_size + i
            sample_id = train_steps * args.global_batch_size + dist.get_world_size() * \
                i + rank
            shard_writer.write(
                {
                    "feature": x[i].astype(np.float16, copy=False),
                    "label": int(y[i]),
                    "sample_id": int(sample_id),
                }
            )

        train_steps += 1

    shard_writer.finalize()

    # each rank converts its own shard to parquet (parallelised across ranks)
    output_dir = os.path.join(args.features_path, args.hf_dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    shard_ds = Dataset.from_file(rank_shard_path)
    shard_ds.to_parquet(
        os.path.join(output_dir, f"shard_{rank:05d}.parquet"),
        compression="zstd",
    )
    del shard_ds
    os.remove(rank_shard_path)

    dist.barrier()
    if rank == 0:
        os.rmdir(tmp_shard_dir)
        print(f"Saved HF dataset to: {output_dir}")

    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, default="features")
    parser.add_argument("--image-size", type=int,
                        choices=[256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hf-dataset-name", type=str, default="imagenet256_latents")
    args = parser.parse_args()
    main(args)
