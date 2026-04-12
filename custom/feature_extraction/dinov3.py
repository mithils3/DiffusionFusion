import os
import argparse
import shutil
import uuid

import numpy as np
import timm
import torch
import torch.distributed as dist
from datasets import Array3D, Dataset, Features, Value, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from custom.util.feature_normalization import normalize_dino_feature_map_tokens
from custom.util.image_transforms import build_center_crop_normalize_transform


def collate_fn(batch):
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


def normalize_model_name(model_name: str) -> str:
    if model_name.startswith("timm/"):
        return model_name.split("/", 1)[1]
    return model_name


def resolve_output_dataset_name(explicit_name, image_size, split):
    if explicit_name:
        return explicit_name
    base_name = f"imagenet{image_size}_dinov3_features"
    if split == "train":
        return base_name
    return f"{base_name}_{split}"


def main(args):
    """
    Extract DINO features and normalize each spatial token independently across
    channels, matching the RAE latent preprocessing recipe.
    """
    assert torch.cuda.is_available(), "Requires at least one GPU."
    assert args.image_size % args.patch_size == 0, (
        f"Image size {args.image_size} must be divisible by patch size {args.patch_size}."
    )

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
    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
    dist.barrier()

    model_name = normalize_model_name(args.model_name)
    model = timm.create_model(
        model_name,
        pretrained=True,
        features_only=True,
        img_size=args.image_size,
    ).to(device)
    model = model.eval()

    local_batch_size = args.global_batch_size // dist.get_world_size()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = build_center_crop_normalize_transform(
        args.image_size,
        mean=data_config.get("mean"),
        std=data_config.get("std"),
    )
    dataset = load_dataset(args.data_path, split=args.split)
    dataset = dataset.with_transform(
        lambda examples: {
            "image": [transforms(image.convert("RGB")) for image in examples["image"]],
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

    patches = args.image_size // args.patch_size

    output_dataset_name = resolve_output_dataset_name(
        args.hf_dataset_name,
        args.image_size,
        args.split,
    )
    output_dir = os.path.join(args.features_path, output_dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    hf_features = Features(
        {
            "feature": Array3D(shape=(args.hidden_size, patches, patches), dtype="float16"),
            "label": Value("int64"),
            "sample_id": Value("int64"),
        }
    )
    samples_per_shard = compute_samples_per_shard(
        (args.hidden_size, patches, patches), args.max_shard_size_mb)
    print(f"Rank {rank}: writing approximately {samples_per_shard} samples per shard.")

    feature_buf = []
    label_buf = []
    sample_id_buf = []
    shard_idx = 0
    local_sample_idx = 0
    rank_sample_offset = rank * len(sampler)

    for batch in tqdm(loader, total=len(loader), desc=f"Rank {rank} [write]"):
        x = batch["image"].to(device)
        y = batch["label"].cpu().numpy()

        with torch.no_grad():
            output = model(x)[-1]  # last feature map (B, C, H, W)
            output = normalize_dino_feature_map_tokens(output)

        output = output.detach().cpu().numpy().astype(np.float16, copy=False)

        for i in range(output.shape[0]):
            sample_id = rank_sample_offset + local_sample_idx
            feature_buf.append(output[i])
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
        print(f"Saved per-token normalized {args.split} HF dataset to: {output_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, default="features")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help='Dataset split to encode. Defaults to "train".',
    )
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--hf-dataset-name",
        type=str,
        default=None,
        help=(
            "Output HF dataset directory name. Defaults to "
            "`imagenet{image_size}_dinov3_features` for train and "
            "`imagenet{image_size}_dinov3_features_{split}` for non-train splits."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="timm/vit_base_patch16_dinov3.lvd1689m",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--max-shard-size-mb", type=int, default=4096)
    args = parser.parse_args()
    main(args)
