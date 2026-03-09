import torch
import torch.distributed as dist
import os
import json
import timm
from datasets import load_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from datasets import Array3D, Dataset, Features, Value, concatenate_datasets
from tqdm import tqdm
import argparse
import numpy as np


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
    os.makedirs(shard_dir, exist_ok=True)
    shard_ds = Dataset.from_dict(
        {
            "feature": features,
            "label": labels,
            "sample_id": sample_ids,
        },
        features=hf_features,
    )
    shard_ds.save_to_disk(shard_dir)
    del shard_ds


def main(args):
    """
    Extracts DINO features and normalizes them to zero mean, unit variance
    (channel-wise) to match the scale of VAE latents.
    """
    assert torch.cuda.is_available(), "Requires at least one GPU."

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

    model = timm.create_model(
        args.model_name,
        pretrained=True,
        features_only=True,
    ).to(device)
    model = model.eval()

    local_batch_size = args.global_batch_size // dist.get_world_size()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    dataset = load_dataset(args.data_path, split="train")
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

    patches = args.image_size // 16

    output_dir = os.path.join(args.features_path, args.hf_dataset_name)
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

    # Pass 1: compute global channel-wise stats without retaining features in RAM.
    channel_sum = np.zeros(args.hidden_size, dtype=np.float64)
    channel_sum_sq = np.zeros(args.hidden_size, dtype=np.float64)
    pixel_count = 0

    for batch in tqdm(loader, total=len(loader), desc=f"Rank {rank} [stats]"):
        x = batch["image"].to(device)

        with torch.no_grad():
            output = model(x)[-1]  # last feature map (B, C, H, W)

        output = output.detach().cpu().float().numpy()

        # Running channel-wise sums: reduce over (B, H, W), keep C
        channel_sum += output.sum(axis=(0, 2, 3))
        channel_sum_sq += (output ** 2).sum(axis=(0, 2, 3))
        pixel_count += output.shape[0] * output.shape[2] * output.shape[3]

    # All-reduce channel stats across ranks to get global mean/std
    sum_t = torch.tensor(channel_sum, dtype=torch.float64, device=device)
    sum_sq_t = torch.tensor(channel_sum_sq, dtype=torch.float64, device=device)
    count_t = torch.tensor([pixel_count], dtype=torch.float64, device=device)
    dist.all_reduce(sum_t)
    dist.all_reduce(sum_sq_t)
    dist.all_reduce(count_t)

    global_mean = (sum_t / count_t).cpu().numpy()            # (C,)
    global_std = np.sqrt(
        (sum_sq_t / count_t).cpu().numpy() - global_mean ** 2
    )                                                         # (C,)
    mean = global_mean.astype(np.float32).reshape(-1, 1, 1)  # (C, 1, 1)
    std = global_std.astype(np.float32).reshape(-1, 1, 1)    # (C, 1, 1)

    # Save normalization stats (needed to un-normalize generated DINO features)
    if rank == 0:
        stats_path = os.path.join(output_dir, "normalization_stats.json")
        with open(stats_path, "w") as f:
            json.dump({"mean": global_mean.tolist(),
                       "std": global_std.tolist()}, f)
        print(f"Saved normalization stats to: {stats_path}")

    # Pass 2: recompute features, normalize on the fly, and write small HF shards.
    dist.barrier()
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

        output = output.detach().cpu().float().numpy()
        output = ((output - mean) / std).astype(np.float16)

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
        print(f"Saved normalized HF dataset to: {output_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, default="features")
    parser.add_argument("--image-size", type=int,
                        choices=[256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hf-dataset-name", type=str,
                        default="imagenet256_dinov3_features")
    parser.add_argument("--model-name", type=str,
                        default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--max-shard-size-mb", type=int, default=4096)
    args = parser.parse_args()
    main(args)
