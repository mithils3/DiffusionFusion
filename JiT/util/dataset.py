import math
from typing import List

from torch.utils.data import Dataset, Sampler
import torch
import torch.distributed as dist
import os
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, latent_dataset, dino_dataset):
        self.latent_dataset = latent_dataset
        self.dino_dataset = dino_dataset

    def __len__(self):
        return len(self.latent_dataset)

    def __getitem__(self, idx):
        latent = torch.as_tensor(self.latent_dataset[idx]["feature"])
        dino = torch.as_tensor(self.dino_dataset[idx]["feature"])
        label = torch.tensor(self.latent_dataset[idx]["label"]).long()
        return {"latent": latent, "dino": dino, "y": label}


class ShardAwareSampler(Sampler):
    """Distributed sampler that shuffles shard order but reads sequentially
    within each shard. This produces a much more sequential disk access
    pattern than a fully-random permutation, which is critical when
    the dataset lives on HDD.

    Each epoch:
      1. Shuffle the order of shards (all ranks see the same order).
      2. Optionally shuffle indices within each shard (default: off for HDD).
      3. Distribute indices across ranks via interleaving.
    """

    def __init__(
        self,
        shard_sizes: List[int],
        num_replicas: int = -1,
        rank: int = -1,
        shuffle_shards: bool = True,
        shuffle_within_shards: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        if num_replicas < 0:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank < 0:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.shard_sizes = shard_sizes
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shards = shuffle_within_shards
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Precompute shard start offsets in the concatenated dataset.
        self.shard_offsets = []
        offset = 0
        for size in shard_sizes:
            self.shard_offsets.append(offset)
            offset += size
        self.total_size = offset

        # Per-replica sample count (mirrors DistributedSampler logic).
        if self.drop_last:
            self.num_samples = self.total_size // self.num_replicas
        else:
            self.num_samples = math.ceil(self.total_size / self.num_replicas)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        num_shards = len(self.shard_sizes)
        if self.shuffle_shards:
            shard_order = torch.randperm(num_shards, generator=g).tolist()
        else:
            shard_order = list(range(num_shards))

        indices = []
        for shard_idx in shard_order:
            offset = self.shard_offsets[shard_idx]
            size = self.shard_sizes[shard_idx]
            if self.shuffle_within_shards:
                within = torch.randperm(size, generator=g).tolist()
                indices.extend(offset + w for w in within)
            else:
                indices.extend(range(offset, offset + size))

        # Distribute across ranks via interleaving (same as DistributedSampler).
        indices = indices[self.rank :: self.num_replicas]

        if self.drop_last:
            indices = indices[: self.num_samples]

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch
