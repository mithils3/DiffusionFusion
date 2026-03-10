import gc
import math
import os
from bisect import bisect_right
from dataclasses import dataclass
from glob import glob
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import Dataset, IterableDataset, Sampler


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


def resolve_feature_dataset_root(data_path: str, dataset_dir_name: str) -> str:
    if os.path.isdir(dataset_dir_name):
        return dataset_dir_name
    return os.path.join(data_path, dataset_dir_name)


def load_feature_dataset(data_path, dataset_dir_name, dtype=np.float16):
    dataset_root = resolve_feature_dataset_root(data_path, dataset_dir_name)
    shard_dirs = sorted(glob(os.path.join(dataset_root, "shard_*")))
    if not shard_dirs:
        raise FileNotFoundError(
            f"No shard directories found under {dataset_root}"
        )

    shard_datasets = [load_from_disk(shard_dir) for shard_dir in shard_dirs]
    sample_id_is_monotonic = True
    prev_last_sample_id = None
    for shard_dir, shard_dataset in zip(shard_dirs, shard_datasets):
        if len(shard_dataset) == 0:
            continue
        first_sample_id = int(shard_dataset[0]["sample_id"])
        last_sample_id = int(shard_dataset[len(shard_dataset) - 1]["sample_id"])
        if first_sample_id > last_sample_id:
            sample_id_is_monotonic = False
            print(
                f"Shard {shard_dir} has descending sample_id order; will sort after concatenation."
            )
            break
        if prev_last_sample_id is not None and first_sample_id <= prev_last_sample_id:
            sample_id_is_monotonic = False
            print(
                f"Shard {shard_dir} overlaps or regresses in sample_id order; will sort after concatenation."
            )
            break
        prev_last_sample_id = last_sample_id

    shard_sizes = [len(ds) for ds in shard_datasets]
    dataset = concatenate_datasets(shard_datasets)
    if sample_id_is_monotonic:
        print(f"{dataset_dir_name}: sample_id order already monotonic across shards, skipping sort.")
    else:
        print(f"{dataset_dir_name}: applying sort('sample_id') to restore deterministic alignment.")
        dataset = dataset.sort("sample_id")
        # After sorting, shard boundaries no longer correspond to contiguous
        # regions, so fall back to treating the whole dataset as one shard.
        shard_sizes = [len(dataset)]
    dataset = dataset.with_format(
        "numpy", columns=["feature"], output_all_columns=True, dtype=dtype
    )
    return dataset, shard_sizes


@dataclass(frozen=True)
class DatasetShardSpan:
    path: str
    global_start: int
    global_end: int
    first_sample_id: int
    last_sample_id: int

    @property
    def size(self) -> int:
        return self.global_end - self.global_start


@dataclass(frozen=True)
class FeatureShardStore:
    name: str
    root: str
    shard_spans: List[DatasetShardSpan]
    total_size: int
    bytes_per_sample: int

    @property
    def shard_sizes(self) -> List[int]:
        return [span.size for span in self.shard_spans]


@dataclass(frozen=True)
class LogicalShardSpan:
    global_start: int
    global_end: int

    @property
    def size(self) -> int:
        return self.global_end - self.global_start


def inspect_feature_shards(data_path: str, dataset_dir_name: str) -> FeatureShardStore:
    dataset_root = resolve_feature_dataset_root(data_path, dataset_dir_name)
    shard_dirs = sorted(glob(os.path.join(dataset_root, "shard_*")))
    if not shard_dirs:
        raise FileNotFoundError(
            f"No shard directories found under {dataset_root}"
        )

    shard_spans = []
    total_size = 0
    prev_last_sample_id = None
    bytes_per_sample = None

    for shard_dir in shard_dirs:
        shard_dataset = load_from_disk(shard_dir)
        shard_size = len(shard_dataset)
        if shard_size == 0:
            continue

        first_sample_id = int(shard_dataset[0]["sample_id"])
        last_sample_id = int(shard_dataset[shard_size - 1]["sample_id"])
        if first_sample_id > last_sample_id:
            raise ValueError(
                f"{dataset_dir_name} shard {shard_dir} has descending sample_id order; "
                "RAM shard loading requires monotonic shard ordering."
            )
        if prev_last_sample_id is not None and first_sample_id <= prev_last_sample_id:
            raise ValueError(
                f"{dataset_dir_name} shard {shard_dir} overlaps or regresses in sample_id order; "
                "RAM shard loading requires monotonic shard ordering."
            )
        prev_last_sample_id = last_sample_id

        if bytes_per_sample is None:
            bytes_per_sample = int(np.asarray(shard_dataset[0]["feature"]).nbytes)

        shard_spans.append(
            DatasetShardSpan(
                path=shard_dir,
                global_start=total_size,
                global_end=total_size + shard_size,
                first_sample_id=first_sample_id,
                last_sample_id=last_sample_id,
            )
        )
        total_size += shard_size

    if not shard_spans or bytes_per_sample is None:
        raise ValueError(
            f"{dataset_dir_name} does not contain any non-empty shards under {dataset_root}."
        )

    return FeatureShardStore(
        name=dataset_dir_name,
        root=dataset_root,
        shard_spans=shard_spans,
        total_size=total_size,
        bytes_per_sample=bytes_per_sample,
    )


def _copy_rows_to_ram(rows: Dict[str, object], dtype: np.dtype) -> Dict[str, np.ndarray]:
    return {
        "feature": np.array(rows["feature"], dtype=dtype, copy=True),
        "label": np.array(rows["label"], dtype=np.int64, copy=True),
        "sample_id": np.array(rows["sample_id"], dtype=np.int64, copy=True),
    }


def _load_feature_range_to_ram(
    store: FeatureShardStore,
    start: int,
    end: int,
    dtype: np.dtype = np.float16,
) -> Dict[str, np.ndarray]:
    if start < 0 or end > store.total_size or start >= end:
        raise ValueError(
            f"Invalid range [{start}, {end}) for dataset {store.name} with total size {store.total_size}."
        )

    shard_ends = [span.global_end for span in store.shard_spans]
    shard_idx = bisect_right(shard_ends, start)

    features = []
    labels = []
    sample_ids = []
    while shard_idx < len(store.shard_spans):
        span = store.shard_spans[shard_idx]
        if span.global_start >= end:
            break

        local_start = max(start, span.global_start) - span.global_start
        local_end = min(end, span.global_end) - span.global_start
        shard_dataset = load_from_disk(span.path).with_format(
            "numpy", columns=["feature"], output_all_columns=True, dtype=dtype
        )
        rows = shard_dataset[:] if local_start == 0 and local_end == span.size else shard_dataset[local_start:local_end]
        copied = _copy_rows_to_ram(rows, dtype=dtype)
        features.append(copied["feature"])
        labels.append(copied["label"])
        sample_ids.append(copied["sample_id"])
        shard_idx += 1

    if not features:
        raise RuntimeError(
            f"Failed to materialize range [{start}, {end}) from dataset {store.name}."
        )

    if len(features) == 1:
        return {
            "feature": features[0],
            "label": labels[0],
            "sample_id": sample_ids[0],
        }

    return {
        "feature": np.concatenate(features, axis=0),
        "label": np.concatenate(labels, axis=0),
        "sample_id": np.concatenate(sample_ids, axis=0),
    }


class RamLoadedShardDataset(IterableDataset):
    """Iterable dataset that keeps one logical shard pair per rank in RAM.

    Logical shards are derived from the larger feature tensor family so that
    the in-memory working set stays bounded even when the latent and DINO
    shard sample counts differ.
    """

    def __init__(
        self,
        latent_store: FeatureShardStore,
        dino_store: FeatureShardStore,
        batch_size: int,
        num_replicas: int = -1,
        rank: int = -1,
        shuffle_shards: bool = True,
        seed: int = 0,
    ):
        if num_replicas < 0:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank < 0:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if batch_size <= 0:
            raise ValueError("batch_size must be positive for RAM shard loading.")

        if latent_store.total_size != dino_store.total_size:
            raise ValueError(
                f"Latent dataset has {latent_store.total_size} samples but DINO dataset has "
                f"{dino_store.total_size}; RAM shard loading requires aligned datasets."
            )

        self.latent_store = latent_store
        self.dino_store = dino_store
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.epoch = 0

        self.logical_shard_store = self._select_logical_shard_store()
        self.logical_shards = [
            LogicalShardSpan(
                global_start=span.global_start,
                global_end=span.global_end,
            )
            for span in self.logical_shard_store.shard_spans
        ]
        if not self.logical_shards:
            raise ValueError("RAM shard loading requires at least one logical shard.")

        self._cached_epoch = None
        self._cached_plan = None

    def _select_logical_shard_store(self) -> FeatureShardStore:
        if self.dino_store.bytes_per_sample > self.latent_store.bytes_per_sample:
            return self.dino_store
        if self.latent_store.bytes_per_sample > self.dino_store.bytes_per_sample:
            return self.latent_store
        if len(self.dino_store.shard_spans) >= len(self.latent_store.shard_spans):
            return self.dino_store
        return self.latent_store

    def _build_epoch_plan(self):
        if self._cached_epoch == self.epoch and self._cached_plan is not None:
            return self._cached_plan

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if self.shuffle_shards:
            shard_order = torch.randperm(len(self.logical_shards), generator=generator).tolist()
        else:
            shard_order = list(range(len(self.logical_shards)))

        shard_indices_by_rank = [[] for _ in range(self.num_replicas)]
        samples_by_rank = [0 for _ in range(self.num_replicas)]
        for shard_idx in shard_order:
            target_rank = min(
                range(self.num_replicas),
                key=lambda replica: (samples_by_rank[replica], replica),
            )
            shard_indices_by_rank[target_rank].append(shard_idx)
            samples_by_rank[target_rank] += self.logical_shards[shard_idx].size

        num_samples_per_rank = min(samples_by_rank) // self.batch_size * self.batch_size
        if num_samples_per_rank <= 0:
            raise ValueError(
                "RAM shard loading produced zero usable samples per rank. "
                "Reduce batch size or use smaller feature shards."
            )

        self._cached_epoch = self.epoch
        self._cached_plan = {
            "shard_indices_by_rank": shard_indices_by_rank,
            "samples_by_rank": samples_by_rank,
            "num_samples_per_rank": num_samples_per_rank,
            "num_batches": num_samples_per_rank // self.batch_size,
            "logical_shard_count": len(self.logical_shards),
            "logical_shard_source": self.logical_shard_store.name,
        }
        return self._cached_plan

    def describe_current_plan(self) -> Dict[str, object]:
        plan = self._build_epoch_plan()
        return {
            "logical_shard_source": plan["logical_shard_source"],
            "logical_shard_count": plan["logical_shard_count"],
            "num_batches": plan["num_batches"],
            "num_samples_per_rank": plan["num_samples_per_rank"],
            "samples_by_rank": list(plan["samples_by_rank"]),
            "dropped_samples_per_rank": [
                total - plan["num_samples_per_rank"] for total in plan["samples_by_rank"]
            ],
        }

    def _load_logical_shard(self, shard_span: LogicalShardSpan) -> Dict[str, np.ndarray]:
        latent_rows = _load_feature_range_to_ram(
            self.latent_store, shard_span.global_start, shard_span.global_end
        )
        dino_rows = _load_feature_range_to_ram(
            self.dino_store, shard_span.global_start, shard_span.global_end
        )

        if not np.array_equal(latent_rows["sample_id"], dino_rows["sample_id"]):
            raise ValueError(
                "Latent and DINO sample_id alignment diverged while materializing RAM shard "
                f"[{shard_span.global_start}, {shard_span.global_end})."
            )
        if not np.array_equal(latent_rows["label"], dino_rows["label"]):
            raise ValueError(
                "Latent and DINO labels diverged while materializing RAM shard "
                f"[{shard_span.global_start}, {shard_span.global_end})."
            )

        return {
            "latent": latent_rows["feature"],
            "dino": dino_rows["feature"],
            "y": latent_rows["label"],
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise RuntimeError(
                "RamLoadedShardDataset does not support DataLoader workers. Use num_workers=0."
            )

        plan = self._build_epoch_plan()
        target_samples = plan["num_samples_per_rank"]
        emitted_samples = 0
        carry_batch = None

        for shard_idx in plan["shard_indices_by_rank"][self.rank]:
            if emitted_samples >= target_samples:
                break

            shard_span = self.logical_shards[shard_idx]
            shard_rows = self._load_logical_shard(shard_span)
            shard_size = shard_rows["y"].shape[0]
            cursor = 0

            if carry_batch is not None:
                needed = self.batch_size - carry_batch["y"].shape[0]
                take = min(needed, shard_size)
                carry_batch = {
                    "latent": np.concatenate(
                        [carry_batch["latent"], shard_rows["latent"][:take]], axis=0
                    ),
                    "dino": np.concatenate(
                        [carry_batch["dino"], shard_rows["dino"][:take]], axis=0
                    ),
                    "y": np.concatenate(
                        [carry_batch["y"], shard_rows["y"][:take]], axis=0
                    ),
                }
                cursor = take
                if carry_batch["y"].shape[0] == self.batch_size:
                    yield {
                        "latent": torch.from_numpy(carry_batch["latent"]),
                        "dino": torch.from_numpy(carry_batch["dino"]),
                        "y": torch.from_numpy(carry_batch["y"]),
                    }
                    emitted_samples += self.batch_size
                    carry_batch = None

            while (
                cursor + self.batch_size <= shard_size
                and emitted_samples + self.batch_size <= target_samples
            ):
                batch_slice = slice(cursor, cursor + self.batch_size)
                yield {
                    "latent": torch.from_numpy(shard_rows["latent"][batch_slice]),
                    "dino": torch.from_numpy(shard_rows["dino"][batch_slice]),
                    "y": torch.from_numpy(shard_rows["y"][batch_slice]),
                }
                emitted_samples += self.batch_size
                cursor += self.batch_size

            if emitted_samples < target_samples and cursor < shard_size:
                max_leftover = min(shard_size - cursor, target_samples - emitted_samples)
                carry_slice = slice(cursor, cursor + max_leftover)
                carry_batch = {
                    "latent": np.array(shard_rows["latent"][carry_slice], copy=True),
                    "dino": np.array(shard_rows["dino"][carry_slice], copy=True),
                    "y": np.array(shard_rows["y"][carry_slice], copy=True),
                }

            del shard_rows
            gc.collect()

        if carry_batch is not None:
            raise RuntimeError(
                "RAM shard loading finished with an incomplete batch. "
                "This should not happen when num_samples_per_rank is batch-aligned."
            )
        if emitted_samples != target_samples:
            raise RuntimeError(
                f"RAM shard loading emitted {emitted_samples} samples on rank {self.rank}, "
                f"expected {target_samples}."
            )

    def __len__(self):
        return self._build_epoch_plan()["num_batches"]

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._cached_epoch = None
        self._cached_plan = None


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
