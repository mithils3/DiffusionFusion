import gc
import json
import os
from bisect import bisect_right
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from glob import glob
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import timm
import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk
from torch.utils.data import IterableDataset

from JiT.util.feature_normalization import normalize_feature_map_tokens
from JiT.util.image_transforms import build_center_crop_normalize_transform


def resolve_feature_dataset_root(data_path: str, dataset_dir_name: str) -> str:
    if os.path.isdir(dataset_dir_name):
        return dataset_dir_name
    return os.path.join(data_path, dataset_dir_name)


def _describe_file_state(path: str) -> str:
    if not os.path.exists(path):
        return "missing"
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return f"unreadable ({exc})"
    if size == 0:
        return "empty"
    return f"{size} bytes"


def _feature_shard_dirs(dataset_root: str) -> List[str]:
    pattern = os.path.join(
        dataset_root, "shard_[0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9]"
    )
    return sorted(path for path in glob(pattern) if os.path.isdir(path))


def _load_feature_shard_dataset(shard_dir: str, dataset_dir_name: str):
    state_path = os.path.join(shard_dir, "state.json")
    dataset_info_path = os.path.join(shard_dir, "dataset_info.json")
    file_states = {
        "state.json": _describe_file_state(state_path),
        "dataset_info.json": _describe_file_state(dataset_info_path),
    }
    missing_or_empty = [
        name for name, state in file_states.items() if state in {"missing", "empty"}
    ]
    if missing_or_empty:
        raise RuntimeError(
            f"{dataset_dir_name} shard {shard_dir} is incomplete: "
            f"{', '.join(f'{name}={file_states[name]}' for name in missing_or_empty)}. "
            "This usually means feature extraction was interrupted after creating the shard "
            "directory. Delete the broken shard directory and regenerate the features."
        )

    try:
        return load_from_disk(shard_dir)
    except (json.JSONDecodeError, FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to load {dataset_dir_name} shard {shard_dir}. "
            f"Metadata status: state.json={file_states['state.json']}, "
            f"dataset_info.json={file_states['dataset_info.json']}. "
            "This shard is likely incomplete or corrupt; delete it and regenerate the "
            f"features. Original error: {type(exc).__name__}: {exc}"
        ) from exc


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


def _parse_rank_from_shard_path(path: str) -> int:
    shard_name = os.path.basename(path)
    parts = shard_name.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Unable to infer rank from shard path {path}; expected shard_XXXXX_YYYYY naming."
        )
    return int(parts[1])


def _load_raw_image_dataset(data_path: str, split: str = "train"):
    if os.path.isdir(data_path):
        dataset = load_from_disk(data_path)
        if hasattr(dataset, "keys"):
            if split not in dataset:
                raise KeyError(
                    f'Expected a "{split}" split in decoder image dataset at {data_path}.'
                )
            return dataset[split]
        return dataset

    return load_dataset(data_path, split=split)


class RawImageStore:
    """Loads original RGB images using the shared center-crop decoder recipe."""

    def __init__(
        self,
        data_path: str,
        sample_id_store: FeatureShardStore,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        image_size: int = 224,
        split: str = "train",
    ):
        self.dataset = _load_raw_image_dataset(data_path, split=split)
        self.dataset_size = len(self.dataset)

        rank_ids = sorted({_parse_rank_from_shard_path(span.path)
                          for span in sample_id_store.shard_spans})
        if not rank_ids:
            raise ValueError(
                "Cannot build RawImageStore without feature shards.")

        self.extraction_world_size = rank_ids[-1] + 1
        if sample_id_store.total_size % self.extraction_world_size != 0:
            raise ValueError(
                f"Feature dataset size {sample_id_store.total_size} is not divisible by inferred extraction "
                f"world size {self.extraction_world_size}."
            )
        self.extraction_sampler_len = sample_id_store.total_size // self.extraction_world_size

        transform_model = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True,
        )
        data_config = timm.data.resolve_model_data_config(transform_model)
        self.transform = build_center_crop_normalize_transform(
            image_size=image_size,
            mean=data_config.get("mean"),
            std=data_config.get("std"),
        )
        del transform_model

    def sample_ids_to_dataset_indices(self, sample_ids: np.ndarray) -> np.ndarray:
        rank = sample_ids // self.extraction_sampler_len
        local = sample_ids % self.extraction_sampler_len
        dataset_indices = rank + local * self.extraction_world_size
        dataset_indices = np.where(
            dataset_indices < self.dataset_size, dataset_indices, dataset_indices - self.dataset_size)
        return dataset_indices.astype(np.int64, copy=False)

    def load_batch(self, sample_ids: np.ndarray) -> torch.Tensor:
        dataset_indices = self.sample_ids_to_dataset_indices(sample_ids)
        rows = self.dataset[dataset_indices.tolist()]
        images = [self.transform(image.convert("RGB"))
                  for image in rows["image"]]
        return torch.stack(images, dim=0)


@dataclass(frozen=True)
class LogicalShardSpan:
    global_start: int
    global_end: int

    @property
    def size(self) -> int:
        return self.global_end - self.global_start


def inspect_feature_shards(data_path: str, dataset_dir_name: str) -> FeatureShardStore:
    dataset_root = resolve_feature_dataset_root(data_path, dataset_dir_name)
    shard_dirs = _feature_shard_dirs(dataset_root)
    if not shard_dirs:
        raise FileNotFoundError(
            f"No shard directories found under {dataset_root}"
        )

    shard_spans = []
    total_size = 0
    prev_last_sample_id = None
    bytes_per_sample = None

    for shard_dir in shard_dirs:
        shard_dataset = _load_feature_shard_dataset(shard_dir, dataset_dir_name)
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
            bytes_per_sample = int(np.asarray(
                shard_dataset[0]["feature"]).nbytes)

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


def _slice_rows(
    rows: Dict[str, np.ndarray],
    row_slice: slice,
    *,
    copy: bool = False,
) -> Dict[str, np.ndarray]:
    if copy:
        return {key: np.array(value[row_slice], copy=True) for key, value in rows.items()}
    return {key: value[row_slice] for key, value in rows.items()}


def _concat_rows(*row_groups: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        key: np.concatenate([rows[key] for rows in row_groups], axis=0)
        for key in row_groups[0]
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
    the in-memory working set stays bounded even when the EVA and DINO
    shard sample counts differ.
    """

    def __init__(
        self,
        eva_store: FeatureShardStore,
        dino_store: FeatureShardStore,
        batch_size: int,
        num_replicas: int = -1,
        rank: int = -1,
        shuffle_shards: bool = True,
        seed: int = 0,
        preload_next_shard: bool = True,
        preload_next_batch: bool = True,
        image_data_path: Optional[str] = None,
        image_data_split: str = "train",
        image_model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        image_size: int = 224,
    ):
        if num_replicas < 0:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank < 0:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive for RAM shard loading.")

        if eva_store.total_size != dino_store.total_size:
            raise ValueError(
                f"EVA dataset has {eva_store.total_size} samples but DINO dataset has "
                f"{dino_store.total_size}; RAM shard loading requires aligned datasets."
            )
        if not image_data_path:
            raise ValueError(
                "RamLoadedShardDataset requires image_data_path because decoder training always needs raw images."
            )

        self.eva_store = eva_store
        self.dino_store = dino_store
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.preload_next_shard = preload_next_shard
        self.preload_next_batch = preload_next_batch
        self.epoch = 0
        self.image_store = RawImageStore(
            image_data_path,
            eva_store,
            split=image_data_split,
            model_name=image_model_name,
            image_size=image_size,
        )

        self.logical_shard_store = self._select_logical_shard_store()
        self.logical_shards = [
            LogicalShardSpan(span.global_start, span.global_end)
            for span in self.logical_shard_store.shard_spans
        ]
        if not self.logical_shards:
            raise ValueError(
                "RAM shard loading requires at least one logical shard.")

        self._cached_epoch = None
        self._cached_plan = None

    def _select_logical_shard_store(self) -> FeatureShardStore:
        return max(
            (self.eva_store, self.dino_store),
            key=lambda store: (
                store.bytes_per_sample,
                len(store.shard_spans),
                store is self.dino_store,
            ),
        )

    def _build_epoch_plan(self):
        if self._cached_epoch == self.epoch and self._cached_plan is not None:
            return self._cached_plan

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if self.shuffle_shards:
            shard_order = torch.randperm(
                len(self.logical_shards), generator=generator).tolist()
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

        num_samples_per_rank = min(
            samples_by_rank) // self.batch_size * self.batch_size
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
        eva_rows = _load_feature_range_to_ram(
            self.eva_store, shard_span.global_start, shard_span.global_end
        )
        dino_rows = _load_feature_range_to_ram(
            self.dino_store, shard_span.global_start, shard_span.global_end
        )

        if not np.array_equal(eva_rows["sample_id"], dino_rows["sample_id"]):
            raise ValueError(
                "EVA and DINO sample_id alignment diverged while materializing RAM shard "
                f"[{shard_span.global_start}, {shard_span.global_end})."
            )
        label_mismatch = eva_rows["label"] != dino_rows["label"]
        if np.any(label_mismatch):
            mismatch_indices = np.flatnonzero(label_mismatch)[:5]
            mismatch_examples = [
                (
                    int(eva_rows["sample_id"][idx]),
                    int(eva_rows["label"][idx]),
                    int(dino_rows["label"][idx]),
                )
                for idx in mismatch_indices
            ]
            raise ValueError(
                "EVA and DINO labels diverged for "
                f"{int(label_mismatch.sum())} samples in RAM shard "
                f"[{shard_span.global_start}, {shard_span.global_end}). "
                "Decoder training requires aligned labels. "
                f"Examples (sample_id, eva_label, dino_label): {mismatch_examples}"
            )

        return {
            "eva": eva_rows["feature"],
            "dino": dino_rows["feature"],
            "y": eva_rows["label"],
            "sample_id": eva_rows["sample_id"],
        }

    def _format_batch(self, rows: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        eva = normalize_feature_map_tokens(torch.from_numpy(rows["eva"]))
        dino = normalize_feature_map_tokens(torch.from_numpy(rows["dino"]))
        return {
            "eva": eva,
            "dino": dino,
            "y": torch.from_numpy(rows["y"]),
            "sample_id": torch.from_numpy(rows["sample_id"]),
            "image": self.image_store.load_batch(rows["sample_id"]),
        }

    def _iter_batch_rows(self) -> Iterator[Dict[str, np.ndarray]]:
        plan = self._build_epoch_plan()
        target_samples = plan["num_samples_per_rank"]
        emitted_samples = 0
        carry_batch = None

        shard_iter = self._iter_rank_shards(plan["shard_indices_by_rank"][self.rank])
        try:
            for _shard_span, shard_rows in shard_iter:
                if emitted_samples >= target_samples:
                    break

                shard_size = shard_rows["y"].shape[0]
                cursor = 0

                if carry_batch is not None:
                    needed = self.batch_size - carry_batch["y"].shape[0]
                    take = min(needed, shard_size)
                    carry_batch = _concat_rows(
                        carry_batch, _slice_rows(shard_rows, slice(0, take)))
                    cursor = take
                    if carry_batch["y"].shape[0] == self.batch_size:
                        yield carry_batch
                        emitted_samples += self.batch_size
                        carry_batch = None

                while (
                    cursor + self.batch_size <= shard_size
                    and emitted_samples + self.batch_size <= target_samples
                ):
                    batch_slice = slice(cursor, cursor + self.batch_size)
                    yield _slice_rows(shard_rows, batch_slice)
                    emitted_samples += self.batch_size
                    cursor += self.batch_size

                if emitted_samples < target_samples and cursor < shard_size:
                    max_leftover = min(shard_size - cursor,
                                       target_samples - emitted_samples)
                    carry_slice = slice(cursor, cursor + max_leftover)
                    carry_batch = _slice_rows(
                        shard_rows, carry_slice, copy=True)

                del shard_rows
                gc.collect()
        finally:
            close_fn = getattr(shard_iter, "close", None)
            if close_fn is not None:
                close_fn()

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

    def _iter_prefetched_batches(
        self,
        batch_rows_iter: Iterator[Dict[str, np.ndarray]],
    ) -> Iterator[Dict[str, torch.Tensor]]:
        if not self.preload_next_batch:
            for rows in batch_rows_iter:
                yield self._format_batch(rows)
            return

        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"decoder-batch-prefetch-r{self.rank}",
        )
        pending_future: Optional[Future] = None
        batch_iter = iter(batch_rows_iter)
        try:
            first_rows = next(batch_iter, None)
            if first_rows is None:
                return
            pending_future = executor.submit(self._format_batch, first_rows)

            for rows in batch_iter:
                current_future = pending_future
                if current_future is None:
                    raise RuntimeError(
                        "Decoder batch prefetcher lost track of the current batch future."
                    )
                pending_future = executor.submit(self._format_batch, rows)
                yield current_future.result()

            current_future = pending_future
            if current_future is None:
                raise RuntimeError(
                    "Decoder batch prefetcher lost track of the final batch future."
                )
            pending_future = None
            yield current_future.result()
        finally:
            if pending_future is not None:
                pending_future.cancel()
            close_fn = getattr(batch_rows_iter, "close", None)
            if close_fn is not None:
                close_fn()
            executor.shutdown(wait=True, cancel_futures=True)

    def _iter_rank_shards(
        self, shard_indices: List[int]
    ) -> Iterator[Tuple[LogicalShardSpan, Dict[str, np.ndarray]]]:
        if not shard_indices:
            return

        if not self.preload_next_shard or len(shard_indices) == 1:
            for shard_idx in shard_indices:
                shard_span = self.logical_shards[shard_idx]
                yield shard_span, self._load_logical_shard(shard_span)
            return

        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"ram-shard-prefetch-r{self.rank}",
        )
        pending_future: Optional[Future] = None
        try:
            first_span = self.logical_shards[shard_indices[0]]
            pending_future = executor.submit(
                self._load_logical_shard, first_span)

            for idx, shard_idx in enumerate(shard_indices):
                shard_span = self.logical_shards[shard_idx]
                current_future = pending_future
                if current_future is None:
                    raise RuntimeError(
                        "RAM shard prefetcher lost track of the current shard future.")

                if idx + 1 < len(shard_indices):
                    next_span = self.logical_shards[shard_indices[idx + 1]]
                    pending_future = executor.submit(
                        self._load_logical_shard, next_span)
                else:
                    pending_future = None

                yield shard_span, current_future.result()
        finally:
            if pending_future is not None:
                pending_future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise RuntimeError(
                "RamLoadedShardDataset does not support DataLoader workers. Use num_workers=0."
            )
        batch_rows_iter = self._iter_batch_rows()
        yield from self._iter_prefetched_batches(batch_rows_iter)

    def __len__(self):
        return self._build_epoch_plan()["num_batches"]

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._cached_epoch = None
        self._cached_plan = None
