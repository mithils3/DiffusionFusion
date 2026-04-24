import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Iterator, Optional

import numpy as np
import timm
import torch
from datasets import load_dataset, load_from_disk

from JiT.util.feature_shards import (
    DatasetShardSpan,
    FeatureShardStore,
    LogicalShardSpan,
    PairedRamLoadedShardDataset,
    inspect_feature_shards,
    load_feature_range_to_ram,
    maybe_append_split_suffix,
    resolve_feature_dir_name,
    resolve_feature_dataset_root,
)
from JiT.util.image_transforms import build_center_crop_normalize_transform


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

        rank_ids = sorted(
            {_parse_rank_from_shard_path(span.path) for span in sample_id_store.shard_spans}
        )
        if not rank_ids:
            raise ValueError("Cannot build RawImageStore without feature shards.")

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
            dataset_indices < self.dataset_size,
            dataset_indices,
            dataset_indices - self.dataset_size,
        )
        return dataset_indices.astype(np.int64, copy=False)

    def load_batch(self, sample_ids: np.ndarray) -> torch.Tensor:
        dataset_indices = self.sample_ids_to_dataset_indices(sample_ids)
        rows = self.dataset[dataset_indices.tolist()]
        images = [self.transform(image.convert("RGB")) for image in rows["image"]]
        return torch.stack(images, dim=0)


class RamLoadedShardDataset(PairedRamLoadedShardDataset):
    """Decoder batch wrapper for paired feature shards plus aligned raw images."""

    def __init__(
        self,
        latent_store: FeatureShardStore,
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
        image_size: int = 256,
    ):
        if not image_data_path:
            raise ValueError(
                "RamLoadedShardDataset requires image_data_path because decoder training always needs raw images."
            )
        super().__init__(
            latent_store=latent_store,
            dino_store=dino_store,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            shuffle_shards=shuffle_shards,
            seed=seed,
            preload_next_shard=preload_next_shard,
        )
        self.preload_next_batch = preload_next_batch
        self.image_store = RawImageStore(
            image_data_path,
            latent_store,
            split=image_data_split,
            model_name=image_model_name,
            image_size=image_size,
        )

    def _format_batch(self, rows: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            "latent": torch.from_numpy(rows["latent"]),
            # DINO features are normalized once during extraction.
            # Repeating layer norm here silently changes float16 shards.
            "dino": torch.from_numpy(rows["dino"]),
            "y": torch.from_numpy(rows["y"]),
            "sample_id": torch.from_numpy(rows["sample_id"]),
            "image": self.image_store.load_batch(rows["sample_id"]),
        }

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

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise RuntimeError(
                "RamLoadedShardDataset does not support DataLoader workers. Use num_workers=0."
            )
        batch_rows_iter = self._iter_batch_rows()
        yield from self._iter_prefetched_batches(batch_rows_iter)


__all__ = [
    "DatasetShardSpan",
    "FeatureShardStore",
    "LogicalShardSpan",
    "RamLoadedShardDataset",
    "RawImageStore",
    "inspect_feature_shards",
    "load_feature_range_to_ram",
    "maybe_append_split_suffix",
    "resolve_feature_dir_name",
    "resolve_feature_dataset_root",
]
