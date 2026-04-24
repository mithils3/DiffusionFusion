from typing import Dict

import numpy as np
import torch

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


class RamLoadedShardDataset(PairedRamLoadedShardDataset):
    """Training batch wrapper for paired latent and DINO RAM shards."""

    def _format_batch(self, rows: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            "latent": torch.from_numpy(rows["latent"]),
            # DINO features are normalized once during extraction.
            # Repeating layer norm here silently changes float16 shards.
            "dino": torch.from_numpy(rows["dino"]),
            "y": torch.from_numpy(rows["y"]),
        }


__all__ = [
    "DatasetShardSpan",
    "FeatureShardStore",
    "LogicalShardSpan",
    "RamLoadedShardDataset",
    "inspect_feature_shards",
    "load_feature_range_to_ram",
    "maybe_append_split_suffix",
    "resolve_feature_dir_name",
    "resolve_feature_dataset_root",
]
