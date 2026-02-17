from torch.utils.data import Dataset
import torch
import os
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        features = np.array(self.hf_dataset[idx]["feature"])
        labels = np.array(self.hf_dataset[idx]["label"])
        print(labels)
        # make sure the dims are features are 3d and labels are 1d
        assert features.ndim == 3, f"Expected features to be 3D, but got {features.ndim}D"
        assert labels.ndim == 1, f"Expected labels to be 1D, but got {labels.ndim}D"
        return {"x": torch.from_numpy(features), "y": torch.from_numpy(labels).long()}
