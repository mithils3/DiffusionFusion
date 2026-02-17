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
        features = torch.tensor(self.hf_dataset[idx]["feature"])
        labels = torch.tensor(self.hf_dataset[idx]["label"]).long()
        print(labels)
        # make sure the dims are features are 3d and labels are 1d
        assert features.ndim == 3, f"Expected features to be 3D, but got {features.ndim}D"
        return {"x": features, "y": labels}
