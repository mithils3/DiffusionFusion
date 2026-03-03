from torch.utils.data import Dataset
import torch
import os
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, latent_dataset, dino_dataset):
        self.latent_dataset = latent_dataset
        self.dino_dataset = dino_dataset

    def __len__(self):
        return len(self.latent_dataset)

    def __getitem__(self, idx):
        latent = torch.tensor(self.latent_dataset[idx]["feature"])
        dino = torch.tensor(self.dino_dataset[idx]["feature"])
        label = torch.tensor(self.latent_dataset[idx]["label"]).long()
        return {"latent": latent, "dino": dino, "y": label}
