"""APTOS-2019 dataset loader with augmentations."""
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class APTOSDataset(Dataset):
    """Load APTOS-2019 fundus images with optional augmentations."""

    def __init__(self, df, img_dir, transform=None, binary=False):
        """
        Args:
            df: DataFrame with columns ['id_code', 'diagnosis']
            img_dir: path to image directory
            transform: albumentations or torchvision transform
            binary: if True, label 0→0, 1-4→1; else keep 0-4 as-is
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.binary = binary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["id_code"] + ".png")
        img = Image.open(img_path).convert("RGB")

        label = int(row["diagnosis"])
        if self.binary:
            label = 0 if label == 0 else 1

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


def get_transforms(train=True, input_size=512):
    """Build torchvision transforms for training or validation."""
    from torchvision import transforms

    if train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomResizedCrop(input_size, scale=(0.85, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def load_data(csv_path, img_dir, binary=False, input_size=512, val_split=0.2, seed=42):
    """Load csv, split into train/val, return DataLoaders."""
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch.utils.data import DataLoader

    df = pd.read_csv(csv_path)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(sss.split(df, df["diagnosis"]))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_ds = APTOSDataset(train_df, img_dir, get_transforms(train=True, input_size=input_size), binary=binary)
    val_ds = APTOSDataset(val_df, img_dir, get_transforms(train=False, input_size=input_size), binary=binary)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, train_df, val_df
