#!/usr/bin/env python3
"""Train Stage 2: DR severity classifier (grades 1-4).
Only trained on DR-positive samples (diagnosis 1-4)."""
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from PIL import Image
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_transforms
from model import create_model, save_checkpoint
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit


class SeverityAPTOSDataset(Dataset):
    """APTOS-2019 dataset variant for severity classification (labels 0-3)."""

    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = df["label"].values.astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]["id_code"] + ".png")
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            import logging
            logging.warning("Missing image, using blank: %s", img_path)
            img = Image.new("RGB", (512, 512), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def load_severity_data(csv_path, img_dir, val_split=0.2, seed=42):
    """Load only DR-positive samples, labels 0-3."""
    df = pd.read_csv(csv_path)
    df = df[df["diagnosis"] >= 1].copy()
    df["label"] = df["diagnosis"] - 1

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(sss.split(df, df["label"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_ds = SeverityAPTOSDataset(train_df, img_dir, get_transforms(train=True))
    val_ds = SeverityAPTOSDataset(val_df, img_dir, get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "qwk": cohen_kappa_score(all_labels, all_preds, weights="quadratic"),
    }
    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    csv_path = "/home/twinkle/app/LLM_paper/Dataset/APTOS-2019/train.csv"
    img_dir = "/home/twinkle/app/LLM_paper/Dataset/APTOS-2019/train_images"
    save_path = "/home/twinkle/app/LLM_paper/DR_classification/checkpoints/dr_severity_b3.pth"

    train_loader, val_loader = load_severity_data(csv_path, img_dir)

    model = create_model(num_classes=4).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    # Compute class weights for imbalanced severity classes
    all_labels_in_train = []
    for _, labels in train_loader:
        all_labels_in_train.extend(labels.tolist())
    class_counts = np.bincount(all_labels_in_train, minlength=4)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 4
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_qwk = -1
    patience_counter = 0

    for epoch in range(50):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1:2d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
              f"val_qwk={val_metrics['qwk']:.4f}")

        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch + 1, val_metrics, save_path)
            print(f"  -> saved best model (qwk={best_qwk:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Training complete. Best val QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    main()
