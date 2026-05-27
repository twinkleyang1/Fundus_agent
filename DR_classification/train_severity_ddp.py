#!/usr/bin/env python3
"""Train Stage 2 DR severity classifier with DistributedDataParallel (4 GPUs)."""
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_transforms
from model import create_model, save_checkpoint


class SeverityAPTOSDataset(Dataset):
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
            img = Image.new("RGB", (512, 512), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def load_severity_data_ddp(csv_path, img_dir, val_split=0.2, seed=42, batch_size=16):
    df = pd.read_csv(csv_path)
    df = df[df["diagnosis"] >= 1].copy()
    df["label"] = df["diagnosis"] - 1

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(sss.split(df, df["label"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_ds = SeverityAPTOSDataset(train_df, img_dir, get_transforms(train=True))
    val_ds = SeverityAPTOSDataset(val_df, img_dir, get_transforms(train=False))

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler,
                            num_workers=0, pin_memory=True)

    return train_loader, val_loader, train_sampler


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False, disable=dist.get_rank() != 0):
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
    for imgs, labels in tqdm(loader, desc="Val", leave=False, disable=dist.get_rank() != 0):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Gather predictions from all ranks
    all_preds = torch.tensor(all_preds, device=device)
    all_labels = torch.tensor(all_labels, device=device)
    gathered_preds = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
    gathered_labels = [torch.zeros_like(all_labels) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_preds, all_preds)
    dist.all_gather(gathered_labels, all_labels)

    if dist.get_rank() == 0:
        all_preds = torch.cat(gathered_preds).cpu().tolist()
        all_labels = torch.cat(gathered_labels).cpu().tolist()

        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(cm)

        metrics = {
            "loss": total_loss / len(loader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "qwk": cohen_kappa_score(all_labels, all_preds, weights="quadratic"),
        }
        return metrics
    return None


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print(f"DDP training on {dist.get_world_size()} GPUs", flush=True)

    csv_path = "/home/twinkle/app/LLM_paper/Dataset/APTOS-2019/train.csv"
    img_dir = "/home/twinkle/app/LLM_paper/Dataset/APTOS-2019/train_images"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "checkpoints", "dr_severity_b3_ddp.pth")

    train_loader, val_loader, train_sampler = load_severity_data_ddp(
        csv_path, img_dir, batch_size=16)

    if local_rank == 0:
        print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)

    model = create_model(num_classes=4).to(device)
    model = DDP(model, device_ids=[local_rank])

    if local_rank == 0:
        print("  Model ready", flush=True)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    df = pd.read_csv(csv_path)
    pos_df = df[df["diagnosis"] >= 1]
    labels_0_3 = pos_df["diagnosis"].values - 1
    class_counts = np.bincount(labels_0_3, minlength=4)
    class_weights = 1.0 / (class_counts.astype(np.float64) + 1e-6)
    class_weights = class_weights / class_weights.sum() * 4
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if local_rank == 0:
        print(f"  Class counts (0-3): {class_counts}", flush=True)
        print(f"  Class weights: {class_weights.tolist()}", flush=True)

    best_qwk = -1
    patience_counter = 0

    for epoch in range(50):
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        if local_rank == 0:
            print(f"Epoch {epoch+1:2d} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
                  f"val_qwk={val_metrics['qwk']:.4f}", flush=True)

            if val_metrics["qwk"] > best_qwk:
                best_qwk = val_metrics["qwk"]
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch + 1, val_metrics, save_path)
                print(f"  -> saved best model (qwk={best_qwk:.4f})", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch+1}", flush=True)
                    break

    dist.barrier()
    if local_rank == 0:
        print(f"Training complete. Best val QWK: {best_qwk:.4f}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
