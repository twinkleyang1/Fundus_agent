#!/usr/bin/env python3
"""Train Stage 1: DR binary classifier (0 vs 1-4)."""
import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_data
from model import create_model, save_checkpoint


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        logits = model(imgs).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device).float()
        logits = model(imgs).squeeze(-1)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().long().tolist())
        all_probs.extend(probs.cpu().tolist())

    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs),
    }
    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    csv_path = "/home/twinkle/app/LLM_paper/Dataset/APTOS-2019/train.csv"
    img_dir = "/home/twinkle/app/LLM_paper/Dataset/APTOS-2019/train_images"
    save_path = "/home/twinkle/app/LLM_paper/DR_classification/checkpoints/dr_binary_b3.pth"

    train_loader, val_loader, _, _ = load_data(csv_path, img_dir, binary=True)

    model = create_model(num_classes=1).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    patience_counter = 0

    for epoch in range(50):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1:2d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | val_auc={val_metrics['auc']:.4f} | "
              f"val_f1={val_metrics['f1']:.4f}")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch + 1, val_metrics, save_path)
            print(f"  -> saved best model (auc={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Training complete. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
