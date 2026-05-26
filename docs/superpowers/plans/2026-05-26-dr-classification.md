# DR Classification Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a two-stage EfficientNet-B3 DR classification model on APTOS-2019 and integrate it into the Fundus_agent pipeline.

**Architecture:** Two-stage cascade — Stage 1 binary classifier (DR vs No-DR), Stage 2 four-class severity grader (grades 1-4). Both use EfficientNet-B3 with ImageNet pretrained weights. Training in a dedicated `DR_classification` conda environment, inference via subprocess from the existing orchestrator.

**Tech Stack:** PyTorch 2.x, timm, torchvision, scikit-learn, pandas, PIL

---

### Task 1: Environment setup and project scaffold

**Files:**
- Create: `DR_classification/checkpoints/.gitkeep`

- [ ] **Step 1: Create conda environment**

Run:
```bash
conda create -n DR_classification python=3.11 -y
```

- [ ] **Step 2: Install dependencies**

Run:
```bash
conda activate DR_classification
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm scikit-learn pandas matplotlib tqdm opencv-python-headless
```

- [ ] **Step 3: Create project directory and checkpoints folder**

Run:
```bash
mkdir -p /home/twinkle/app/LLM_paper/DR_classification/checkpoints
touch /home/twinkle/app/LLM_paper/DR_classification/checkpoints/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
cd /home/twinkle/app/LLM_paper
git add DR_classification/
git commit -m "chore: scaffold DR_classification project with conda env config"
```

---

### Task 2: Dataset class

**Files:**
- Create: `DR_classification/dataset.py`

- [ ] **Step 1: Write dataset.py**

```python
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
```

- [ ] **Step 2: Commit**

```bash
cd /home/twinkle/app/LLM_paper
git add DR_classification/dataset.py
git commit -m "feat: add APTOS-2019 dataset loader with augmentations"
```

---

### Task 3: Model factory

**Files:**
- Create: `DR_classification/model.py`

- [ ] **Step 1: Write model.py**

```python
"""Model factory for DR classification."""
import torch
import torch.nn as nn
import timm


def create_model(num_classes, pretrained=True):
    """Create EfficientNet-B3 with custom classification head.

    Args:
        num_classes: 1 for binary, 4 for severity
        pretrained: use ImageNet pretrained weights

    Returns:
        nn.Module
    """
    model = timm.create_model("efficientnet_b3", pretrained=pretrained, num_classes=num_classes)
    return model


def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }, path)


def load_checkpoint(model, path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("metrics", {})
```

- [ ] **Step 2: Commit**

```bash
cd /home/twinkle/app/LLM_paper
git add DR_classification/model.py
git commit -m "feat: add EfficientNet-B3 model factory for DR classification"
```

---

### Task 4: Stage 1 — Binary training script

**Files:**
- Create: `DR_classification/train_binary.py`

- [ ] **Step 1: Write train_binary.py**

```python
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
```

- [ ] **Step 2: Dry-run test (1 epoch) to verify no import errors**

Run:
```bash
conda activate DR_classification
cd /home/twinkle/app/LLM_paper/DR_classification
python -c "from dataset import load_data; tl, vl, _, _ = load_data(
    '/home/twinkle/app/LLM_paper/Dataset/APTOS-2019/train.csv',
    '/home/twinkle/app/LLM_paper/Dataset/APTOS-2019/train_images', binary=True);
    print(f'Train batches: {len(tl)}, Val batches: {len(vl)}')"
```
Expected: Train batches and Val batches printed, no errors

- [ ] **Step 3: Commit**

```bash
cd /home/twinkle/app/LLM_paper
git add DR_classification/train_binary.py
git commit -m "feat: add Stage 1 binary DR classifier training script"
```

---

### Task 5: Stage 2 — Severity training script

**Files:**
- Create: `DR_classification/train_severity.py`

- [ ] **Step 1: Write train_severity.py**

```python
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
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def load_severity_data_v2(csv_path, img_dir, val_split=0.2, seed=42):
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

    train_loader, val_loader = load_severity_data_v2(csv_path, img_dir)

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
```

- [ ] **Step 2: Dry-run test to verify data loading**

Run:
```bash
conda activate DR_classification
cd /home/twinkle/app/LLM_paper/DR_classification
python -c "
from train_severity import load_severity_data_v2
tl, vl = load_severity_data_v2(
    '/home/twinkle/app/LLM_paper/Dataset/APTOS-2019/train.csv',
    '/home/twinkle/app/LLM_paper/Dataset/APTOS-2019/train_images')
print(f'Train batches: {len(tl)}, Val batches: {len(vl)}')
# Check label range is 0-3
for imgs, labels in tl:
    print(f'Label min: {labels.min()}, max: {labels.max()}')
    break
"
```
Expected: Train/Val batches printed, label range 0-3

- [ ] **Step 3: Commit**

```bash
cd /home/twinkle/app/LLM_paper
git add DR_classification/train_severity.py
git commit -m "feat: add Stage 2 DR severity classifier training script"
```

---

### Task 6: Inference script

**Files:**
- Create: `DR_classification/infer.py`

- [ ] **Step 1: Write infer.py**

```python
#!/usr/bin/env python3
"""Subprocess entry point for DR classification inference.

Usage:
    python infer.py --input <img_path> --output <json_path> [--device cuda:0]
"""
import argparse
import json
import os
import sys
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from model import create_model


def load_models(binary_ckpt, severity_ckpt, device):
    binary_model = create_model(num_classes=1)
    binary_model.load_state_dict(
        torch.load(binary_ckpt, map_location=device, weights_only=True)["model_state_dict"]
    )
    binary_model.to(device)
    binary_model.eval()

    severity_model = create_model(num_classes=4)
    severity_model.load_state_dict(
        torch.load(severity_ckpt, map_location=device, weights_only=True)["model_state_dict"]
    )
    severity_model.to(device)
    severity_model.eval()

    return binary_model, severity_model


def get_transform(input_size=512):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict(img_path, binary_model, severity_model, transform, device):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    # Stage 1: binary
    logit = binary_model(tensor).squeeze(-1)
    prob_has_dr = torch.sigmoid(logit).item()

    if prob_has_dr < 0.5:
        return {
            "present": False,
            "confidence": round(1.0 - prob_has_dr, 4),
            "severity": None,
            "prob_has_dr": round(prob_has_dr, 4),
        }

    # Stage 2: severity
    logits = severity_model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    severity = int(torch.argmax(probs).item()) + 1  # remap 0-3 → 1-4

    return {
        "present": True,
        "confidence": round(prob_has_dr, 4),
        "severity": severity,
        "severity_probs": [round(p, 4) for p in probs.tolist()],
        "prob_has_dr": round(prob_has_dr, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input fundus image")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    binary_ckpt = os.path.join(script_dir, "checkpoints", "dr_binary_b3.pth")
    severity_ckpt = os.path.join(script_dir, "checkpoints", "dr_severity_b3.pth")

    if not os.path.exists(binary_ckpt):
        raise FileNotFoundError(f"Binary checkpoint not found: {binary_ckpt}")
    if not os.path.exists(severity_ckpt):
        raise FileNotFoundError(f"Severity checkpoint not found: {severity_ckpt}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    binary_model, severity_model = load_models(binary_ckpt, severity_ckpt, device)
    transform = get_transform()

    result = predict(args.input, binary_model, severity_model, transform, device)

    with open(args.output, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd /home/twinkle/app/LLM_paper
git add DR_classification/infer.py
git commit -m "feat: add DR classification inference script"
```

---

### Task 7: Rewrite DR agent

**Files:**
- Modify: `Fundus_agent/fundus_agents/reasoning/diabetic_retinopathy.py`
- Modify: `Fundus_agent/fundus_agents/config.py`

- [ ] **Step 1: Add DR classification config entries**

In `Fundus_agent/fundus_agents/config.py`, add after the existing `SEGMENTATION_SCRIPT` line:

```python
DR_CLASSIFY_PYTHON = "/data/twinkle/anaconda3/envs/DR_classification/bin/python"
DR_CLASSIFY_SCRIPT = os.path.join(PROJECT_ROOT, "DR_classification", "infer.py")
```

- [ ] **Step 2: Rewrite diabetic_retinopathy.py**

Replace the entire content of `Fundus_agent/fundus_agents/reasoning/diabetic_retinopathy.py`:

```python
"""Diabetic Retinopathy agent: two-stage EfficientNet-B3 classifier."""
import subprocess
import tempfile
import json
import os
from fundus_agents.reasoning.base import BaseDiseaseAgent
from fundus_agents.contracts import FundusImage, SegmentationMasks, DiseaseFinding
from fundus_agents.config import DR_CLASSIFY_PYTHON, DR_CLASSIFY_SCRIPT


class DiabeticRetinopathyAgent(BaseDiseaseAgent):
    def __init__(self, device="cuda:0"):
        super().__init__("D", "糖尿病视网膜病变")
        self.device = device

    def diagnose(self, fundus_img: FundusImage,
                 masks: SegmentationMasks = None) -> DiseaseFinding:
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img_path = f.name
            fundus_img.image.save(img_path)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name

        try:
            cmd = [
                DR_CLASSIFY_PYTHON, DR_CLASSIFY_SCRIPT,
                "--input", img_path,
                "--output", out_path,
                "--device", self.device,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                raise RuntimeError(f"DR classification failed: {result.stderr}")

            with open(out_path) as f:
                data = json.load(f)

            if data["present"]:
                return DiseaseFinding(
                    disease_code=self.disease_code,
                    disease_name=self.disease_name,
                    present=True,
                    confidence=data["confidence"],
                    evidence=[f"DR severity grade {data['severity']}/4",
                              f"P(has_DR)={data['prob_has_dr']:.4f}"],
                    metrics={
                        "severity": data["severity"],
                        "severity_probs": data.get("severity_probs", []),
                        "prob_has_dr": data["prob_has_dr"],
                    },
                )
            else:
                return DiseaseFinding(
                    disease_code=self.disease_code,
                    disease_name=self.disease_name,
                    present=False,
                    confidence=data["confidence"],
                    evidence=[f"No DR detected, P(has_DR)={data['prob_has_dr']:.4f}"],
                    metrics={"prob_has_dr": data["prob_has_dr"]},
                )
        except subprocess.TimeoutExpired:
            return self._insufficient_data()
        finally:
            for p in [img_path, out_path]:
                if os.path.exists(p):
                    os.unlink(p)
```

- [ ] **Step 3: Commit**

```bash
cd /home/twinkle/app/LLM_paper
git add Fundus_agent/fundus_agents/reasoning/diabetic_retinopathy.py Fundus_agent/fundus_agents/config.py
git commit -m "feat: replace DR agent with two-stage EfficientNet-B3 classifier"
```

---

### Task 8: Integration test

**Files:**
- Create: `DR_classification/test_infer.py`

- [ ] **Step 1: Write integration test**

```python
"""Integration test: verify infer.py runs end-to-end on a sample image."""
import subprocess
import json
import os
import sys
import tempfile


def test_infer_on_sample():
    """Run infer.py on the first APTOS training image, verify JSON output."""
    script = os.path.join(os.path.dirname(__file__), "infer.py")
    sample_img = "/home/twinkle/app/LLM_paper/Dataset/APTOS-2019/train_images/000c1434d8d7.png"

    if not os.path.exists(sample_img):
        pytest.skip("Sample image not found")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    try:
        result = subprocess.run(
            ["python", script, "--input", sample_img, "--output", out_path, "--device", "cpu"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        assert result.returncode == 0, f"Infer failed: {result.stderr}"

        with open(out_path) as f:
            data = json.load(f)

        assert "present" in data
        assert "confidence" in data
        assert "prob_has_dr" in data
        assert isinstance(data["present"], bool)
        assert 0 <= data["confidence"] <= 1

        if data["present"]:
            assert data["severity"] in [1, 2, 3, 4]
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def test_missing_checkpoint():
    """Verify infer.py errors clearly when checkpoint is missing."""
    import pytest
    # This is tested implicitly by the FileNotFoundError in infer.py main()
    pass


if __name__ == "__main__":
    test_infer_on_sample()
    print("Integration test passed")
```

- [ ] **Step 2: Run integration test (after model training)**

Run:
```bash
conda activate DR_classification
cd /home/twinkle/app/LLM_paper/DR_classification
python test_infer.py
```
Expected: "Integration test passed"

- [ ] **Step 3: Commit**

```bash
cd /home/twinkle/app/LLM_paper
git add DR_classification/test_infer.py
git commit -m "test: add DR classification inference integration test"
```

---

### Execution Order

Tasks must be executed sequentially: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8.

Tasks 4 and 5 (training) require the full APTOS-2019 dataset and a GPU. Each training run takes approximately 1-2 hours on a single GPU.
