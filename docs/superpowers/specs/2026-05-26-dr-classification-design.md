# DR Classification Model — Design Doc

## Overview

用 APTOS-2019 数据集训练 DR 分类模型，替代 Fundus_agent 中现有的基于规则/VL 的 DR agent。采用两阶段级联方案：先判断有无 DR，再判断严重程度。

## Architecture

```
APTOS-2019 (3662 images)
         │
         ├──→ Stage 1: EfficientNet-B3 二分类
         │         ├── Label: 0→neg, 1-4→pos
         │         ├── Output: P(has_DR)
         │         └── Threshold: >0.5 → positive
         │
         └──→ Stage 2: EfficientNet-B3 四分类
                   ├── Label: 1→1, 2→2, 3→3, 4→4
                   ├── Only trained on DR-positive samples (1857)
                   └── Output: severity ∈ {1,2,3,4}
```

## Project Structure

```
/home/twinkle/app/LLM_paper/DR_classification/
├── train_binary.py       # Stage 1 training
├── train_severity.py     # Stage 2 training
├── dataset.py            # Data loading + augmentations
├── model.py              # Model factory (EfficientNet-B3)
├── infer.py              # Inference entry point
└── checkpoints/
    ├── dr_binary_b3.pth
    └── dr_severity_b3.pth
```

## Training Config

| Parameter | Value |
|-----------|-------|
| Backbone | EfficientNet-B3 (ImageNet pretrained) |
| Input size | 512×512 |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-4 |
| LR schedule | Cosine annealing |
| Epochs | 50, early stop patience=10 |
| Batch size | 32 |
| Loss (Stage 1) | BCEWithLogitsLoss |
| Loss (Stage 2) | CrossEntropyLoss (class weighted) |
| Data split | 80/20 stratified |
| Augmentations | RandomFlip, RandomRotation(±30°), ColorJitter, RandomResizedCrop |
| Environment | Conda env `DR_classification`, PyTorch 2.x + CUDA 12 |

## Integration

Replace `fundus_agents/reasoning/diabetic_retinopathy.py`:

- Remove dependency on `SegmentationMasks`
- Stage 1 inference first; if P(has_DR) < 0.5, return negative
- If positive, run Stage 2 to get severity
- Output `DiseaseFinding` with `present`, `confidence`, `severity`
- Orchestrator interface unchanged

## Metrics

- Stage 1: Accuracy, Precision, Recall, F1, AUC-ROC
- Stage 2: Accuracy, Quadratic Weighted Kappa, Confusion Matrix

## Error Handling

- Missing checkpoint → explicit error, no silent fallback
- Image read failure → skip and log
- GPU unavailable → fallback to CPU with warning
