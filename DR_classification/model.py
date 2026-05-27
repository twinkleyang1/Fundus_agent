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
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }, path)


def load_checkpoint(model, path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("metrics", {})
