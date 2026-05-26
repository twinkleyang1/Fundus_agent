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
