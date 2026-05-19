#!/usr/bin/env python3
"""Subprocess entry point for segmentation inference (runs in openmmlab env).

Usage:
    python run_segmentation.py --checkpoint <path> --input <img> --output <npy> --target disc_cup
"""
import argparse
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="disc_cup")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Load model via mmseg
    from mmseg.apis import init_model, inference_model

    # Infer config from checkpoint path if not provided
    config_path = args.config
    if not config_path:
        import os, glob
        # Look in the same directory as checkpoint first
        ckpt_dir = os.path.dirname(args.checkpoint)
        configs = glob.glob(os.path.join(ckpt_dir, "*.py"))
        if configs:
            config_path = configs[0]
        else:
            # Look one level up
            work_dir = os.path.dirname(ckpt_dir)
            configs = glob.glob(os.path.join(work_dir, "*.py"))
            if configs:
                config_path = configs[0]

    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No config file found for checkpoint {args.checkpoint}"
        )

    model = init_model(config_path, args.checkpoint, device=args.device)

    # Run inference
    img = np.array(Image.open(args.input).convert("RGB"))
    result = inference_model(model, img)

    # Extract masks based on target
    masks = {}
    if args.target == "disc_cup":
        seg_map = result.pred_sem_seg.data.cpu().numpy()
        masks["disc"] = (seg_map == 1).astype(np.uint8)
        masks["cup"] = (seg_map == 2).astype(np.uint8)
    elif args.target == "vessels":
        seg_map = result.pred_sem_seg.data.cpu().numpy()
        masks["vessels"] = (seg_map == 1).astype(np.uint8)
    elif args.target == "macula":
        seg_map = result.pred_sem_seg.data.cpu().numpy()
        masks["macula"] = (seg_map == 1).astype(np.uint8)
    elif args.target == "lesions":
        seg_map = result.pred_sem_seg.data.cpu().numpy()
        masks["lesions"] = {
            "hemorrhages": (seg_map == 1).astype(np.uint8),
            "hard_exudates": (seg_map == 2).astype(np.uint8),
            "soft_exudates": (seg_map == 3).astype(np.uint8),
            "microaneurysms": (seg_map == 4).astype(np.uint8),
            "drusen": (seg_map == 5).astype(np.uint8),
        }

    np.save(args.output, masks)


if __name__ == "__main__":
    main()
