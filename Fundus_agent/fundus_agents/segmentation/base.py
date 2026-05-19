"""Base classes for segmentation workers."""
import abc
import subprocess
import tempfile
import os
import numpy as np
from fundus_agents.contracts import FundusImage, SegmentationMasks


class BaseSegmentationWorker(abc.ABC):
    @abc.abstractmethod
    def run(self, fundus_img: FundusImage) -> SegmentationMasks:
        ...


class SubprocessSegmentationWorker(BaseSegmentationWorker):
    """Runs segmentation via subprocess in the openmmlab conda environment."""

    def __init__(self, python_bin: str, script_path: str,
                 model_config: str, checkpoint: str, target: str):
        self.python_bin = python_bin
        self.script_path = script_path
        self.model_config = model_config
        self.checkpoint = checkpoint
        self.target = target

    def run(self, fundus_img: FundusImage) -> SegmentationMasks:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img_path = f.name
            fundus_img.image.save(img_path)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            out_path = f.name

        try:
            cmd = [
                self.python_bin, self.script_path,
                "--config", self.model_config,
                "--checkpoint", self.checkpoint,
                "--input", img_path,
                "--output", out_path,
                "--target", self.target,
            ]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Segmentation failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"Segmentation timed out after 120s (target={self.target})"
                )
            masks_dict = np.load(out_path, allow_pickle=True).item()
            return self._dict_to_masks(masks_dict)
        finally:
            for p in [img_path, out_path]:
                if os.path.exists(p):
                    os.unlink(p)

    def _dict_to_masks(self, masks_dict: dict) -> SegmentationMasks:
        from fundus_agents.contracts import BinaryMask, LesionMasks
        masks = SegmentationMasks(lesions=LesionMasks())
        for key in ["disc", "cup", "vessels", "macula"]:
            if key in masks_dict:
                setattr(masks, key, BinaryMask(data=masks_dict[key]))
        # Lesion sub-masks
        if "lesions" in masks_dict:
            lesion_dict = masks_dict["lesions"]
            for attr in ["hemorrhages", "hard_exudates", "soft_exudates",
                         "microaneurysms", "drusen"]:
                if attr in lesion_dict:
                    setattr(masks.lesions, attr,
                            BinaryMask(data=lesion_dict[attr]))
        return masks
