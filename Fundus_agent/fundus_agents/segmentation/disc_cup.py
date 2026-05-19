"""Optic disc and cup segmentation using existing trained SegFormer model."""
import os
import glob
from fundus_agents.contracts import FundusImage, SegmentationMasks, LesionMasks
from fundus_agents.segmentation.base import BaseSegmentationWorker
from fundus_agents.config import MMSEG_WORK_DIR, OPENMMLAB_PYTHON


class DiscCupWorker(BaseSegmentationWorker):
    """Direct inference using the existing mmseg model within openmmlab env."""

    def __init__(self, use_subprocess: bool = True):
        self.use_subprocess = use_subprocess

    def run(self, fundus_img: FundusImage) -> SegmentationMasks:
        if self.use_subprocess:
            return self._run_subprocess(fundus_img)
        return self._run_direct(fundus_img)

    def _run_subprocess(self, fundus_img: FundusImage) -> SegmentationMasks:
        from fundus_agents.segmentation.base import SubprocessSegmentationWorker
        from fundus_agents.config import OPENMMLAB_PYTHON

        script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "script", "run_segmentation.py"
        )

        checkpoint = self._find_checkpoint()

        worker = SubprocessSegmentationWorker(
            python_bin=OPENMMLAB_PYTHON,
            script_path=script,
            model_config="",
            checkpoint=checkpoint,
            target="disc_cup",
        )
        return worker.run(fundus_img)

    def _find_checkpoint(self) -> str:
        pattern = os.path.join(MMSEG_WORK_DIR, "**", "*.pth")
        candidates = glob.glob(pattern, recursive=True)
        segformer = [c for c in candidates if "segformer" in c.lower()]
        if segformer:
            return segformer[0]
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"No checkpoint found in {MMSEG_WORK_DIR}")

    def _run_direct(self, fundus_img: FundusImage) -> SegmentationMasks:
        # Placeholder for direct inference (requires openmmlab environment)
        return SegmentationMasks(lesions=LesionMasks())
