"""Macula segmentation worker — stub until model is trained."""
from fundus_agents.contracts import FundusImage, SegmentationMasks, LesionMasks
from fundus_agents.segmentation.base import BaseSegmentationWorker


class MaculaWorker(BaseSegmentationWorker):
    def run(self, fundus_img: FundusImage) -> SegmentationMasks:
        return SegmentationMasks(lesions=LesionMasks())
