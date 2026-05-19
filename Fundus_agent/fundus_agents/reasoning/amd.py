"""AMD agent: classifier on drusen/atrophy features, falls back to VL."""
from fundus_agents.reasoning.base import BaseDiseaseAgent
from fundus_agents.contracts import FundusImage, SegmentationMasks, DiseaseFinding


class AMDAgent(BaseDiseaseAgent):
    def __init__(self, vl_model=None):
        super().__init__("A", "老年性黄斑变性")
        self.vl_model = vl_model

    def diagnose(self, fundus_img: FundusImage,
                 masks: SegmentationMasks) -> DiseaseFinding:
        if masks.lesions.drusen.available:
            return self._classifier_diagnose(masks)
        if self.vl_model is not None:
            return self._vl_diagnose(fundus_img)
        return self._insufficient_data()

    def _classifier_diagnose(self, masks: SegmentationMasks) -> DiseaseFinding:
        drusen_area = masks.lesions.drusen.area
        present = drusen_area > 10
        return DiseaseFinding(
            disease_code=self.disease_code, disease_name=self.disease_name,
            present=present, confidence=0.75,
            evidence=[f"Drusen area: {drusen_area:.1f} pixels"],
            metrics={"drusen_area": drusen_area}
        )

    def _vl_diagnose(self, fundus_img: FundusImage) -> DiseaseFinding:
        return DiseaseFinding(
            disease_code=self.disease_code, disease_name=self.disease_name,
            present=None, confidence=0.0,
            evidence=["VL model not configured for AMD"], metrics={}
        )
