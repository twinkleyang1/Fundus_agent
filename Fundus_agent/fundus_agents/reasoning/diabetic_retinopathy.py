"""Diabetic Retinopathy agent: classifier on lesion features, falls back to VL."""
import numpy as np
from fundus_agents.reasoning.base import BaseDiseaseAgent
from fundus_agents.contracts import FundusImage, SegmentationMasks, DiseaseFinding


class DiabeticRetinopathyAgent(BaseDiseaseAgent):
    def __init__(self, vl_model=None):
        super().__init__("D", "糖尿病视网膜病变")
        self.vl_model = vl_model

    def diagnose(self, fundus_img: FundusImage,
                 masks: SegmentationMasks) -> DiseaseFinding:
        # If lesion masks available, use feature-based classifier
        if masks.lesions.hemorrhages.available or \
           masks.lesions.hard_exudates.available:
            return self._classifier_diagnose(masks)

        # Fallback to VL model
        if self.vl_model is not None:
            return self._vl_diagnose(fundus_img)
        return self._insufficient_data()

    def _classifier_diagnose(self, masks: SegmentationMasks) -> DiseaseFinding:
        heme_area = masks.lesions.hemorrhages.area
        exudate_area = masks.lesions.hard_exudates.area
        ma_count = self._count_lesions(masks.lesions.microaneurysms)

        features = {
            "heme_area": heme_area,
            "exudate_area": exudate_area,
            "ma_count": ma_count,
        }

        present = heme_area > 20 or exudate_area > 20 or ma_count > 5
        confidence = 0.75 if present else 0.85

        return DiseaseFinding(
            disease_code=self.disease_code,
            disease_name=self.disease_name,
            present=present, confidence=confidence,
            evidence=[f"Lesion features: heme_area={heme_area:.1f}, "
                      f"exudate_area={exudate_area:.1f}, ma_count={ma_count}"],
            metrics=features
        )

    def _vl_diagnose(self, fundus_img: FundusImage) -> DiseaseFinding:
        return DiseaseFinding(
            disease_code=self.disease_code, disease_name=self.disease_name,
            present=None, confidence=0.0,
            evidence=["VL model not configured for DR"], metrics={}
        )

    def _count_lesions(self, mask) -> int:
        if mask is None or mask.data is None:
            return 0
        # Simple connected component count
        data = mask.data if hasattr(mask, 'data') else mask
        if data is None:
            return 0
        labeled = np.zeros_like(data, dtype=int)
        label_id = 0
        h, w = data.shape
        for y in range(h):
            for x in range(w):
                if data[y, x] > 0 and labeled[y, x] == 0:
                    label_id += 1
                    # Flood fill
                    stack = [(y, x)]
                    while stack:
                        py, px = stack.pop()
                        if 0 <= py < h and 0 <= px < w and data[py, px] > 0 and labeled[py, px] == 0:
                            labeled[py, px] = label_id
                            stack.extend([(py-1, px), (py+1, px), (py, px-1), (py, px+1)])
        return label_id
