"""Myopia agent: rule-based using disc tilt and peripapillary atrophy."""
import numpy as np
from fundus_agents.reasoning.base import BaseDiseaseAgent
from fundus_agents.contracts import FundusImage, SegmentationMasks, DiseaseFinding


class MyopiaAgent(BaseDiseaseAgent):
    def __init__(self):
        super().__init__("M", "近视")

    def diagnose(self, fundus_img: FundusImage,
                 masks: SegmentationMasks) -> DiseaseFinding:
        if not masks.disc.available:
            return self._insufficient_data()

        tilt_ratio = self._compute_tilt(masks.disc.data)
        has_atrophy = masks.macula.available and masks.macula.area > 50

        evidence = []
        myopic = False

        if tilt_ratio > 1.5:
            evidence.append(f"Disc tilt ratio={tilt_ratio:.2f} indicates myopic tilt")
            myopic = True
        else:
            evidence.append(f"Disc tilt ratio={tilt_ratio:.2f} within normal range")

        if has_atrophy:
            evidence.append("Peripapillary atrophy detected")
            myopic = True

        confidence = 0.8 if myopic else 0.85

        return DiseaseFinding(
            disease_code=self.disease_code,
            disease_name=self.disease_name,
            present=myopic,
            confidence=confidence,
            evidence=evidence,
            metrics={"tilt_ratio": round(tilt_ratio, 3)}
        )

    def _compute_tilt(self, disc: np.ndarray) -> float:
        ys, xs = np.where(disc > 0)
        if len(ys) < 10:
            return 1.0
        y_span = ys.max() - ys.min()
        x_span = xs.max() - xs.min()
        if x_span == 0:
            return 1.0
        return max(y_span / x_span, x_span / y_span)
