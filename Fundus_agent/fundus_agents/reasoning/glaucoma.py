"""Glaucoma agent: rule-based diagnosis using CDR and ISNT rule."""
import numpy as np
from fundus_agents.reasoning.base import BaseDiseaseAgent
from fundus_agents.contracts import FundusImage, SegmentationMasks, DiseaseFinding
from fundus_agents.config import GLAUCOMA_CDR_THRESHOLD


class GlaucomaAgent(BaseDiseaseAgent):
    def __init__(self, cdr_threshold: float = GLAUCOMA_CDR_THRESHOLD):
        super().__init__("G", "青光眼")
        self.cdr_threshold = cdr_threshold

    def diagnose(self, fundus_img: FundusImage,
                 masks: SegmentationMasks) -> DiseaseFinding:
        if not masks.disc.available or not masks.cup.available:
            return self._insufficient_data()

        cdr = self._compute_cdr(masks.disc.data, masks.cup.data)
        isnt_violated = self._check_isnt(masks.disc.data, masks.cup.data)

        evidence = []
        glaucomatous = False

        if cdr > self.cdr_threshold:
            evidence.append(f"CDR={cdr:.2f} exceeds threshold {self.cdr_threshold}")
            glaucomatous = True
        else:
            evidence.append(f"CDR={cdr:.2f} within normal range")

        if isnt_violated:
            evidence.append("ISNT rule violated: inferior rim thinner than nasal")
            glaucomatous = True

        confidence = min(0.95, (cdr - 0.4) / 0.5) if glaucomatous else 0.9

        return DiseaseFinding(
            disease_code=self.disease_code,
            disease_name=self.disease_name,
            present=glaucomatous,
            confidence=round(confidence, 3),
            evidence=evidence,
            metrics={"cdr": round(cdr, 3), "isnt_violated": isnt_violated}
        )

    def _compute_cdr(self, disc: np.ndarray, cup: np.ndarray) -> float:
        disc_area = np.sum(disc > 0)
        cup_area = np.sum(cup > 0)
        if disc_area == 0:
            return 0.0
        return float(cup_area / disc_area)

    def _check_isnt(self, disc: np.ndarray, cup: np.ndarray) -> bool:
        rim = (disc > 0).astype(np.uint8) - (cup > 0).astype(np.uint8)
        rim = np.clip(rim, 0, 1)
        h, w = disc.shape
        cy, cx = h // 2, w // 2

        # For vertical quadrants (inferior/superior) measure mean horizontal rim
        # width per row. For horizontal quadrants (nasal/temporal) measure mean
        # vertical rim width per column.  This avoids geometric bias from the
        # rectangular quadrant shapes when the disc is centrally located.
        inferior = float(np.mean(np.sum(rim[cy:, :] > 0, axis=1)))
        superior = float(np.mean(np.sum(rim[:cy, :] > 0, axis=1)))
        nasal = float(np.mean(np.sum(rim[:, cx:] > 0, axis=0)))
        temporal = float(np.mean(np.sum(rim[:, :cx] > 0, axis=0)))

        return inferior < nasal
