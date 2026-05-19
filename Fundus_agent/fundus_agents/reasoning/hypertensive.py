"""Hypertensive retinopathy agent: rule-based using AVR and arteriolar signs."""
import numpy as np
from fundus_agents.reasoning.base import BaseDiseaseAgent
from fundus_agents.contracts import FundusImage, SegmentationMasks, DiseaseFinding
from fundus_agents.config import HYPERTENSIVE_AVR_THRESHOLD


class HypertensiveAgent(BaseDiseaseAgent):
    def __init__(self, avr_threshold: float = HYPERTENSIVE_AVR_THRESHOLD):
        super().__init__("H", "高血压视网膜病变")
        self.avr_threshold = avr_threshold

    def diagnose(self, fundus_img: FundusImage,
                 masks: SegmentationMasks) -> DiseaseFinding:
        if not masks.vessels.available:
            return self._insufficient_data()

        avr = self._compute_avr(masks.vessels.data)
        has_hemorrhages = masks.lesions.hemorrhages.available and \
            masks.lesions.hemorrhages.area > 10

        evidence = []
        hypertensive = False

        if avr < self.avr_threshold:
            evidence.append(f"AVR={avr:.2f} below threshold {self.avr_threshold:.2f}")
            hypertensive = True
        else:
            evidence.append(f"AVR={avr:.2f} normal")

        if has_hemorrhages and avr < 0.8:
            evidence.append("Hemorrhages present with arteriolar narrowing")
            hypertensive = True

        confidence = 0.85 if hypertensive else 0.8

        return DiseaseFinding(
            disease_code=self.disease_code,
            disease_name=self.disease_name,
            present=hypertensive,
            confidence=confidence,
            evidence=evidence,
            metrics={"avr": round(avr, 3)}
        )

    def _compute_avr(self, vessel_mask: np.ndarray) -> float:
        binary = (vessel_mask > 0).astype(np.uint8)
        labeled = self._connected_components(binary)
        n_labels = labeled.max()
        if n_labels < 2:
            return 1.0
        areas = [int(np.sum(labeled == i)) for i in range(1, n_labels + 1)]
        areas.sort()
        mid = len(areas) // 2
        artery_mean = float(np.mean(areas[:mid])) if mid > 0 else float(areas[0])
        vein_mean = float(np.mean(areas[mid:])) if mid < len(areas) else float(areas[-1])
        if vein_mean == 0:
            return 1.0
        return artery_mean / vein_mean

    @staticmethod
    def _connected_components(binary: np.ndarray) -> np.ndarray:
        """Two-pass connected component labeling (4-connectivity)."""
        h, w = binary.shape
        label_arr = np.zeros((h, w), dtype=np.int32)
        next_label = 1
        # Label equivalence table: parent[i] = i
        parents = []

        # First pass
        for y in range(h):
            for x in range(w):
                if binary[y, x] == 0:
                    continue
                neighbors = []
                if y > 0 and label_arr[y - 1, x] > 0:
                    neighbors.append(label_arr[y - 1, x])
                if x > 0 and label_arr[y, x - 1] > 0:
                    neighbors.append(label_arr[y, x - 1])
                if not neighbors:
                    label_arr[y, x] = next_label
                    parents.append(next_label)
                    next_label += 1
                else:
                    min_label = min(neighbors)
                    label_arr[y, x] = min_label
                    for n in neighbors:
                        if n != min_label:
                            # Union: make all point to min
                            root_n = parents[n - 1]
                            while root_n != parents[root_n - 1]:
                                root_n = parents[root_n - 1]
                            root_min = parents[min_label - 1]
                            while root_min != parents[root_min - 1]:
                                root_min = parents[root_min - 1]
                            # Union
                            parents[root_n - 1] = root_min

        # Second pass: flatten labels
        for y in range(h):
            for x in range(w):
                if label_arr[y, x] > 0:
                    lbl = label_arr[y, x]
                    root = parents[lbl - 1]
                    while root != parents[root - 1]:
                        root = parents[root - 1]
                    label_arr[y, x] = root

        # Renumber consecutive
        unique = np.unique(label_arr)
        mapping = {old: new for new, old in enumerate(unique)}
        out = np.zeros_like(label_arr)
        for y in range(h):
            for x in range(w):
                out[y, x] = mapping[label_arr[y, x]]
        return out
