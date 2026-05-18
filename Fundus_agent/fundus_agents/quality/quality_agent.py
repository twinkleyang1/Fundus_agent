"""Quality assessment agent: evaluates if a fundus image is adequate for diagnosis."""
import cv2
import numpy as np
from fundus_agents.contracts import FundusImage, QualityReport
from fundus_agents.config import QUALITY_PASS_THRESHOLD


class QualityAgent:
    def __init__(self, pass_threshold: float = QUALITY_PASS_THRESHOLD):
        self.pass_threshold = pass_threshold

    def assess(self, fundus_img: FundusImage) -> QualityReport:
        img = np.array(fundus_img.image.convert("RGB"))
        issues = []
        scores = []

        # 1. Blur detection (Laplacian variance)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(lap_var / 500.0, 1.0)
        scores.append(blur_score)
        if blur_score < 0.4:
            issues.append(f"excessive_blur: laplacian_var={lap_var:.1f}")

        # 2. Field-of-view detection (non-black region ratio)
        mask = gray > 10
        fov_ratio = np.sum(mask) / mask.size
        fov_score = min(fov_ratio / 0.6, 1.0)
        scores.append(fov_score)
        if fov_ratio < 0.3:
            issues.append(f"incomplete_fov: fov_ratio={fov_ratio:.2f}")

        # 3. Brightness check
        mean_brightness = np.mean(gray[mask]) if np.any(mask) else 0
        brightness_score = 1.0 if 30 < mean_brightness < 220 else 0.3
        scores.append(brightness_score)
        if mean_brightness <= 30:
            issues.append(f"too_dark: mean_brightness={mean_brightness:.1f}")
        elif mean_brightness >= 220:
            issues.append(f"overexposed: mean_brightness={mean_brightness:.1f}")

        # Aggregate
        overall_score = float(np.mean(scores))
        passed = overall_score >= self.pass_threshold and len(issues) == 0

        return QualityReport(passed=passed, score=overall_score, issues=issues)
