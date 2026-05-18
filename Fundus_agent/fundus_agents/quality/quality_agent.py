"""Quality assessment agent: evaluates if a fundus image is adequate for diagnosis."""
import cv2
import numpy as np
from fundus_agents.contracts import FundusImage, QualityReport
from fundus_agents.config import (
    QUALITY_PASS_THRESHOLD,
    BLUR_LAPLACIAN_DIVISOR,
    BLUR_THRESHOLD,
    FOREGROUND_THRESHOLD,
    EXPECTED_FOV_RATIO,
    MIN_FOV_RATIO,
    MIN_BRIGHTNESS,
    MAX_BRIGHTNESS,
    BRIGHTNESS_PENALTY_SCORE,
)


class QualityAgent:
    def __init__(self, pass_threshold: float = QUALITY_PASS_THRESHOLD):
        self.pass_threshold = pass_threshold

    def assess(self, fundus_img: FundusImage) -> QualityReport:
        try:
            img = np.array(fundus_img.image.convert("RGB"))
            issues = []
            scores = []

            # 1. Blur detection (normalized Laplacian variance)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            lap_var_per_pixel = lap_var / (gray.shape[0] * gray.shape[1])
            blur_score = min(lap_var_per_pixel * BLUR_LAPLACIAN_DIVISOR * 10, 1.0)  # resolution-independent
            scores.append(blur_score)
            if blur_score < BLUR_THRESHOLD:
                issues.append(f"excessive_blur: laplacian_var={lap_var:.1f}")

            # 2. Field-of-view detection (non-black region ratio)
            mask = gray > FOREGROUND_THRESHOLD
            fov_ratio = np.sum(mask) / mask.size
            fov_score = min(fov_ratio / EXPECTED_FOV_RATIO, 1.0)
            scores.append(fov_score)
            if fov_ratio < MIN_FOV_RATIO:
                issues.append(f"incomplete_fov: fov_ratio={fov_ratio:.2f}")

            # 3. Brightness check
            mean_brightness = np.mean(gray[mask]) if np.any(mask) else 0
            brightness_score = 1.0 if MIN_BRIGHTNESS < mean_brightness < MAX_BRIGHTNESS else BRIGHTNESS_PENALTY_SCORE
            scores.append(brightness_score)
            if mean_brightness <= MIN_BRIGHTNESS:
                issues.append(f"too_dark: mean_brightness={mean_brightness:.1f}")
            elif mean_brightness >= MAX_BRIGHTNESS:
                issues.append(f"overexposed: mean_brightness={mean_brightness:.1f}")

            # Aggregate
            overall_score = float(np.mean(scores))
            passed = overall_score >= self.pass_threshold and len(issues) == 0

            return QualityReport(passed=passed, score=overall_score, issues=issues)
        except Exception as e:
            return QualityReport(passed=False, score=0.0, issues=[f"quality_assessment_error: {e}"])
