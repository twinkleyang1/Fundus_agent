"""Diabetic Retinopathy agent: two-stage EfficientNet-B3 classifier."""
import subprocess
import tempfile
import json
import os
from fundus_agents.reasoning.base import BaseDiseaseAgent
from fundus_agents.contracts import FundusImage, SegmentationMasks, DiseaseFinding
from fundus_agents.config import DR_CLASSIFY_PYTHON, DR_CLASSIFY_SCRIPT


class DiabeticRetinopathyAgent(BaseDiseaseAgent):
    def __init__(self, device="cuda:0"):
        super().__init__("D", "糖尿病视网膜病变")
        self.device = device

    def diagnose(self, fundus_img: FundusImage,
                 masks: SegmentationMasks = None) -> DiseaseFinding:
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img_path = f.name
            fundus_img.image.save(img_path)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name

        try:
            cmd = [
                DR_CLASSIFY_PYTHON, DR_CLASSIFY_SCRIPT,
                "--input", img_path,
                "--output", out_path,
                "--device", self.device,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                raise RuntimeError(f"DR classification failed: {result.stderr}")

            with open(out_path) as f:
                data = json.load(f)

            if data["present"]:
                return DiseaseFinding(
                    disease_code=self.disease_code,
                    disease_name=self.disease_name,
                    present=True,
                    confidence=data["confidence"],
                    evidence=[f"DR severity grade {data['severity']}/4",
                              f"P(has_DR)={data['prob_has_dr']:.4f}"],
                    metrics={
                        "severity": data["severity"],
                        "severity_probs": data.get("severity_probs", []),
                        "prob_has_dr": data["prob_has_dr"],
                    },
                )
            else:
                return DiseaseFinding(
                    disease_code=self.disease_code,
                    disease_name=self.disease_name,
                    present=False,
                    confidence=data["confidence"],
                    evidence=[f"No DR detected, P(has_DR)={data['prob_has_dr']:.4f}"],
                    metrics={"prob_has_dr": data["prob_has_dr"]},
                )
        except subprocess.TimeoutExpired:
            return self._insufficient_data()
        finally:
            for p in [img_path, out_path]:
                if os.path.exists(p):
                    os.unlink(p)
