"""Tests for the master orchestrator."""
import numpy as np
from PIL import Image
from fundus_agents.orchestrator import MasterOrchestrator
from fundus_agents.contracts import (
    FundusImage, QualityReport, SegmentationMasks, BinaryMask,
    DiseaseFinding, StructuredReport
)


class StubQualityAgent:
    def assess(self, fundus_img):
        return QualityReport(passed=True, score=0.9, issues=[])


class StubSegmentationWorker:
    def run(self, fundus_img):
        h, w = fundus_img.image.size
        disc = BinaryMask(data=np.zeros((w, h), dtype=np.uint8))
        disc.data[100:200, 100:200] = 1
        cup = BinaryMask(data=np.zeros((w, h), dtype=np.uint8))
        cup.data[130:170, 130:170] = 1
        from fundus_agents.contracts import LesionMasks
        return SegmentationMasks(disc=disc, cup=cup, lesions=LesionMasks())


class StubDiseaseAgent:
    def __init__(self, disease_code, disease_name, finding_present=True):
        self.disease_code = disease_code
        self.disease_name = disease_name
        self.finding_present = finding_present

    def diagnose(self, fundus_img, masks):
        return DiseaseFinding(
            disease_code=self.disease_code,
            disease_name=self.disease_name,
            present=self.finding_present,
            confidence=0.95,
            evidence=["stub evidence"],
            metrics={}
        )


class StubReportAgent:
    def generate(self, image_path, quality, findings, trace):
        return StructuredReport(
            image_path=image_path, quality=quality,
            findings=findings, summary="stub report", reasoning_trace=trace
        )


def test_orchestrator_full_pipeline():
    orch = MasterOrchestrator(
        quality_agent=StubQualityAgent(),
        segmentation_workers={"disc_cup": StubSegmentationWorker()},
        disease_agents=[StubDiseaseAgent("G", "青光眼")],
        report_agent=StubReportAgent(),
    )
    img = Image.new("RGB", (512, 512), color=(128, 80, 60))
    f_img = FundusImage(image=img, path="test.jpg")
    report = orch.run(f_img)
    assert report.quality.passed
    assert len(report.findings) == 1
    assert report.findings[0].disease_code == "G"


def test_orchestrator_rejects_bad_quality():
    class FailQualityAgent:
        def assess(self, fundus_img):
            return QualityReport(passed=False, score=0.2,
                                issues=["too_blurry"])

    orch = MasterOrchestrator(
        quality_agent=FailQualityAgent(),
        segmentation_workers={},
        disease_agents=[],
        report_agent=StubReportAgent(),
    )
    img = Image.new("RGB", (512, 512), color=(0, 0, 0))
    f_img = FundusImage(image=img, path="test.jpg")
    report = orch.run(f_img)
    assert not report.quality.passed
    assert report.summary != ""
