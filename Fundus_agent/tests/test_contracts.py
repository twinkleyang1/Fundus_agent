"""Test contract types are constructable and behave correctly."""
import numpy as np
from fundus_agents.contracts import (
    FundusImage, QualityReport, BinaryMask, LesionMasks,
    SegmentationMasks, DiseaseFinding, StructuredReport
)
from PIL import Image


def test_binary_mask_available_when_data_present():
    mask = BinaryMask(data=np.ones((10, 10)))
    assert mask.available
    assert mask.area == 100.0


def test_binary_mask_unavailable_when_none():
    mask = BinaryMask()
    assert not mask.available
    assert mask.area == 0.0


def test_quality_report_defaults():
    r = QualityReport(passed=True, score=0.85)
    assert r.passed
    assert r.issues == []


def test_segmentation_masks_all_unavailable_by_default():
    masks = SegmentationMasks()
    assert not masks.disc.available
    assert not masks.cup.available
    assert not masks.vessels.available


def test_disease_finding_holds_evidence():
    f = DiseaseFinding(
        disease_code="G", disease_name="青光眼",
        present=True, confidence=0.9,
        evidence=["CDR=0.72 > 0.6"], metrics={"cdr": 0.72}
    )
    assert f.present is True
    assert len(f.evidence) == 1


def test_structured_report_construction():
    q = QualityReport(passed=True, score=0.9)
    r = StructuredReport(
        image_path="test.jpg", quality=q, findings=[], summary="正常眼底"
    )
    assert r.quality.passed
    assert r.summary == "正常眼底"
