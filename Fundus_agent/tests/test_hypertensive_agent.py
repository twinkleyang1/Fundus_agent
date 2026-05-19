"""Tests for hypertensive retinopathy agent."""
import numpy as np
from PIL import Image
from fundus_agents.reasoning.hypertensive import HypertensiveAgent
from fundus_agents.contracts import FundusImage, SegmentationMasks, BinaryMask, LesionMasks


def test_detects_hypertensive_with_narrow_arterioles():
    agent = HypertensiveAgent()
    h, w = 512, 512
    vessels = BinaryMask(data=np.zeros((h, w), dtype=np.uint8))
    # Simulate: narrow arteriole + wide venule
    vessels.data[100:400, 200:210] = 1   # arteriole (narrow, 10px)
    vessels.data[100:400, 240:260] = 1   # venule (wide, 20px)
    hemorrhages = BinaryMask(data=np.zeros((h, w), dtype=np.uint8))
    hemorrhages.data[300:320, 250:270] = 1

    masks = SegmentationMasks(
        vessels=vessels,
        lesions=LesionMasks(hemorrhages=hemorrhages)
    )
    img = FundusImage(image=Image.new("RGB", (512, 512)), path="test.jpg")
    finding = agent.diagnose(img, masks)
    assert finding.present is True
    assert finding.metrics["avr"] < 0.67
    assert len(finding.evidence) > 0


def test_normal_when_vessels_not_available():
    agent = HypertensiveAgent()
    masks = SegmentationMasks(lesions=LesionMasks())
    img = FundusImage(image=Image.new("RGB", (512, 512)), path="test.jpg")
    finding = agent.diagnose(img, masks)
    assert finding.present is None
