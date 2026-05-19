"""Tests for myopia agent."""
import numpy as np
from PIL import Image
from fundus_agents.reasoning.myopia import MyopiaAgent
from fundus_agents.contracts import FundusImage, SegmentationMasks, BinaryMask, LesionMasks


def test_detects_myopia_with_tilted_disc():
    agent = MyopiaAgent()
    h, w = 512, 512
    disc = BinaryMask(data=np.zeros((h, w), dtype=np.uint8))
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    # Ellipse: wider horizontally than vertically (tilted appearance)
    disc_mask = ((y - cy) ** 2) / (60 ** 2) + ((x - cx) ** 2) / (120 ** 2) <= 1
    disc.data[disc_mask] = 1
    masks = SegmentationMasks(disc=disc, lesions=LesionMasks())
    img = FundusImage(image=Image.new("RGB", (512, 512)), path="test.jpg")
    finding = agent.diagnose(img, masks)
    assert finding.present is True
    assert "tilt_ratio" in finding.metrics
    assert len(finding.evidence) > 0


def test_normal_round_disc():
    agent = MyopiaAgent()
    h, w = 512, 512
    disc = BinaryMask(data=np.zeros((h, w), dtype=np.uint8))
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    disc_mask = (y - cy) ** 2 + (x - cx) ** 2 <= 70 ** 2
    disc.data[disc_mask] = 1
    masks = SegmentationMasks(disc=disc, lesions=LesionMasks())
    img = FundusImage(image=Image.new("RGB", (512, 512)), path="test.jpg")
    finding = agent.diagnose(img, masks)
    assert finding.present is False


def test_insufficient_data_without_disc():
    agent = MyopiaAgent()
    masks = SegmentationMasks(lesions=LesionMasks())
    img = FundusImage(image=Image.new("RGB", (512, 512)), path="test.jpg")
    finding = agent.diagnose(img, masks)
    assert finding.present is None
