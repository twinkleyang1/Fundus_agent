"""Tests for glaucoma reasoning agent."""
import numpy as np
from PIL import Image
from fundus_agents.reasoning.glaucoma import GlaucomaAgent
from fundus_agents.contracts import FundusImage, SegmentationMasks, BinaryMask, LesionMasks


def make_disc_cup_masks(disc_area_px=10000, cup_area_px=4900):
    """Create synthetic disc and cup masks. cup/disc = CDR."""
    h, w = 512, 512
    disc = BinaryMask(data=np.zeros((h, w), dtype=np.uint8))
    cup = BinaryMask(data=np.zeros((h, w), dtype=np.uint8))
    disc_r = int(np.sqrt(disc_area_px / np.pi))
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    disc_mask = (y - cy) ** 2 + (x - cx) ** 2 <= disc_r ** 2
    disc.data[disc_mask] = 1
    cup_r = int(np.sqrt(cup_area_px / np.pi))
    cup_mask = (y - cy) ** 2 + (x - cx) ** 2 <= cup_r ** 2
    cup.data[cup_mask] = 1
    masks = SegmentationMasks(disc=disc, cup=cup, lesions=LesionMasks())
    return masks


def test_glaucoma_detected_with_high_cdr():
    agent = GlaucomaAgent()
    masks = make_disc_cup_masks(disc_area_px=10000, cup_area_px=7200)  # CDR=0.72
    img = FundusImage(image=Image.new("RGB", (512, 512)), path="test.jpg")
    finding = agent.diagnose(img, masks)
    assert finding.present is True
    assert finding.metrics["cdr"] > 0.6
    assert len(finding.evidence) > 0


def test_glaucoma_not_detected_with_normal_cdr():
    agent = GlaucomaAgent()
    masks = make_disc_cup_masks(disc_area_px=10000, cup_area_px=3600)  # CDR=0.36
    img = FundusImage(image=Image.new("RGB", (512, 512)), path="test.jpg")
    finding = agent.diagnose(img, masks)
    assert finding.present is False


def test_glaucoma_insufficient_data_without_masks():
    agent = GlaucomaAgent()
    masks = SegmentationMasks(lesions=LesionMasks())
    img = FundusImage(image=Image.new("RGB", (512, 512)), path="test.jpg")
    finding = agent.diagnose(img, masks)
    assert finding.present is None
