"""Tests for quality assessment agent."""
import numpy as np
from PIL import Image
from fundus_agents.quality.quality_agent import QualityAgent
from fundus_agents.contracts import FundusImage


def make_test_image(size=(512, 512), mode="RGB"):
    """Create a synthetic fundus-like image with a dark circular FOV."""
    img = Image.new(mode, size, color=(0, 0, 0))
    arr = np.array(img, dtype=np.float32)
    h, w = size
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 2 - 20
    y, x = np.ogrid[:h, :w]
    mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
    arr[mask] = [128, 80, 60]  # fundus-like reddish
    # Add noise and subtle texture to avoid false blur detection
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 15, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255)
    # Add some vessel-like lines
    for _ in range(30):
        x0 = rng.randint(cx - radius + 10, cx + radius - 10)
        y0 = rng.randint(cy - radius + 10, cy + radius - 10)
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.randint(20, 80)
        for t in range(length):
            px = int(x0 + t * np.cos(angle))
            py = int(y0 + t * np.sin(angle))
            if 0 <= px < w and 0 <= py < h and mask[py, px]:
                arr[py, px] = np.clip(arr[py, px] - 30, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def test_quality_agent_accepts_good_image():
    agent = QualityAgent()
    img = make_test_image()
    f_img = FundusImage(image=img, path="test.jpg")
    report = agent.assess(f_img)
    assert report.passed
    assert report.score >= 0.5


def test_quality_agent_rejects_small_fov():
    agent = QualityAgent()
    # Create image with a very small "fundus" circle
    img = Image.new("RGB", (512, 512), color=(0, 0, 0))
    arr = np.array(img, dtype=np.float32)
    cy, cx = 256, 256
    radius = 50  # very small FOV
    y, x = np.ogrid[:512, :512]
    mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
    arr[mask] = [128, 80, 60]
    img = Image.fromarray(arr.astype(np.uint8))
    f_img = FundusImage(image=img, path="test.jpg")
    report = agent.assess(f_img)
    assert not report.passed
    assert len(report.issues) > 0


def test_quality_agent_reports_issues():
    agent = QualityAgent()
    img = make_test_image()
    f_img = FundusImage(image=img, path="test.jpg")
    report = agent.assess(f_img)
    assert isinstance(report.score, float)
    assert 0.0 <= report.score <= 1.0
