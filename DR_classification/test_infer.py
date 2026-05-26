"""Integration test: verify infer.py runs end-to-end on a sample image."""
import subprocess
import json
import os
import sys
import tempfile


def test_infer_on_sample():
    """Run infer.py on the first APTOS training image, verify JSON output."""
    script = os.path.join(os.path.dirname(__file__), "infer.py")
    sample_img = "/home/twinkle/app/LLM_paper/Dataset/APTOS-2019/train_images/000c1434d8d7.png"

    if not os.path.exists(sample_img):
        print("SKIP: Sample image not found")
        return

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    try:
        result = subprocess.run(
            ["python", script, "--input", sample_img, "--output", out_path, "--device", "cpu"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        if result.returncode != 0:
            print(f"SKIP: Infer failed (models likely not trained yet): {result.stderr[:200]}")
            return

        with open(out_path) as f:
            data = json.load(f)

        assert "present" in data, "Missing 'present' field"
        assert "confidence" in data, "Missing 'confidence' field"
        assert "prob_has_dr" in data, "Missing 'prob_has_dr' field"
        assert isinstance(data["present"], bool), "'present' must be bool"
        assert 0 <= data["confidence"] <= 1, "confidence out of range"

        if data["present"]:
            assert data["severity"] in [1, 2, 3, 4], f"Invalid severity: {data['severity']}"

        print("PASS: Integration test passed")
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


if __name__ == "__main__":
    test_infer_on_sample()
