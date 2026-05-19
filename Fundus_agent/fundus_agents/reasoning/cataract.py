"""Cataract agent: uses VL model to assess lens opacity via vessel clarity."""
import re
import torch
from fundus_agents.reasoning.base import BaseDiseaseAgent
from fundus_agents.contracts import FundusImage, SegmentationMasks, DiseaseFinding


class CataractAgent(BaseDiseaseAgent):
    PROMPT = (
        "Assess this fundus image for signs of cataract (lens opacity). "
        "Cataract causes blurring and reduced vessel clarity in the fundus image. "
        "If vessels appear hazy or indistinct despite good image quality, "
        "cataract is likely. Answer with: "
        "Cataract: Yes/No, Confidence: 0-1, Evidence: <brief description>"
    )

    def __init__(self, model=None, processor=None):
        super().__init__("C", "白内障")
        self._model = model
        self._processor = processor

    def diagnose(self, fundus_img: FundusImage,
                 masks: SegmentationMasks) -> DiseaseFinding:
        if self._model is None or self._processor is None:
            return DiseaseFinding(
                disease_code=self.disease_code, disease_name=self.disease_name,
                present=None, confidence=0.0,
                evidence=["VL model not loaded"], metrics={}
            )
        return self._vl_diagnose(fundus_img)

    def _vl_diagnose(self, fundus_img: FundusImage) -> DiseaseFinding:
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": fundus_img.image},
                    {"type": "text", "text": self.PROMPT},
                ],
            }]
            prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=prompt, images=fundus_img.image, return_tensors="pt"
            )
            inputs = {k: v.to(self._model.device) if isinstance(v, torch.Tensor)
                      else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs, max_new_tokens=128, do_sample=False, temperature=0.1
                )
            input_len = inputs["input_ids"].shape[1]
            output = self._processor.decode(
                generated_ids[0][input_len:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            return self._parse_response(output.strip())
        except Exception as e:
            return DiseaseFinding(
                disease_code=self.disease_code, disease_name=self.disease_name,
                present=None, confidence=0.0, evidence=[f"VL error: {e}"], metrics={}
            )

    def _parse_response(self, text: str) -> DiseaseFinding:
        present = "yes" in text.lower()
        # Check "no" appears before "yes" — if so, it's likely "No"
        lower = text.lower()
        yes_pos = lower.find("yes")
        no_pos = lower.find("no")
        if no_pos >= 0 and (yes_pos < 0 or no_pos < yes_pos):
            present = False
        conf_match = re.search(r'confidence[:\s]*([0-9.]+)', text, re.IGNORECASE)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        return DiseaseFinding(
            disease_code=self.disease_code, disease_name=self.disease_name,
            present=present, confidence=min(confidence, 1.0),
            evidence=[text], metrics={}
        )
