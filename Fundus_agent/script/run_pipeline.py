#!/usr/bin/env python3
"""End-to-end fundus multi-agent pipeline."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fundus_agents.contracts import FundusImage
from fundus_agents.orchestrator import MasterOrchestrator
from fundus_agents.quality import QualityAgent
from fundus_agents.segmentation import DiscCupWorker, VesselWorker, MaculaWorker, LesionWorker
from fundus_agents.reasoning import (
    GlaucomaAgent, HypertensiveAgent, MyopiaAgent,
    DiabeticRetinopathyAgent, AMDAgent, CataractAgent, OtherAgent,
)
from fundus_agents.report import ReportAgent
from PIL import Image


def build_orchestrator(load_vl: bool = False):
    quality_agent = QualityAgent()

    segmentation_workers = {
        "disc_cup": DiscCupWorker(use_subprocess=False),
        "vessels": VesselWorker(),
        "macula": MaculaWorker(),
        "lesions": LesionWorker(),
    }

    vl_model, vl_processor = None, None
    if load_vl:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from fundus_agents.config import VL_MODELS
        model_path = VL_MODELS["Qwen2.5-VL-7B-Instruct"]
        vl_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, low_cpu_mem_usage=True
        )
        vl_model.eval()

    disease_agents = [
        GlaucomaAgent(),
        HypertensiveAgent(),
        MyopiaAgent(),
        DiabeticRetinopathyAgent(vl_model=vl_model),
        AMDAgent(vl_model=vl_model),
        CataractAgent(model=vl_model, processor=vl_processor),
        OtherAgent(model=vl_model, processor=vl_processor),
    ]

    report_agent = ReportAgent()

    return MasterOrchestrator(
        quality_agent=quality_agent,
        segmentation_workers=segmentation_workers,
        disease_agents=disease_agents,
        report_agent=report_agent,
    )


def main():
    parser = argparse.ArgumentParser(description="Fundus Multi-Agent Pipeline")
    parser.add_argument("--input", help="Single image path")
    parser.add_argument("--input-dir", help="Directory of images")
    parser.add_argument("--output", default="pipeline_results.json")
    parser.add_argument("--load-vl", action="store_true", help="Load VL model")
    parser.add_argument("--max-samples", type=int, default=0, help="Max images (0=all)")
    args = parser.parse_args()

    orch = build_orchestrator(load_vl=args.load_vl)
    results = []

    if args.input:
        img_paths = [args.input]
    elif args.input_dir:
        img_paths = sorted(
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if f.endswith((".jpg", ".png", ".jpeg"))
        )
    else:
        print("Specify --input or --input-dir")
        sys.exit(1)

    if args.max_samples and args.max_samples > 0:
        img_paths = img_paths[:args.max_samples]

    t_start = time.time()
    for i, img_path in enumerate(img_paths):
        t0 = time.time()
        try:
            img = Image.open(img_path).convert("RGB")
            f_img = FundusImage(image=img, path=img_path, metadata={"index": i})
            report = orch.run(f_img)
            result = {
                "index": i, "filename": os.path.basename(img_path),
                "quality_passed": report.quality.passed,
                "quality_score": report.quality.score,
                "findings": [
                    {"code": f.disease_code, "name": f.disease_name,
                     "present": f.present, "confidence": f.confidence,
                     "evidence": f.evidence}
                    for f in report.findings
                ],
                "summary": report.summary,
                "time_s": round(time.time() - t0, 2),
            }
        except Exception as e:
            result = {"index": i, "filename": os.path.basename(img_path),
                      "error": str(e), "time_s": round(time.time() - t0, 2)}
        results.append(result)
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(img_paths)}] {os.path.basename(img_path)} "
                  f"({result.get('time_s', 0):.1f}s)", flush=True)

    output = {
        "total_images": len(img_paths),
        "total_time_s": round(time.time() - t_start, 1),
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
