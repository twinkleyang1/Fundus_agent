#!/usr/bin/env python3
"""Evaluate pipeline output against ODIR-5K ground truth."""
import json
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fundus_agents.config import CLASS_NAMES, CSV_PATH
import pandas as pd


def load_gt():
    df = pd.read_csv(CSV_PATH)
    gt = {}
    for _, row in df.iterrows():
        fname = row.get("filename", "")
        if not fname or pd.isna(fname):
            continue
        labels = [code for code in CLASS_NAMES if int(row.get(code, 0)) == 1]
        gt[fname] = labels if labels else ["N"]
    return gt


def pred_labels_from_findings(findings):
    return [f["code"] for f in findings if f["present"] is True] or ["N"]


def calculate_metrics(results, gt):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for r in results:
        fname = r["filename"]
        true_set = set(gt.get(fname, ["N"]))
        pred_set = set(pred_labels_from_findings(r.get("findings", [])))
        for code in CLASS_NAMES:
            in_true = code in true_set
            in_pred = code in pred_set
            if in_true and in_pred:
                tp[code] += 1
            elif in_pred and not in_true:
                fp[code] += 1
            elif in_true and not in_pred:
                fn[code] += 1
    per_class = {}
    for code, name in CLASS_NAMES.items():
        t, f_p, f_n = tp[code], fp[code], fn[code]
        p = t / (t + f_p) if (t + f_p) > 0 else 0.0
        r = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[code] = {"name": name, "tp": t, "fp": f_p, "fn": f_n,
                           "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
    n = len(results)
    exact = sum(1 for r in results if set(gt.get(r["filename"], ["N"])) ==
                set(pred_labels_from_findings(r.get("findings", []))))
    total_correct = sum(
        sum(1 for code in CLASS_NAMES if (code in set(gt.get(r["filename"], ["N"]))) ==
            (code in set(pred_labels_from_findings(r.get("findings", [])))))
        for r in results)
    total_labels = n * len(CLASS_NAMES)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)
    return {
        "exact_match_accuracy": round(exact / n, 4) if n else 0,
        "hamming_accuracy": round(total_correct / total_labels, 4) if total_labels else 0,
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class, "total_samples": n,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_pipeline.py <pipeline_results.json>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        data = json.load(f)
    gt = load_gt()
    metrics = calculate_metrics(data["results"], gt)
    print(f"\nSamples: {metrics['total_samples']}")
    print(f"Exact Match Acc: {metrics['exact_match_accuracy']:.4f}")
    print(f"Hamming Acc:     {metrics['hamming_accuracy']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"\nPer-class F1:")
    for code in sorted(CLASS_NAMES):
        pc = metrics["per_class"][code]
        print(f"  {code} ({pc['name']}): F1={pc['f1']:.4f} P={pc['precision']:.4f} R={pc['recall']:.4f}")
    out_path = sys.argv[1].replace(".json", "_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
