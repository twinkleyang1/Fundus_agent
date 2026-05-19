"""Master orchestrator: deterministic 4-phase pipeline."""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fundus_agents.contracts import FundusImage, StructuredReport, DiseaseFinding


class MasterOrchestrator:
    def __init__(self, quality_agent, segmentation_workers,
                 disease_agents, report_agent):
        self.quality_agent = quality_agent
        self.segmentation_workers = segmentation_workers
        self.disease_agents = disease_agents
        self.report_agent = report_agent

    def run(self, fundus_img: FundusImage) -> StructuredReport:
        trace = {}
        t_start = time.time()

        # Phase 1: Quality
        quality = self.quality_agent.assess(fundus_img)
        trace["quality"] = {"score": quality.score, "issues": quality.issues,
                            "time": round(time.time() - t_start, 3)}
        if not quality.passed:
            return self.report_agent.generate(
                fundus_img.path, quality, [],
                {"status": "rejected_by_quality", "trace": trace}
            )

        # Phase 2: Parallel segmentation
        masks = self._run_segmentation(fundus_img, trace)

        # Phase 3: Parallel disease reasoning
        findings = self._run_disease_reasoning(fundus_img, masks, trace)

        # Phase 4: Report
        report = self.report_agent.generate(
            fundus_img.path, quality, findings, trace
        )
        trace["total_time"] = round(time.time() - t_start, 3)
        report.reasoning_trace = trace
        return report

    def _run_segmentation(self, fundus_img, trace):
        from fundus_agents.contracts import SegmentationMasks
        results = {}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(self.segmentation_workers)) as ex:
            futures = {
                ex.submit(worker.run, fundus_img): name
                for name, worker in self.segmentation_workers.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    trace[f"seg_error_{name}"] = str(e)
                    results[name] = SegmentationMasks()
        trace["segmentation_time"] = round(time.time() - t0, 3)
        return self._merge_masks(results)

    def _merge_masks(self, results):
        from fundus_agents.contracts import SegmentationMasks, LesionMasks
        merged = SegmentationMasks(lesions=LesionMasks())
        for name, masks in results.items():
            if masks.disc.available:
                merged.disc = masks.disc
            if masks.cup.available:
                merged.cup = masks.cup
            if masks.vessels.available:
                merged.vessels = masks.vessels
            if masks.macula.available:
                merged.macula = masks.macula
            # Merge lesion sub-masks
            for attr in ["hemorrhages", "hard_exudates", "soft_exudates",
                         "microaneurysms", "drusen"]:
                src = getattr(masks.lesions, attr, None)
                if src and src.available:
                    setattr(merged.lesions, attr, src)
        return merged

    def _run_disease_reasoning(self, fundus_img, masks, trace):
        findings = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(self.disease_agents)) as ex:
            futures = {
                ex.submit(agent.diagnose, fundus_img, masks): agent
                for agent in self.disease_agents
            }
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    findings.append(future.result())
                except Exception as e:
                    trace[f"reasoning_error_{agent.disease_code}"] = str(e)
                    findings.append(DiseaseFinding(
                        disease_code=agent.disease_code,
                        disease_name=agent.disease_name,
                        present=None, confidence=0.0,
                        evidence=[f"ERROR: {e}"], metrics={}
                    ))
        trace["reasoning_time"] = round(time.time() - t0, 3)
        return findings
