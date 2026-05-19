"""Report generation agent: produces structured JSON + natural language summary."""
from fundus_agents.contracts import QualityReport, DiseaseFinding, StructuredReport


class ReportAgent:
    def generate(self, image_path: str, quality: QualityReport,
                 findings: list, reasoning_trace: dict) -> StructuredReport:
        if not quality.passed:
            return StructuredReport(
                image_path=image_path, quality=quality, findings=[],
                summary=self._rejection_summary(quality),
                reasoning_trace=reasoning_trace
            )

        summary_parts = []
        present_diseases = []
        absent_diseases = []
        uncertain_diseases = []

        for f in findings:
            if f.present is True:
                present_diseases.append(f)
            elif f.present is False:
                absent_diseases.append(f)
            else:
                uncertain_diseases.append(f)

        if present_diseases:
            names = "、".join(f.disease_name for f in present_diseases)
            summary_parts.append(f"检测到以下疾病: {names}。")
            for f in present_diseases:
                evidence_text = "; ".join(f.evidence[:2])
                summary_parts.append(
                    f"{f.disease_name}(置信度: {f.confidence:.0%}): {evidence_text}。"
                )
        else:
            summary_parts.append("未检测到明确疾病征象，眼底表现大致正常。")

        if absent_diseases:
            names = "、".join(f.disease_name for f in absent_diseases)
            summary_parts.append(f"以下疾病未检出: {names}。")

        if uncertain_diseases:
            names = "、".join(f.disease_name for f in uncertain_diseases)
            summary_parts.append(f"以下疾病无法评估(数据不足): {names}。")

        return StructuredReport(
            image_path=image_path, quality=quality, findings=findings,
            summary="".join(summary_parts), reasoning_trace=reasoning_trace
        )

    def _rejection_summary(self, quality: QualityReport) -> str:
        issues = "; ".join(quality.issues)
        return (
            f"图像质量不合格(评分: {quality.score:.2f})，无法进行诊断。"
            f"问题: {issues}。"
        )
