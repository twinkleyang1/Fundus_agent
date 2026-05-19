"""Tests for report generation agent."""
from fundus_agents.report.report_agent import ReportAgent
from fundus_agents.contracts import QualityReport, DiseaseFinding


def test_report_generates_summary_for_present_diseases():
    agent = ReportAgent()
    quality = QualityReport(passed=True, score=0.9, issues=[])
    findings = [
        DiseaseFinding(disease_code="G", disease_name="青光眼",
                       present=True, confidence=0.92,
                       evidence=["CDR=0.72"], metrics={"cdr": 0.72}),
        DiseaseFinding(disease_code="D", disease_name="糖尿病视网膜病变",
                       present=False, confidence=0.88,
                       evidence=["No lesions detected"], metrics={}),
    ]
    report = agent.generate("img_001.jpg", quality, findings, {"total_time": 1.5})
    assert "青光眼" in report.summary
    assert "糖尿病视网膜病变" in report.summary
    assert report.reasoning_trace["total_time"] == 1.5
    assert len(report.findings) == 2


def test_report_handles_rejected_images():
    agent = ReportAgent()
    quality = QualityReport(passed=False, score=0.2, issues=["too_blurry"])
    report = agent.generate("img_002.jpg", quality, [], {})
    assert "质量" in report.summary
    assert len(report.findings) == 0


def test_report_handles_null_findings():
    agent = ReportAgent()
    quality = QualityReport(passed=True, score=0.8, issues=[])
    findings = [
        DiseaseFinding(disease_code="A", disease_name="老年性黄斑变性",
                       present=None, confidence=0.0,
                       evidence=["Insufficient data"], metrics={}),
    ]
    report = agent.generate("img_003.jpg", quality, findings, {})
    assert "无法评估" in report.summary
