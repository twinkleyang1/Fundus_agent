"""Message types for inter-agent communication. All immutable where possible."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from PIL import Image


@dataclass
class FundusImage:
    image: Image.Image
    path: str
    metadata: dict = field(default_factory=dict)


@dataclass
class QualityReport:
    passed: bool
    score: float
    issues: list = field(default_factory=list)


@dataclass
class BinaryMask:
    data: Optional[np.ndarray] = None

    @property
    def available(self) -> bool:
        return self.data is not None

    @property
    def area(self) -> float:
        if not self.available:
            return 0.0
        return float(np.sum(self.data > 0))


@dataclass
class LesionMasks:
    hemorrhages: BinaryMask = field(default_factory=BinaryMask)
    hard_exudates: BinaryMask = field(default_factory=BinaryMask)
    soft_exudates: BinaryMask = field(default_factory=BinaryMask)
    microaneurysms: BinaryMask = field(default_factory=BinaryMask)
    drusen: BinaryMask = field(default_factory=BinaryMask)


@dataclass
class SegmentationMasks:
    disc: BinaryMask = field(default_factory=BinaryMask)
    cup: BinaryMask = field(default_factory=BinaryMask)
    vessels: BinaryMask = field(default_factory=BinaryMask)
    macula: BinaryMask = field(default_factory=BinaryMask)
    lesions: LesionMasks = field(default_factory=LesionMasks)


@dataclass
class DiseaseFinding:
    disease_code: str       # "D", "G", "C", "A", "H", "M", "N", "O"
    disease_name: str       # "糖尿病视网膜病变", ...
    present: Optional[bool] # None = insufficient data
    confidence: float       # 0-1
    evidence: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


@dataclass
class StructuredReport:
    image_path: str
    quality: QualityReport
    findings: list  # list of DiseaseFinding
    summary: str
    reasoning_trace: dict = field(default_factory=dict)
