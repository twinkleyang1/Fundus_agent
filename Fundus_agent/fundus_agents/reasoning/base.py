"""Base class for all disease reasoning agents."""
import abc
from fundus_agents.contracts import FundusImage, SegmentationMasks, DiseaseFinding


class BaseDiseaseAgent(abc.ABC):
    def __init__(self, disease_code: str, disease_name: str):
        self.disease_code = disease_code
        self.disease_name = disease_name

    @abc.abstractmethod
    def diagnose(self, fundus_img: FundusImage,
                 masks: SegmentationMasks) -> DiseaseFinding:
        ...

    def _insufficient_data(self) -> DiseaseFinding:
        return DiseaseFinding(
            disease_code=self.disease_code,
            disease_name=self.disease_name,
            present=None, confidence=0.0,
            evidence=["Insufficient segmentation data for diagnosis"],
            metrics={}
        )
