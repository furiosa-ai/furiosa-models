from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy
import numpy as np

from furiosa.registry import Model


def sigmoid(x: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    return 1 / (1 + np.exp(-x))


def calibration_ltrbbox(bbox: np.ndarray, width: float, height: float) -> np.ndarray:
    bbox[:, 0] *= width
    bbox[:, 1] *= height
    bbox[:, 2] *= width
    bbox[:, 3] *= height
    return bbox


@dataclass
class LtrbBoundingBox:
    left: float
    top: float
    right: float
    bottom: float

    def __iter__(self) -> List[float]:
        return iter([self.left, self.top, self.right, self.bottom])


@dataclass
class ObjectDetectionResult:
    boundingbox: LtrbBoundingBox
    score: float
    label: str
    index: int


class PostProcessor(ABC):
    @abstractmethod
    def eval(self, inputs: Sequence[numpy.ndarray], *args: Any, **kwargs: Any):
        pass
