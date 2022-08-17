from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    return 1 / (1 + np.exp(-x))


def calibration_ltrbbox(bbox, width, height):
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


def collate_data(
    data: Sequence[Sequence[np.ndarray]], num_element: int, axis: int = 0
) -> List[np.ndarray]:
    """Collate lists of samples into batches.

    Args:
        data (List[List[np.ndarray]]): a list of a list of numpy array
        num_element (int): data dimension
        axis (int, optional): a mini-batch axis. Defaults to 0.

    Returns:
        List[np.ndarray]: a list of mini-batched numpy array.
    """
    batch_data = [[]] * num_element
    for i in range(num_element):
        feat = []
        for b in range(len(data)):
            feat.append(data[b][i])
        batch_data[i] = np.concatenate(feat, axis=axis)
    return batch_data
