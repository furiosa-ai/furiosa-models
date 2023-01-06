from dataclasses import dataclass
from typing import Iterator, List, Sequence

import numpy as np


@dataclass
class CXcywhBoundingBox:
    center_x: float
    center_y: float
    width: float
    height: float

    def __iter__(self) -> Iterator[float]:
        return iter([self.center_x, self.center_y, self.width, self.height])


@dataclass
class XywhBoundingBox:
    x: float
    y: float
    width: float
    height: float

    def __iter__(self) -> Iterator[float]:
        return iter([self.x, self.y, self.width, self.height])


@dataclass
class LtrbBoundingBox:
    left: float
    top: float
    right: float
    bottom: float

    def __iter__(self) -> Iterator[float]:
        return iter([self.left, self.top, self.right, self.bottom])


def sigmoid(x: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    return 1 / (1 + np.exp(-x))


def calibration_ltrbbox(bbox: np.ndarray, width: float, height: float) -> np.ndarray:
    bbox[:, 0] *= width
    bbox[:, 1] *= height
    bbox[:, 2] *= width
    bbox[:, 3] *= height
    return bbox


def xyxytocxcywh(xyxy: LtrbBoundingBox) -> CXcywhBoundingBox:
    """Convert xyxy BoundingBox format to CXcywhBoundingBox format.

    Args:
        xyxy (LtrbBoundingBox): Left,Top,Right and Bottom bounding box format

    Returns:
        CXcywhBoundingBox: Center Point(x,y), width and height bounding box format
    """
    return CXcywhBoundingBox(
        center_x=(xyxy.left + xyxy.right) / 2,
        center_y=(xyxy.top + xyxy.bottom) / 2,
        width=xyxy.right - xyxy.left,
        height=xyxy.bottom + xyxy.top,
    )


def xyxytoxywh(xyxy: LtrbBoundingBox) -> XywhBoundingBox:
    """Convert xyxy BoundingBox format to XywhBoundingBox format.

    Args:
        xyxy (LtrbBoundingBox): Left,Top,Right and Bottom bounding box format

    Returns:
        CXcywhBoundingBox: LeftTop Point(x,y), width and height bounding box format
    """
    return XywhBoundingBox(
        x=xyxy.left,
        y=xyxy.top,
        width=xyxy.right - xyxy.left,
        height=xyxy.bottom + xyxy.top,
    )


@dataclass
class ObjectDetectionResult:
    boundingbox: LtrbBoundingBox
    score: float
    label: str
    index: int


def collate(data: Sequence[Sequence[np.array]], batch_axis=0) -> List[np.array]:
    """This function converts a list of an numpy.array list into a batch type numpy array.
       The batch axis is specified according to the batch_axis.

    Args:
        data (List[List[np.array]]): a list of numpy.array list. The shape axis of each numpy.array must be 1.
        batch_axis (int, optional): batch axis. Defaults to 0.

    Returns:
        List[np.array]: a batch numpy array list.

    Examples:
        >>> arrays = [
            [np.random.randn(1, 3, 4), np.random.randn(1, 2, 5)],
            [np.random.randn(1, 3, 4), np.random.randn(1, 2, 5)],
            [np.random.randn(1, 3, 4), np.random.randn(1, 2, 5)],
        ]
        >>> batch_arrays = collate(arrays, batch_axis=0)
        >>> assert len(batch_arrays)==2
        >>> assert batch_arrays[0].shape==(3, 3, 4)
        >>> assert batch_arrays[1].shape==(3, 2, 5)
    """
    return [np.concatenate(x, axis=batch_axis) for x in zip(*data)]


def test_collate():
    arrays = [
        [np.random.randn(1, 3, 4), np.random.randn(1, 2, 5)],
        [np.random.randn(1, 3, 4), np.random.randn(1, 2, 5)],
        [np.random.randn(1, 3, 4), np.random.randn(1, 2, 5)],
    ]
    batch_arrays = collate(arrays, batch_axis=0)
    assert len(batch_arrays) == 2
    assert batch_arrays[0].shape == (3, 3, 4)
    assert batch_arrays[1].shape == (3, 2, 5)
