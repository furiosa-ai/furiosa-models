from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class CXcywhBoundingBox:
    center_x: float
    center_y: float
    width: float
    height: float

    def __iter__(self) -> List[float]:
        return iter([self.center_x, self.center_y, self.width, self.height])


@dataclass
class XywhBoundingBox:
    x: float
    y: float
    width: float
    height: float

    def __iter__(self) -> List[float]:
        return iter([self.x, self.y, self.width, self.height])


@dataclass
class LtrbBoundingBox:
    left: float
    top: float
    right: float
    bottom: float

    def __iter__(self) -> List[float]:
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
    return CXcywhBoundingBox(
        center_x=(xyxy.left + xyxy.right) / 2,
        center_y=(xyxy.top + xyxy.bottom) / 2,
        width=xyxy.right - xyxy.left,
        height=xyxy.bottom + xyxy.top,
    )


def xyxytoxywh(xyxy: LtrbBoundingBox) -> XywhBoundingBox:
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


from ..furiosa_models_native import nms_internal_ops_fast_rust


def _nms_internal_ops_fast_rust(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, eps: float = 1e-5
) -> List[int]:
    return nms_internal_ops_fast_rust(boxes, scores, iou_threshold, eps)


# Malisiewicz et al.
def _nms_internal_ops_fast_py(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, eps: float = 1e-5
) -> List[int]:
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick_indices: List[int] = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by score of the bounding box
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(scores)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick_indices.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        # compute the ratio of overlap
        inter_union_area = w * h
        overlap_area = inter_union_area / (
            area[idxs[last]] + area[idxs[:last]] - inter_union_area + eps
        )
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap_area > iou_threshold)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick_indices


def nms_internal_ops_fast(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, eps: float = 1e-5
) -> List[int]:
    """Non-maximum Suppression(NMS)
       Select the boxes out of many overlapped regions with scores based on some criteria(IoU Threshold Value).
       The criterion for the overlapping regions is if an intersect between two regions is greater than the iou threshold value.

    Args:
        boxes (np.ndarray): A list of candiate boxes corresponding confidence score.
            They have to be in (left, top, right, bottom) format with left <= right and top <= bottom.
        scores (np.ndarray): scores for each candidate boxes. It's dimension has N x 1.
        iou_threshold (float): discards all overlapping boxes with the overlap > iou_threshold.
        eps (float, optional): preventing during overlapping between boxes from NaN. Defaults to 1e-5.

    Returns:
        List[int]: the list of indices of the element filtered by NMS, sorted in decreasing order of scores.
    """
    return _nms_internal_ops_fast_py(boxes, scores, iou_threshold, eps)
    # for PyO3 Competible Testing Version
    # return _nms_internal_ops_fast_rust(boxes, scores, iou_threshold, eps)
