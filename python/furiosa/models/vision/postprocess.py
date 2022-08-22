import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    return 1 / (1 + np.exp(-x))


def calibration_box(bbox, width, height):
    bbox[:, 0] *= width
    bbox[:, 1] *= height
    bbox[:, 2] *= width
    bbox[:, 3] *= height

    bbox[:, 2] -= bbox[:, 0]
    bbox[:, 3] -= bbox[:, 1]
    return bbox
