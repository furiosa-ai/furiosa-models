from .. import native
from typing import Sequence
import numpy as np


class BoxDecoder:
    def __init__(self, num_classes: int, anchors: np.ndarray, stride: np.ndarray) -> None:
        """Yolov5 BoxDecoder creator

        Args:
            nc (int): the number of classes
            anchors (np.ndarray): anchors: layers x the number of anchors x 2(aspect ratio: width, height)
            stride (np.ndarray): stride per layers: image_height / feature_height for each layer.
        """
        self.native = native.yolov5.RustPostProcessor(anchors, num_classes, stride)

    def __call__(self, feats: np.ndarray, conf_thres: float) -> np.ndarray:
        """_summary_

        Args:
            feats (np.ndarray): model's output. This feat will expected to be in (center_x, center_y, width, height, confidence score, C0..Cn)
            conf_thres (float): confidence threshold value

        Returns:
            np.ndarray: batch x detected box x (left,top,right,bottom,conf,index of classes)
        """

        assert all(isinstance(feat, np.ndarray) for feat in feats)

        boxes = self.native.eval(feats, conf_thres)
        return [boxes]
