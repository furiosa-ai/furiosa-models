from box_decode import cbox_decode
import numpy as np


class BoxDecoderC(object):
    def __init__(self, nc: int, anchors: np.ndarray, stride: np.ndarray) -> None:
        """Yolov5 BoxDecoder creator

        Args:
            nc (int): the number of classes
            anchors (np.ndarray): anchors: layers x the number of anchors x 2(aspect ratio: width, height)
            stride (np.ndarray): stride per layers: image_height / feature_height for each layer.
        """
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor(xyxy,conf)
        self.anchors = anchors
        self.nl = anchors.shape[0]  # number of detection layers
        self.na = anchors.shape[1]  # number of anchors
        self.stride = stride

    def __call__(self, feats: np.ndarray, conf_thres: float) -> np.ndarray:
        """_summary_

        Args:
            feats (np.ndarray): model's output. This feat will expected to be in (center_x, center_y, width, height, confidence score, C0..Cn)
            conf_thres (float): confidence threshold value

        Returns:
            np.ndarray: batch x detected box x (left,top,right,bottom,conf,index of classes)
        """

        assert all(isinstance(feat, np.ndarray) for feat in feats)

        out_boxes_batched = cbox_decode(self.anchors, self.stride, conf_thres, feats)

        return out_boxes_batched
