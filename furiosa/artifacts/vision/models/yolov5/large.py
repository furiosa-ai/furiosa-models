import pathlib

import cv2
import numpy as np
import torch
import torchvision
import yaml
from furiosa.registry import Model

from .box_decode.box_decoder import BoxDecoderC
from .utils.transforms import letterbox

INPUT_SIZE = (640, 640)
IOU_THRES = 0.45
OUTPUT_SHAPES = [(1, 45, 80, 80), (1, 45, 40, 40), (1, 45, 20, 20)]

with open(pathlib.Path(__file__).parent / "datasets/yolov5l/cfg.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    ANCHORS = np.float32(cfg["anchors"])
    CLASS_NAMES = cfg["class_names"]


def _compute_stride():
    img_h = INPUT_SIZE[1]
    feat_h = np.float32([shape[2] for shape in OUTPUT_SHAPES])
    strides = img_h / feat_h
    return strides


BOX_DECODER = BoxDecoderC(
    nc=len(CLASS_NAMES),
    anchors=ANCHORS,
    stride=_compute_stride(),
    conf_thres=0.25,
)


def _nms(prediction, iou_thres=0.45, class_agnostic=True):
    # Checks
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    min_wh, max_wh = (  # noqa: F841
        2,
        7680,
    )  # (pixels) minimum and maximum box width and height

    output = []
    for x in prediction:  # image index, image inference
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # Batched NMS
        if not class_agnostic:
            c = x[:, 5:6] * max_wh  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        else:
            boxes, scores = x[:, :4], x[:, 4]

        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        output.append(x[i].numpy())

    return output


class YoloV5LargeModel(Model):
    def get_class_names(self):
        return CLASS_NAMES

    def get_class_count(self):
        return len(CLASS_NAMES)

    def get_output_feat_count(self):
        return ANCHORS.shape[0]

    def get_anchor_per_layer_count(self):
        return ANCHORS.shape[1]

    def _resize(self, img):
        w, h = INPUT_SIZE
        return letterbox(img, (h, w), auto=False)

    def _cvt_color(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _reshape_output(self, feat):
        return np.ascontiguousarray(
            feat.reshape(
                feat.shape[0],
                self.get_anchor_per_layer_count(),
                self.get_class_count() + 5,
                feat.shape[2],
                feat.shape[3],
            ).transpose(0, 1, 3, 4, 2)
        )

    def preprocess(self, img):
        img, (sx, sy), (padw, padh) = self._resize(img)

        img = self._cvt_color(img)

        assert sx == sy
        scale = sx

        return np.stack([img]), [(scale, (padw, padh))]

    def postprocess(self, feats, preproc_params):
        feats_batched = [feat.numpy() for feat in feats]

        boxes_batched = []

        for i, (scale, (padw, padh)) in enumerate(preproc_params):
            feats = [f[i : i + 1] for f in feats_batched]  # noqa: E203
            feats = [self._reshape_output(f) for f in feats]
            boxes = BOX_DECODER(feats)
            boxes = _nms(boxes, IOU_THRES)[0]

            # rescale boxes
            boxes[:, [0, 2]] = (1 / scale) * (boxes[:, [0, 2]] - padw)
            boxes[:, [1, 3]] = (1 / scale) * (boxes[:, [1, 3]] - padh)

            boxes_batched.append(boxes)

        return boxes_batched
