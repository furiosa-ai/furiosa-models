import numpy as np
from typing import Tuple, List
import torch
import torchvision
from dataclasses import dataclass
from .utils.transforms import letterbox
import cv2
from .box_decode.box_decoder import BoxDecoderC

@dataclass
class LtrbBoundingBox:
    left: float
    top: float
    right: float
    bottom: float

@dataclass
class DetectecdResult:
    boundingbox: LtrbBoundingBox
    score: float
    predicted_class: str
    index: int

def _nms(prediction, iou_thres=0.45, class_agnostic=True):
    # Checks
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

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

def _resize(img, model_input_size):
    w, h = model_input_size
    return letterbox(img, (h, w), auto=False)

def _reshape_output(feat, anchor_per_layer_count: int, num_classes: int):
    return np.ascontiguousarray(
        feat.reshape(
            feat.shape[0],
            anchor_per_layer_count,
            num_classes + 5,
            feat.shape[2],
            feat.shape[3],
        ).transpose(0, 1, 3, 4, 2)
    )

def preprocess(img: np.array, model_input_size: Tuple[int,int], input_color_format: str):
    # image format must be chw
    img, (sx, sy), (padw, padh) = _resize(img, model_input_size)

    if input_color_format == "bgr":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert sx == sy
    scale = sx

    return np.stack([img]), [(scale, (padw, padh))]

def postprocess(feats, box_decoder: BoxDecoderC, anchor_per_layer_count: int, class_names: List[str], preproc_params, iou_thres) -> List[DetectecdResult]:
    feats_batched = [feat.numpy() for feat in feats]

    boxes_batched = []

    for i, (scale, (padw, padh)) in enumerate(preproc_params):
        feats = [f[i : i + 1] for f in feats_batched]  # noqa: E203
        feats = [_reshape_output(f, anchor_per_layer_count, len(class_names)) for f in feats]
        boxes = box_decoder(feats)
        boxes = _nms(boxes, iou_thres)[0]

        # rescale boxes
        boxes[:, [0, 2]] = (1 / scale) * (boxes[:, [0, 2]] - padw)
        boxes[:, [1, 3]] = (1 / scale) * (boxes[:, [1, 3]] - padh)

        for box in boxes:
            boxes_batched.append( DetectecdResult(
                    index=int(box[5]),
                    predicted_class=class_names[int(box[5])],
                    score=box[4],
                    boundingbox=LtrbBoundingBox(
                        left = box[0],
                        top = box[1],
                        right = box[2],
                        bottom = box[3]
                    )
                )
            )

    return boxes_batched
