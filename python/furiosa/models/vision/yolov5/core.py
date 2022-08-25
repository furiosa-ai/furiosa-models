from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from furiosa.models.vision.postprocess import (
    LtrbBoundingBox,
    ObjectDetectionResult,
    nms_internal_ops_fast,
)

from .box_decode.box_decoder import BoxDecoderC

_INPUT_SIZE = (640, 640)
_GRID_CELL_OUTPUT_SHAPES = [(80, 80), (40, 40), (20, 20)]


def _letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """Letterboxing is a method for fitting an image within
       a constrained space dictated by the model requirements.
       If some axis is smaller than new_shape, it will be filled with color.
    Args:
        im (np.ndarray): a numpy image. Its shape must be Channel x Height x Width.
        new_shape (Tuple[int, int], optional): Targeted Image size. Defaults to (640, 640).
        color (Tuple[int, int, int], optional): Padding Color Value. Defaults to (114, 114, 114).
        auto (bool, optional): If True, calculate padding width and height along to stride  Default to True.
        scaleFill (bool, optional): If True and auto is False, stretch an give image to target shape without any padding. Default to False.
        scaleup (bool, optional): If True, only scale down. Default to True.
        stride (int, optional): an output size of an image must be divied by a stride it is dependent on a model. This is only valid for auto is True. Default to 32.

    Returns:
        Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        The first element is an padded-resized image. The second element is a resized image. The last element is padded sizes, respectivly width and height.
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def _nms(
    prediction: Sequence[np.ndarray], iou_thres: float = 0.45, class_agnostic: bool = True
) -> List[np.ndarray]:
    """Internal Non-Maxima Suppression for BoxDecode

    Args:
        prediction (Sequence[np.ndarray]): Batch x 6(left,top,right,bottom,confidence,class index)
        iou_thres (float, optional): IoU Threshold. Defaults to 0.45.
        class_agnostic (bool, optional): Class Agnostic. Defaults to True.

    Returns:
        List[np.ndarray]: Detected Boxes
    """
    # Checks
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    min_wh, max_wh = (  # noqa: F841
        2,
        7680,
    )  # (pixels) minimum and maximum box width and height

    batched_output = []
    for x in prediction:
        # Batched NMS
        if not class_agnostic:
            class_index = x[:, 5:6] * max_wh  # classe index * max_wh
            # c = 0, offset 0 + (xyxy)
            # c = 1, +max_wh + (xyxy)
            # c = 2, +2*max_wh + (xyxy)
            # ...
            # boxes of different classes can never overlap each other.
            boxes, scores = x[:, :4] + class_index, x[:, 4]  # boxes (offset by class), scores
        else:
            boxes, scores = x[:, :4], x[:, 4]
        i = nms_internal_ops_fast(boxes[:, :4], scores, iou_thres)  # NMS
        batched_output.append(x[i, :])

    return batched_output


def _resize(img, model_input_size):
    w, h = model_input_size
    return _letterbox(img, (h, w), auto=False)


def _reshape_output(feat: np.ndarray, anchor_per_layer_count: int, num_classes: int):
    return np.ascontiguousarray(
        feat.reshape(
            feat.shape[0],  # batch
            anchor_per_layer_count,
            num_classes + 5,  # boundingbox(4) + objectness score + classes score of that object
            feat.shape[2],  # the number of width grid
            feat.shape[3],  # the number of height grid
        ).transpose(0, 1, 3, 4, 2)
    )


def _compute_stride() -> np.ndarray:
    img_h = _INPUT_SIZE[1]
    feat_h = np.float32([shape[0] for shape in _GRID_CELL_OUTPUT_SHAPES])  # a size of grid cell
    strides = img_h / feat_h
    return strides


def boxdecoder(class_names: Sequence[str], anchors: np.ndarray) -> BoxDecoderC:
    """Calculate the left, top, right, and bottom of the box from the coordinates
       predicted by a model.

    Args:
        class_names (Sequence[str]): A list of class string.
        anchors (np.ndarray): a aspect ratio array of anchors.

    Returns:
        BoxDecoderC Callable Instance
    """
    return BoxDecoderC(
        nc=len(class_names),
        anchors=anchors,
        stride=_compute_stride(),
    )


def preprocess(
    img_list: Sequence[np.ndarray], input_color_format: str
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Yolov5 preprocess

    Args:
        img (Sequence[np.ndarray]): Color images have (Batch, Channel, Height, Width) dimensions.
        input_color_format (str): a color format: rgb(Red,Green,Blue), bgr(Blue,Green,Red)

    Returns:
        Tuple[np.ndarray, List[Dict[str, Any]]]: a pre-processed image, scales and padded sizes(width,height) per images.
        The first element is a preprocessing image, and a second element is a dictionary object to be used for postprocess.
        'scale' key of the returned dict has a rescaled ratio per width(=target/width) and height(=target/height),
        and the 'pad' key has padded width and height pixels. Specially, the last dictionary element of returing
        tuple will be passed to postprocessing as a parameter to calculate predicted coordinates on normalized coordinates back to an input image cooridnates.
    """
    # image format must be chw
    batched_image = []
    batched_proc_params = []
    for i, img in enumerate(img_list):
        img, (sx, sy), (padw, padh) = _resize(img, _INPUT_SIZE)

        if input_color_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert sx == sy, "yolov5 must be the same rescale for width and height"
        scale = sx
        batched_image.append(img)
        batched_proc_params.append({"scale": scale, "pad": (padw, padh)})

    return np.stack(batched_image, axis=0), batched_proc_params


def postprocess(
    batch_feats: Sequence[np.ndarray],
    box_decoder: BoxDecoderC,
    anchor_per_layer_count: int,
    class_names: Sequence[str],
    batch_preproc_params: Sequence[Dict[str, Any]],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> List[List[ObjectDetectionResult]]:
    """Yolov5 PostProcess.

    Args:
        batch_feats (Sequence[np.ndarray]): Model numpy version outputs. This numpy array expects to be in Batch x Features(the number anchors x (5+the number of clsss)) x N x N.
        box_decoder (BoxDecoderC): A box decoder. It has several informations to decode: (xyxy, confidence threshold, anchor_grid, stride, number of classes).
        anchor_per_layer_count (int): The number of anchors per layers.
        class_names (Sequence[str]): A list of class names.
        batch_preproc_params (Dict[str, Any]): The components manipulated by the preprocessor to be used for information recovery: image scaling ratio, padding size in width and height.
        conf_threshold (float, optional): Confidence score threshold. The default to 0.25
        iou_thres (float, optional): IoU threshold value for the NMS processing. The default to 0.45.

    Returns:
        List[List[ObjectDetectionResult]]: Detected bounding boxes(class index, class name, score, 2D box coordinates(left,top,right,bottom)).
            This ObjectDetectionResult inherits from dataclass, so that it could be converted to Tuple (by astuple funciton in the dataclass package).
    """

    batch_feats = [
        _reshape_output(f, anchor_per_layer_count, len(class_names)) for f in batch_feats
    ]
    batched_boxes = box_decoder(batch_feats, conf_thres)
    batched_boxes = _nms(batched_boxes, iou_thres)

    batched_detected_boxes = []
    for boxes, preproc_params in zip(batched_boxes, batch_preproc_params):
        scale = preproc_params['scale']
        padw, padh = preproc_params['pad']
        detected_boxes = []
        # rescale boxes
        boxes[:, [0, 2]] = (1 / scale) * (boxes[:, [0, 2]] - padw)
        boxes[:, [1, 3]] = (1 / scale) * (boxes[:, [1, 3]] - padh)

        for box in boxes:
            detected_boxes.append(
                ObjectDetectionResult(
                    index=int(box[5]),
                    label=class_names[int(box[5])],
                    score=box[4],
                    boundingbox=LtrbBoundingBox(
                        left=box[0], top=box[1], right=box[2], bottom=box[3]
                    ),
                )
            )
        batched_detected_boxes.append(detected_boxes)

    return batched_detected_boxes
