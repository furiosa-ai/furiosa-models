from abc import ABC
from typing import Any, Dict, List, Sequence, Tuple, Type, Union

import cv2
import numpy as np
import numpy.typing as npt

from .. import native
from ...types import ObjectDetectionModel, Platform, PostProcessor, PreProcessor
from ...vision.postprocess import LtrbBoundingBox, ObjectDetectionResult

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
        The first element is an padded-resized image. The second element is a resized image. The last element is padded sizes, respectively width and height.
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


class YOLOv5PreProcessor(PreProcessor):
    @staticmethod
    def __call__(
        images: Sequence[Union[str, np.ndarray]], color_format: str = "bgr"
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Preprocess input images to a batch of input tensors

        Args:
            images: Color images have (NCHW: Batch, Channel, Height, Width) dimensions.
            color_format:  'bgr' (default) or 'rgb'

        Returns:
            a pre-processed image, scales and padded sizes(width,height) per images.
                The first element is a stacked numpy array containing a batch of images.
                To learn more about the outputs of preprocess (i.e., model inputs),
                please refer to [YOLOv5l Inputs](yolov5l.md#inputs) or
                [YOLOv5m Inputs](yolov5m.md#inputs).

                The second element is a list of dict objects about the original images.
                Each dict object has the following keys. 'scale' key of the returned dict has a rescaled ratio
                per width(=target/width) and height(=target/height), and the 'pad' key has padded width and height
                pixels. Specially, the last dictionary element of returing tuple will be passed to postprocessing
                as a parameter to calculate predicted coordinates on normalized
                coordinates back to an input image coordinator.
        """
        # image format must be chw
        batched_image = []
        batched_proc_params = []
        if isinstance(images, str):
            images = [images]
        for i, img in enumerate(images):
            if type(img) == str:
                img = cv2.imread(img)
                if img is None:
                    raise FileNotFoundError(img)
            img, (sx, sy), (padw, padh) = _resize(img, _INPUT_SIZE)

            if color_format == "bgr":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            assert sx == sy, "yolov5 must be the same rescale for width and height"
            scale = sx
            batched_image.append(img)
            batched_proc_params.append({"scale": scale, "pad": (padw, padh)})

        return np.stack(batched_image, axis=0), batched_proc_params


class YOLOv5PostProcessor(PostProcessor):
    def __init__(self, anchors: npt.ArrayLike, class_names: Sequence[str]):
        """
        native (RustProcessor): A native postprocessor. It has several information to decode: (xyxy, confidence threshold, anchor_grid, stride, number of classes).
        class_names (Sequence[str]): A list of class names.
        """
        self.anchors = anchors
        self.class_names = class_names
        self.anchor_per_layer_count = anchors.shape[1]
        self.native = native.yolov5.RustPostProcessor(anchors, _compute_stride())

    def __call__(
        self,
        model_outputs: Sequence[np.ndarray],
        contexts: Sequence[Dict[str, Any]],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> List[List[ObjectDetectionResult]]:
        """Convert the outputs of this model to a list of bounding boxes, scores and labels

        Args:
            model_outputs: P3/8, P4/16, P5/32 features from yolov5l model.
                To learn more about the outputs of preprocess (i.e., model inputs),
                please refer to [YOLOv5l Outputs](yolov5l.md#outputs) or
                [YOLOv5m Outputs](yolov5m.md#outputs).
            contexts: A configuration for each image generated by the preprocessor.
                For example, it could be the reduction ratio of the image, the actual image width and height.
            conf_thres: Confidence score threshold. The default to 0.25
            iou_thres: IoU threshold value for the NMS processing. The default to 0.45.

        Returns:
            Detected Bounding Box and its score and label represented as `ObjectDetectionResult`.
                The details of `ObjectDetectionResult` can be found below.

        Definition of ObjectDetectionResult and LtrbBoundingBox:
            ::: furiosa.models.vision.postprocess.LtrbBoundingBox
                options:
                    show_source: true
            ::: furiosa.models.vision.postprocess.ObjectDetectionResult
                options:
                    show_source: true
        """

        model_outputs = [
            _reshape_output(f, self.anchor_per_layer_count, len(self.class_names))
            for f in model_outputs
        ]

        batched_boxes = self.native.eval(model_outputs, conf_thres, iou_thres)

        batched_detected_boxes = []
        for boxes, preproc_params in zip(batched_boxes, contexts):
            scale = preproc_params['scale']
            padw, padh = preproc_params['pad']
            detected_boxes = []
            # rescale boxes

            for box in boxes:
                detected_boxes.append(
                    ObjectDetectionResult(
                        index=box.class_id,
                        label=self.class_names[box.class_id],
                        score=box.score,
                        boundingbox=LtrbBoundingBox(
                            left=(box.left - padw) / scale,
                            top=(box.top - padh) / scale,
                            right=(box.right - padw) / scale,
                            bottom=(box.bottom - padh) / scale,
                        ),
                    )
                )
            batched_detected_boxes.append(detected_boxes)

        return batched_detected_boxes


class YOLOv5Base(ObjectDetectionModel, ABC):

    postprocessor_map: Dict[Platform, Type[PostProcessor]] = {
        Platform.PYTHON: YOLOv5PostProcessor,
    }

    @staticmethod
    def get_compiler_config() -> Dict:
        return {
            "without_quantize": {
                "parameters": [
                    {
                        "permute": [0, 2, 3, 1],
                    }
                ]
            }
        }
