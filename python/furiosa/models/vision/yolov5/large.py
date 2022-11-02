"""YOLOv5L

YOLOv5 is the latest object detection model developed by [ultralytics]().

Example:
    a =

Attributes:
    CLASSES (List[str]): a list of class names
"""
__all__ = ['CLASSES', 'preprocess', '_get_anchor_per_layer_count', 'postprocess', 'YOLOv5l']

import pathlib
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml

from furiosa.registry import Format, Metadata, Publication

from . import core as _yolov5
from ...model import ObjectDetectionModel
from ...utils import EXT_DFG, EXT_ENF, EXT_ONNX
from ..postprocess import ObjectDetectionResult
from .core import preprocess

with open(pathlib.Path(__file__).parent / "datasets/yolov5l/cfg.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    _ANCHORS: np.array = np.float32(cfg["anchors"])
    _CLASS_NAMES: List[str] = cfg["class_names"]

_BOX_DECODER = _yolov5.boxdecoder(_CLASS_NAMES, _ANCHORS)


class YOLOv5l(ObjectDetectionModel):
    """YOLOv5 Large model"""

    @classmethod
    def get_artifact_name(cls):
        return "yolov5l_int8"

    @classmethod
    def load_aux(cls, artifacts: Dict[str, bytes], *args, **kwargs):
        return cls(
            name="YoloV5Large",
            source=artifacts[EXT_ONNX],
            dfg=artifacts[EXT_DFG],
            enf=artifacts[EXT_ENF],
            format=Format.ONNX,
            family="YOLOv5",
            version="v5",
            metadata=Metadata(
                description="YOLOv5 large model",
                publication=Publication(url="https://github.com/ultralytics/yolov5"),
            ),
            *args,
            **kwargs,
        )

    def compile_config(self, model_input_format="hwc"):
        return {
            "without_quantize": {
                "parameters": [
                    {
                        "input_min": 0.0,
                        "input_max": 1.0,
                        "permute": [0, 2, 3, 1]
                        if model_input_format == "hwc"
                        else [0, 1, 2, 3],  # bchw
                    }
                ]
            }
        }


def _get_anchor_per_layer_count() -> int:
    """Anchors per layers

    Returns:
        int: the number of anchors for yolov5l
    """
    return _ANCHORS.shape[1]


CLASSES: List[str] = _CLASS_NAMES
"""Class names"""


def preprocess(
    images: Sequence[np.ndarray], input_color_format: str
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Preprocess a batch of images in numpy

    Args:
        images (Sequence[np.ndarray]): Color images have (NCHW: Batch, Channel, Height, Width) dimensions.
        input_color_format (str): 'rgb' (Red, Green, Blue) or 'bgr' (Blue, Green, Red).

    Returns:
        a pre-processed image, scales and padded sizes(width,height) per images.
            The first element is a preprocessing image, and a second element is a dictionary object to be used for postprocess.
            'scale' key of the returned dict has a rescaled ratio per width(=target/width) and height(=target/height),
            and the 'pad' key has padded width and height pixels. Specially, the last dictionary element of returing
            tuple will be passed to postprocessing as a parameter to calculate predicted coordinates on normalized
            coordinates back to an input image coordinator.
    """
    return _yolov5.preprocess(images, input_color_format)


def postprocess(
    model_outputs: Sequence[np.array],
    context: Sequence[Dict[str, Any]],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> List[List[ObjectDetectionResult]]:
    """Convert the outputs of this model to a list of bounding boxes, scores and labels

    Args:
        model_outputs (Sequence[np.array]): P3/8, P4/16, P5/32 features from yolov5l model.
        context (Dict[str, Any]): A configuration for each image generated by the preprocessor.
            For example, it could be the reduction ratio of the image, the actual image width and height.
        conf_threshold (float, optional): Confidence score threshold. The default to 0.25
        iou_thres (float, optional): IoU threshold value for the NMS processing. The default to 0.45.

    Returns:
        Detected Bounding Box and its score and label represented as `ObjectDetectionResult`.
            To learn more about `ObjectDetectionResult`, 'Definition of ObjectDetectionResult' can be found below.

    Definition of ObjectDetectionResult:
        ::: furiosa.models.vision.postprocess.LtrbBoundingBox
            options:
                show_root_heading: false
                show_source: true
        ::: furiosa.models.vision.postprocess.ObjectDetectionResult
            options:
                show_root_heading: false
                show_source: true
    """
    return _yolov5.postprocess(
        model_outputs,
        _BOX_DECODER,
        _get_anchor_per_layer_count(),
        _CLASS_NAMES,
        context,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )
