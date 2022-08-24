"""Yolov5l Module

Attributes:
    CLASSES (List[str]): a list of class names
"""
__all__ = ['CLASSES', 'preprocess', 'get_anchor_per_layer_count', 'postprocess', 'YoloV5LargeModel']

import pathlib
from typing import Any, Dict, List, Sequence

import numpy as np
import yaml

from furiosa.registry import Model

from . import core as _yolov5

with open(pathlib.Path(__file__).parent / "datasets/yolov5l/cfg.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    _ANCHORS: np.array = np.float32(cfg["anchors"])
    _CLASS_NAMES: List[str] = cfg["class_names"]

_BOX_DECODER = _yolov5.boxdecoder(_CLASS_NAMES, _ANCHORS)


class YoloV5LargeModel(Model):
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


def get_anchor_per_layer_count() -> int:
    """Anchors per layers

    Returns:
        int: the number of anchors for yolov5l
    """
    return _ANCHORS.shape[1]


CLASSES: List[str] = _CLASS_NAMES
preprocess = _yolov5.preprocess


def postprocess(
    batch_feats: Sequence[np.array],
    batch_preproc_param: Sequence[Dict[str, Any]],
) -> List[List[_yolov5.ObjectDetectionResult]]:
    """Yolov5l Postprocess

    Args:
        batch_feats (Sequence[np.array]): P3/8, P4/16, P5/32 features from yolov5l model.
        batch_preproc_param (Dict[str, Any]): configurations from preprocess.

    Returns:
        yolov5.ObjectDetectionResult: Detected Bounding Box and its score and label by Yolov5l.
    """
    return _yolov5.postprocess(
        batch_feats,
        _BOX_DECODER,
        get_anchor_per_layer_count(),
        _CLASS_NAMES,
        batch_preproc_param,
    )
