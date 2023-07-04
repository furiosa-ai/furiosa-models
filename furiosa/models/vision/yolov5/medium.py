"""YOLOv5m Module

Attributes:
    CLASSES (List[str]): a list of class names
"""
import pathlib
from typing import Dict, List

import numpy as np
import yaml

from ...types import Format, Metadata, Publication
from ...utils import EXT_CALIB_YAML, EXT_ENF, EXT_ONNX
from .core import YOLOv5Base, YOLOv5PostProcessor, YOLOv5PreProcessor

with open(pathlib.Path(__file__).parent / "datasets/yolov5m/cfg.yaml", "r") as f:
    configuration = yaml.safe_load(f)
    _ANCHORS: np.array = np.float32(configuration["anchors"])  # aspect ratio: (width, height)
    CLASSES: List[str] = configuration["class_names"]

__all__ = ['CLASSES', 'YOLOv5m']


class YOLOv5m(YOLOv5Base):
    """YOLOv5 Medium model"""

    classes = CLASSES

    @staticmethod
    def get_artifact_name():
        return "yolov5m"

    @classmethod
    def load(cls, use_native: bool = False):
        if use_native:
            raise NotImplementedError("No native implementation for YOLOv5")
        return cls(
            name="YOLOv5Medium",
            format=Format.ONNX,
            family="YOLOv5",
            version="v5",
            metadata=Metadata(
                description="YOLOv5 medium model",
                publication=Publication(url="https://github.com/ultralytics/yolov5"),
            ),
            preprocessor=YOLOv5PreProcessor(),
            postprocessor=YOLOv5PostProcessor(_ANCHORS, CLASSES),
        )
