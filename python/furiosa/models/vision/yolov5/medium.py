"""Yolov5m Module

Attributes:
    CLASSES (List[str]): a list of class names
"""
import pathlib
from typing import Dict, List

import numpy as np
import yaml

from furiosa.registry.model import Format, Metadata, Publication

from ...utils import EXT_DFG, EXT_ENF, EXT_ONNX
from .core import YOLOv5Base, YOLOv5PostProcessor, YOLOv5PreProcessor

with open(pathlib.Path(__file__).parent / "datasets/yolov5m/cfg.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    _ANCHORS: np.array = np.float32(cfg["anchors"])  # aspect ratio: (width, height)
    CLASSES: List[str] = cfg["class_names"]

__all__ = ['CLASSES', 'YOLOv5m']


class YOLOv5m(YOLOv5Base):
    """YOLOv5 Medium model"""

    classes = CLASSES

    @staticmethod
    def get_artifact_name():
        return "yolov5m_int8"

    @classmethod
    def load_aux(cls, artifacts: Dict[str, bytes], use_native: bool = False, *args, **kwargs):
        if use_native:
            raise NotImplementedError("No native implementation for YOLOv5")
        return cls(
            name="YOLOv5Medium",
            source=artifacts[EXT_ONNX],
            dfg=artifacts[EXT_DFG],
            enf=artifacts[EXT_ENF],
            format=Format.ONNX,
            family="YOLOv5",
            version="v5",
            compiler_config=cls.get_compiler_config(),
            metadata=Metadata(
                description="YOLOv5 medium model",
                publication=Publication(url="https://github.com/ultralytics/yolov5"),
            ),
            preprocessor=YOLOv5PreProcessor(),
            postprocessor=YOLOv5PostProcessor(_ANCHORS, CLASSES),
        )
