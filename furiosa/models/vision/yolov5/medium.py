"""YOLOv5m Module

Attributes:
    CLASSES (List[str]): a list of class names
"""
import pathlib
from typing import List, Union

import numpy as np
import yaml

from ..._utils import validate_postprocessor_type
from ...types import Metadata, Platform, Publication
from .core import YOLOv5Base

with open(pathlib.Path(__file__).parent / "datasets/yolov5m/cfg.yaml", "r") as f:
    configuration = yaml.safe_load(f)
    _ANCHORS: np.array = np.float32(configuration["anchors"])  # aspect ratio: (width, height)
    CLASSES: List[str] = configuration["class_names"]

__all__ = ['CLASSES', 'YOLOv5m']


class YOLOv5m(YOLOv5Base):
    """YOLOv5 Medium model"""

    classes: List[str] = CLASSES

    def __init__(self, *, postprocessor_type: Union[str, Platform] = Platform.RUST):
        postprocessor_type = Platform(postprocessor_type)
        validate_postprocessor_type(postprocessor_type, self.postprocessor_map.keys())
        super().__init__(
            name="YOLOv5Medium",
            metadata=Metadata(
                description="YOLOv5 medium model",
                publication=Publication(url="https://github.com/ultralytics/yolov5"),
            ),
            postprocessor=self.postprocessor_map[postprocessor_type](_ANCHORS, CLASSES),
        )

        self._artifact_name = "yolov5m"
