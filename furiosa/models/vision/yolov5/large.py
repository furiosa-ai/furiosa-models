"""YOLOv5l Module

Attributes:
    CLASSES (List[str]): a list of class names
"""
from functools import cached_property
import pathlib
from typing import List, Union

import numpy as np
from pydantic import computed_field
import yaml

from ...types import Metadata, Platform, PostProcessor, Publication
from .core import YOLOv5Base

with open(pathlib.Path(__file__).parent / "datasets/yolov5l/cfg.yaml", "r") as f:
    configuration = yaml.safe_load(f)
    _ANCHORS: np.array = np.float32(configuration["anchors"])  # aspect ratio: (width, height)
    CLASSES: List[str] = configuration["class_names"]

__all__ = ['CLASSES', 'YOLOv5l']


class YOLOv5l(YOLOv5Base):
    """YOLOv5 Large model"""

    classes: List[str] = CLASSES

    def __init__(self, postprocessor_type: Union[str, Platform] = Platform.RUST):
        super().__init__(
            name="YOLOv5Large",
            metadata=Metadata(
                description="YOLOv5 large model",
                publication=Publication(url="https://github.com/ultralytics/yolov5"),
            ),
            postprocessor_type=postprocessor_type,
        )

        self._artifact_name = "yolov5l"

    @computed_field(repr=False)
    @cached_property
    def postprocessor(self) -> PostProcessor:
        return self.postprocessor_map[self.postprocessor_type](_ANCHORS, CLASSES)
