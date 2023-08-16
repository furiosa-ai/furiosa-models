from abc import ABC, abstractmethod
import datetime
from enum import Enum
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal

import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer
from typing_extensions import TypeAlias
import yaml

from ._utils import EXT_CALIB_YAML, EXT_ONNX, resolve_model_source, resolve_source

# Context type alias
Context: TypeAlias = Any


class Platform(str, Enum):
    """Implemented platform"""

    PYTHON = "PYTHON"
    C = "C"
    CPP = "CPP"
    RUST = "RUST"

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value == value.upper():
                return member


class ModelTaskType(str, Enum):
    """Model's task type"""

    OBJECT_DETECTION = "OBJECT_DETECTION"
    IMAGE_CLASSIFICATION = "IMAGE_CLASSIFICATION"


class Format(str, Enum):
    """Model binary format to represent the binary specified."""

    ONNX = "ONNX"
    TFLite = "TFLITE"


class PreProcessor(ABC):
    @abstractmethod
    def __call__(self, inputs: Any) -> Tuple[Sequence[npt.ArrayLike], Sequence[Context]]:
        ...


class PostProcessor(ABC):
    @abstractmethod
    def __call__(self, model_outputs: Sequence[npt.ArrayLike], contexts: Sequence[Context]) -> Any:
        ...


class RustPostProcessor(PostProcessor):
    platform = Platform.RUST


class PythonPostProcessor(PostProcessor):
    platform = Platform.PYTHON


class Publication(BaseModel, extra='forbid'):
    """Model publication information."""

    authors: Optional[List[str]] = None
    title: Optional[str] = None
    publisher: Optional[str] = None
    date: Optional[datetime.date] = None
    url: Optional[str] = None


class Metadata(BaseModel, extra='forbid'):
    """Model metadata to understand a model."""

    description: Optional[str] = None
    publication: Optional[Publication] = None


class Tags(BaseModel, extra='forbid'):
    """Model tags to understand a model."""

    content_type: Optional[str] = None


class Model(ABC, BaseModel):
    """Represent the artifacts and metadata of a neural network model

    Attributes:
        name: a name of this model
        task_type: the task type of this model
        format: the binary format type of model origin; e.g., ONNX, tflite
        family: the model family
        version: the model version
        metadata: the model metadata
        tags: the model tags
        origin: an origin f32 binary in ONNX or tflite. It can be used for compiling this model
            with or without quantization and proper compiler configuration
        tensor_name_to_range: the calibration ranges of each tensor in origin
        preprocessor: a preprocessor to preprocess input tensors
        postprocessor: a postprocessor to postprocess output tensors
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    task_type: ModelTaskType
    format: Format
    family: Optional[str] = None
    version: Optional[str] = None
    metadata: Optional[Metadata] = None
    tags: Optional[Tags] = None

    _artifact_name: str

    preprocessor: PreProcessor = Field(..., repr=False, exclude=True)
    postprocessor: PostProcessor = Field(..., repr=False, exclude=True)

    def preprocess(self, *args, **kwargs) -> Tuple[Sequence[npt.ArrayLike], Sequence[Context]]:
        """preprocess input tensors
        """
        return self.preprocessor(*args, **kwargs)

    def postprocess(self, *args, **kwargs):
        """postprocess output tensors
        """
        return self.postprocessor(*args, **kwargs)

    @computed_field(repr=False)
    @cached_property
    def origin(self) -> bytes:
        return resolve_source(self._artifact_name, EXT_ONNX)

    @computed_field(repr=False)
    @cached_property
    def tensor_name_to_range(self) -> Dict[str, List[float]]:
        calib_yaml = resolve_source(self._artifact_name, EXT_CALIB_YAML)
        return yaml.full_load(calib_yaml)

    def model_source(self, num_pe: Literal[1, 2] = 2) -> bytes:
        """the executable binary for furiosa runtime and NPU. It can be
            directly fed to `furiosa.runtime.create_runner`. If model binary is not compiled yet,
            it will be quantized & compiled automatically if possible

        Args:
            num_pe: number of PE to be used.
        """
        if num_pe not in (1, 2):
            raise ValueError(f"Invalid num_pe: {num_pe}")

        # TODO: Add in-memory cached value(like cached_property), currently uses disk-cached value
        return resolve_model_source(self._artifact_name, num_pe=num_pe)

    def resolve_all(self):
        """resolve all non-cached properties(origin, tensor_name_to_range, model_sources)
        """
        _ = self.origin, self.tensor_name_to_range
        for num_pe in (1, 2):
            _ = self.model_source(num_pe=num_pe)

    @field_serializer('format')
    def serialize_format(self, format: Format):
        return format.value

    @field_serializer('task_type')
    def serialize_task_type(self, task_type: ModelTaskType):
        return task_type.value


class ObjectDetectionModel(Model, ABC):
    """Object Detection Model Base Class"""

    def __init__(self, *args, **kwargs):
        super().__init__(task_type=ModelTaskType.OBJECT_DETECTION, *args, **kwargs)


class ImageClassificationModel(Model, ABC):
    """Image Classification Model Base Class"""

    def __init__(self, *args, **kwargs):
        super().__init__(task_type=ModelTaskType.IMAGE_CLASSIFICATION, *args, **kwargs)
