from abc import ABC, abstractmethod
import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy.typing as npt
from pydantic import BaseConfig, BaseModel, Extra, Field
from typing_extensions import TypeAlias

from furiosa.common.thread import synchronous

from .utils import load_artifacts

# Context type alias
Context: TypeAlias = Any


class PreProcessor(ABC):
    @abstractmethod
    def __call__(self, inputs: Any) -> Tuple[Sequence[npt.ArrayLike], Sequence[Context]]:
        ...


class Platform(IntEnum):
    """Implemented platform"""

    PYTHON = 0
    C = 1
    CPP = 2
    RUST = 3

    def is_native_platform(self):
        return self != self.PYTHON


class PostProcessor(ABC):
    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def __call__(self, model_outputs: Sequence[npt.ArrayLike], contexts: Sequence[Context]) -> Any:
        ...


class ModelTaskType(IntEnum):
    """Model's task type"""

    OBJECT_DETECTION = 0
    IMAGE_CLASSIFICATION = 1


class Config(BaseConfig):
    # Extra fields not permitted
    extra: Extra = Extra.forbid


class Format(str, Enum):
    """Model binary format to represent the binary specified."""

    ONNX = "onnx"
    TFLite = "tflite"


class Publication(BaseModel):
    """Model publication information."""

    __config__ = Config

    authors: Optional[List[str]] = None
    title: Optional[str] = None
    publisher: Optional[str] = None
    date: Optional[datetime.date] = None
    url: Optional[str] = None


class Metadata(BaseModel):
    """Model metadata to understand a model."""

    __config__ = Config

    description: Optional[str] = None
    publication: Optional[Publication] = None


class Tags(BaseModel):
    class Config:
        extra = Extra.allow

    content_type: Optional[str] = None


class ModelTensor(BaseModel):
    name: str
    datatype: str
    shape: List[int]
    tags: Optional[Tags] = None


class Model(ABC, BaseModel):
    """Represent the artifacts and metadata of a neural network model

    Attributes:
        name: a name of this model
        format: the binary format type of model source; e.g., ONNX, tflite
        source: a source binary in ONNX or tflite. It can be used for compiling this model
            with a custom compiler configuration.
        enf: the executable binary for furiosa runtime and NPU
        calib_yaml: the calibration ranges in yaml format for quantization
        version: model version
        inputs: data type and shape of input tensors
        outputs: data type and shape of output tensors
        compiler_config: a pre-defined compiler option
    """

    class Config(BaseConfig):
        extra: Extra = Extra.forbid
        # To allow Session, Processor type
        arbitrary_types_allowed = True
        use_enum_values = True

    name: str
    source: bytes = Field(repr=False)
    format: Format
    enf: Optional[bytes] = Field(repr=False)
    calib_yaml: Optional[str] = Field(repr=False)

    family: Optional[str] = None
    version: Optional[str] = None

    metadata: Optional[Metadata] = None

    inputs: Optional[List[ModelTensor]] = []
    outputs: Optional[List[ModelTensor]] = []

    postprocessor_map: Optional[Dict[Platform, Type[PostProcessor]]] = None

    preprocessor: Optional[PreProcessor] = None
    postprocessor: Optional[PostProcessor] = None

    @staticmethod
    @abstractmethod
    def get_artifact_name() -> str:
        ...

    @classmethod
    @abstractmethod
    def load_aux(
        cls, artifacts: Dict[str, bytes], use_native: Optional[bool] = None, *args, **kwargs
    ):
        ...

    @classmethod
    async def load_async(cls, use_native: Optional[bool] = None, *args, **kwargs) -> 'Model':
        return cls.load_aux(
            await load_artifacts(cls.get_artifact_name()), use_native, *args, **kwargs
        )

    @classmethod
    def load(cls, use_native: Optional[bool] = None, *args, **kwargs) -> 'Model':
        return cls.load_aux(
            synchronous(load_artifacts)(cls.get_artifact_name()), use_native, *args, **kwargs
        )

    def preprocess(self, *args, **kwargs) -> Tuple[Sequence[npt.ArrayLike], Sequence[Context]]:
        assert self.preprocessor
        return self.preprocessor(*args, **kwargs)

    def postprocess(self, *args, **kwargs):
        assert self.postprocessor
        return self.postprocessor(*args, **kwargs)


class ObjectDetectionModel(Model, ABC):
    """Object Detection Model Base Class"""

    task_type: ModelTaskType = ModelTaskType.OBJECT_DETECTION


class ImageClassificationModel(Model, ABC):
    """Image Classification Model Base Class"""

    task_type: ModelTaskType = ModelTaskType.IMAGE_CLASSIFICATION
