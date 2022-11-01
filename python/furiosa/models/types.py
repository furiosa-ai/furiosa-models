from abc import ABC, abstractmethod
import datetime
from enum import Enum, IntEnum
from typing import Dict, List, Optional

from pydantic import BaseConfig, BaseModel, Extra, Field

from furiosa.common.thread import synchronous
from furiosa.runtime.session import Session

from .utils import load_artifacts, model_file_name
from .vision.postprocess import PostProcessor


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


class ModelTaskType(IntEnum):
    object_detection = 0
    image_classification = 1


class PreProcessor:
    ...


class Processor:
    preprocessor: Optional[PreProcessor]
    postprocessor: Optional[PostProcessor]


class Model(BaseModel, ABC):
    """Model for a Furiosa SDK."""

    class Config(Config):
        # To allow Session, Processor type
        arbitrary_types_allowed = True

    name: str
    source: bytes = Field(repr=False)
    format: Format
    dfg: Optional[bytes] = Field(repr=False)
    enf: Optional[bytes] = Field(repr=False)

    family: Optional[str] = None
    version: Optional[str] = None

    metadata: Optional[Metadata] = None

    inputs: Optional[List[ModelTensor]] = []
    outputs: Optional[List[ModelTensor]] = []

    compiler_config: Optional[Dict] = None

    # Runtime-related fields
    _session: Optional[Session]
    processor: Optional[Processor]

    @staticmethod
    @abstractmethod
    def get_artifact_name() -> str:
        ...

    @classmethod
    @abstractmethod
    def load_aux(cls, artifacts: Dict[str, bytes], *args, **kwargs):
        ...

    @classmethod
    async def load_async(cls, use_native_post=False, *args, **kwargs) -> 'Model':
        artifact_name = model_file_name(cls.get_artifact_name(), use_native_post)
        return cls.load_aux(await load_artifacts(artifact_name), *args, **kwargs)

    @classmethod
    def load(cls, use_native_post: bool = False, *args, **kwargs) -> 'Model':
        artifact_name = model_file_name(cls.get_artifact_name(), use_native_post)
        return cls.load_aux(synchronous(load_artifacts)(artifact_name), *args, **kwargs)


class ObjectDetectionModel(Model, ABC):
    task_type: ModelTaskType = ModelTaskType.object_detection


class ImageClassificationModel(Model, ABC):
    task_type: ModelTaskType = ModelTaskType.image_classification
