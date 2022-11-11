from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, Dict, Optional, Sequence, Tuple, Type

import numpy.typing as npt
from pydantic import BaseConfig, Extra
from typing_extensions import TypeAlias

from furiosa.common.thread import synchronous
from furiosa.registry.model import Model as RegistryModel

from .utils import load_artifacts, model_file_name

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


class Model(RegistryModel, ABC):
    class Config(BaseConfig):
        extra: Extra = Extra.forbid
        # To allow Session, Processor type
        arbitrary_types_allowed = True
        use_enum_values = True

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
        artifact_name = model_file_name(cls.get_artifact_name(), use_native)
        return cls.load_aux(await load_artifacts(artifact_name), use_native, *args, **kwargs)

    @classmethod
    def load(cls, use_native: Optional[bool] = None, *args, **kwargs) -> 'Model':
        artifact_name = model_file_name(cls.get_artifact_name(), use_native)
        return cls.load_aux(synchronous(load_artifacts)(artifact_name), use_native, *args, **kwargs)

    def preprocess(self, *args, **kwargs) -> Tuple[Sequence[npt.ArrayLike], Sequence[Context]]:
        assert self.preprocessor is not None
        return self.preprocessor(*args, **kwargs)

    def postprocess(self, *args, **kwargs):
        assert self.postprocessor is not None
        return self.postprocessor(*args, **kwargs)


class ObjectDetectionModel(Model, ABC):
    """Object Detection Model Base Class"""

    task_type: ModelTaskType = ModelTaskType.OBJECT_DETECTION


class ImageClassificationModel(Model, ABC):
    """Image Classification Model Base Class"""

    task_type: ModelTaskType = ModelTaskType.IMAGE_CLASSIFICATION
