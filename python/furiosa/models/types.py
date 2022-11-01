from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy.typing as npt
from pydantic import BaseConfig, BaseModel, Extra
from typing_extensions import TypeAlias

from furiosa.common.thread import synchronous
from furiosa.registry.model import Format, Metadata
from furiosa.registry.model import Model as RegistryModel
from furiosa.registry.model import Publication

from .utils import load_artifacts, model_file_name

# Context type alias
Context: TypeAlias = Any


class PreProcessor(ABC):
    @abstractmethod
    def __call__(
        self, inputs: Any, *args, **kwargs
    ) -> Tuple[Sequence[npt.ArrayLike], Sequence[Context]]:
        ...


class PostProcessor(ABC):
    @abstractmethod
    def __call__(
        self, inputs: Sequence[npt.ArrayLike], contexts: Sequence[Context], *args, **kwargs
    ):
        ...


class DataProcessor:
    """Data pre/post processor with context (even if doesn't needed)"""

    preprocessor: PreProcessor
    postprocessor: PostProcessor


class ModelTaskType(IntEnum):
    """Model's task type"""

    OBJECT_DETECTION = 0
    IMAGE_CLASSIFICATION = 1


class Model(RegistryModel, BaseModel, ABC):
    class Config(BaseConfig):
        extra: Extra = Extra.forbid
        # To allow Session, Processor type
        arbitrary_types_allowed = True

    processor: Optional[DataProcessor] = None

    @staticmethod
    @abstractmethod
    def get_artifact_name() -> str:
        ...

    @classmethod
    @abstractmethod
    def load_aux(cls, artifacts: Dict[str, bytes], use_native: bool, *args, **kwargs):
        ...

    @classmethod
    async def load_async(cls, use_native: bool = False, *args, **kwargs) -> 'Model':
        artifact_name = model_file_name(cls.get_artifact_name(), use_native)
        return cls.load_aux(await load_artifacts(artifact_name), use_native, *args, **kwargs)

    @classmethod
    def load(cls, use_native: bool = False, *args, **kwargs) -> 'Model':
        artifact_name = model_file_name(cls.get_artifact_name(), use_native)
        return cls.load_aux(synchronous(load_artifacts)(artifact_name), use_native, *args, **kwargs)

    def preprocess(self, *args, **kwargs) -> Tuple[Sequence[npt.ArrayLike], Sequence[Context]]:
        assert self.processor is not None
        return self.processor.preprocessor(*args, **kwargs)

    def postprocess(self, *args, **kwargs):
        assert self.processor is not None
        return self.processor.postprocessor(*args, **kwargs)


class ObjectDetectionModel(Model, ABC):
    """Object Detection Model Base Class"""

    task_type: ModelTaskType = ModelTaskType.OBJECT_DETECTION


class ImageClassificationModel(Model, ABC):
    """Image Classification Model Base Class"""

    task_type: ModelTaskType = ModelTaskType.IMAGE_CLASSIFICATION
