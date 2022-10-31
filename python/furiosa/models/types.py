from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, Dict, Tuple

from furiosa.common.thread import synchronous
from furiosa.registry.model import Model as RegistryModel

from .utils import load_artifacts, model_file_name


class ModelTaskType(IntEnum):
    object_detection = 0
    image_classification = 1


class Model(ABC, RegistryModel):
    @classmethod
    @abstractmethod
    def get_artifact_name(cls) -> str:
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

    @abstractmethod
    def preprocess(self, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def postprocess(self, *args, **kwargs) -> Any:
        ...


class ContextedModel(Model, ABC):
    @abstractmethod
    def preprocess(self, *args, **kwargs) -> Tuple:
        ...

    @abstractmethod
    def postprocess(self, *args, **kwargs) -> Any:
        ...


class ObjectDetectionModel(Model, ABC):
    task_type: ModelTaskType = ModelTaskType.object_detection


class ImageClassificationModel(Model, ABC):
    task_type: ModelTaskType = ModelTaskType.image_classification
