from typing import Any, Dict, List, Sequence, Union

import cv2
import numpy
import numpy as np

from furiosa.common.thread import synchronous
from furiosa.registry import Format, Metadata, Model, Publication

from .. import native
from ...errors import ArtifactNotFound
from ...utils import EXT_DFG, EXT_ENF, EXT_ONNX, load_artifacts, model_file_name
from ..common.datasets import imagenet1k
from ..postprocess import PostProcessor
from ..preprocess import center_crop, resize_with_aspect_ratio

CLASSES: List[str] = imagenet1k.ImageNet1k_CLASSES

_ARTIFACT_NAME = "mlcommons_resnet50_v1.5_int8"


class ResNet50(Model):
    """MLCommons ResNet50 model"""

    @classmethod
    def __load(cls, artifacts: Dict[str, bytes], *args, **kwargs):
        return cls(
            name="ResNet50",
            source=artifacts[EXT_ONNX],
            dfg=artifacts[EXT_DFG],
            enf=artifacts[EXT_ENF],
            format=Format.ONNX,
            family="ResNet",
            version="v1.5",
            metadata=Metadata(
                description="ResNet50 v1.5 int8 ImageNet-1K",
                publication=Publication(url="https://arxiv.org/abs/1512.03385.pdf"),
            ),
            *args,
            **kwargs,
        )

    @classmethod
    async def load_async(cls, use_native_post=False, *args, **kwargs) -> Model:
        artifact_name = model_file_name(_ARTIFACT_NAME, use_native_post)
        return cls.__load(await load_artifacts(artifact_name), *args, **kwargs)

    @classmethod
    def load(cls, use_native_post: bool = False, *args, **kwargs) -> Model:
        artifact_name = model_file_name(_ARTIFACT_NAME, use_native_post)
        return cls.__load(synchronous(load_artifacts)(artifact_name), *args, **kwargs)


def preprocess(image: Union[str, np.ndarray]) -> np.array:
    """Read and preprocess an image located at image_path."""
    # https://github.com/mlcommons/inference/blob/af7f5a0b856402b9f461002cfcad116736a8f8af/vision/classification_and_detection/python/main.py#L37-L39
    # https://github.com/mlcommons/inference/blob/af7f5a0b856402b9f461002cfcad116736a8f8af/vision/classification_and_detection/python/dataset.py#L168-L184
    if type(image) == str:
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_with_aspect_ratio(image, 224, 224, percent=87.5, interpolation=cv2.INTER_AREA)
    image = center_crop(image, 224, 224)
    image = np.asarray(image, dtype=np.float32)
    # https://github.com/mlcommons/inference/blob/af7f5a0b856402b9f461002cfcad116736a8f8af/vision/classification_and_detection/python/dataset.py#L178
    image -= np.array([123.68, 116.78, 103.94], dtype=np.float32)
    image = image.transpose([2, 0, 1])
    return image[np.newaxis, ...]


def postprocess(outputs: Sequence[numpy.ndarray]) -> str:
    return CLASSES[int(outputs[0]) - 1]


class Resnet50PostProcessor(PostProcessor):
    def eval(self, inputs: Sequence[numpy.ndarray], *args: Any, **kwargs: Any) -> str:
        return CLASSES[self._native.eval(inputs) - 1]


class NativePostProcessor(Resnet50PostProcessor):
    def __init__(self, model: Model):
        if not model.dfg:
            raise ArtifactNotFound(model.name, "dfg")

        self._native = native.resnet50.PostProcessor(model.dfg)

        super().__init__()
