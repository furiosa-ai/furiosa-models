from typing import Any, Dict, List, Sequence, Union

import cv2
import numpy
import numpy as np

from furiosa.models.types import Format, Metadata, Publication

from .. import native
from ...errors import ArtifactNotFound
from ...types import ImageClassificationModel
from ...utils import EXT_DFG, EXT_ENF, EXT_ONNX
from ..common.datasets import imagenet1k
from ..postprocess import PostProcessor
from ..preprocess import center_crop, resize_with_aspect_ratio

CLASSES: List[str] = imagenet1k.ImageNet1k_CLASSES


class ResNet50(ImageClassificationModel):
    """MLCommons ResNet50 model"""

    @classmethod
    def get_artifact_name(cls):
        return "mlcommons_resnet50_v1.5_int8"

    @classmethod
    def load_aux(cls, artifacts: Dict[str, bytes], *args, **kwargs):
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
    def __init__(self, model: ResNet50):
        if not model.dfg:
            raise ArtifactNotFound(model.name, "dfg")

        self._native = native.resnet50.PostProcessor(model.dfg)

        super().__init__()
