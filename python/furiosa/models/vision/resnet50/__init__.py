from typing import Any, Dict, List, Sequence, Union

import cv2
import numpy as np

from furiosa.registry import Format, Metadata, Publication

from .. import native
from ...errors import ArtifactNotFound
from ...model import ClassificationModel
from ...utils import EXT_DFG, EXT_ENF, EXT_ONNX
from ..common.datasets import imagenet1k
from ..postprocess import PostProcessor
from ..preprocess import center_crop, resize_with_aspect_ratio

CLASSES: List[str] = imagenet1k.ImageNet1k_CLASSES
"""A list of class names"""


class ResNet50(ClassificationModel):
    """ResNet50 model"""

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
    """Preprocess an input image to an input tensor of ResNet50.
    This function can take a standard image file (e.g., jpg, gif, png) and return a numpy array.

    Args:
        image: A path of an image or an image loaded as numpy from `cv2.imread()`
    """
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


def postprocess(model_outputs: Sequence[np.ndarray]) -> str:
    """Convert the outputs of a model to a label string, such as car and cat.

    Arguments:
        model_outputs: the outputs of the model
    """
    return CLASSES[int(model_outputs[0]) - 1]


class Resnet50PostProcessor(PostProcessor):
    def eval(self, model_outputs: Sequence[np.ndarray], *args: Any, **kwargs: Any) -> str:
        return CLASSES[self._native.eval(model_outputs) - 1]


class NativePostProcessor(Resnet50PostProcessor):
    """Native postprocessing implementation optimized for NPU

    This class provides another version of the postprocessing implementation
    which is highly optimized for NPU. The implementation leverages the NPU IO architecture and runtime.

    To use this implementation, when this model is loaded, the parameter `use_native_post=True`
    should be passed to `load()` or `load_aync()`. Then, `NativePostProcess` object should
    be created with the model object. `eval()` method should be called to postprocess.

    !!! Examples
        ```python
        --8<-- "docs/examples/resnet50_native.py"
        ```
    """

    def __init__(self, model: ResNet50):
        if not model.dfg:
            raise ArtifactNotFound(model.name, "dfg")

        self._native = native.resnet50.PostProcessor(model.dfg)

        super().__init__()
