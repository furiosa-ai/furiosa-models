import logging
from typing import Any, Dict, List, Sequence, Tuple, Type, Union

import cv2
import numpy as np
import numpy.typing as npt

from furiosa.registry.model import Format, Metadata, Publication

from .. import native
from ...errors import ArtifactNotFound
from ...types import ImageClassificationModel, Platform, PostProcessor, PreProcessor
from ...utils import EXT_DFG, EXT_ENF, EXT_ONNX, get_field_default
from ..common.datasets import imagenet1k
from ..preprocess import center_crop, resize_with_aspect_ratio

CLASSES: List[str] = imagenet1k.ImageNet1k_CLASSES

logger = logging.getLogger(__name__)


class ResNet50PreProcessor(PreProcessor):
    @staticmethod
    def __call__(image: Union[str, npt.ArrayLike]) -> Tuple[np.array, None]:
        """Convert an input image to a model input tensor

        Args:
            image: A path of an image or
                an image loaded as a numpy array in BGR order.

        Returns:
            The first element of tuple is a numpy array.
                To learn more about the output of preprocess,
                please refer to [Inputs](resnet50_v1.5.md#inputs).
        """
        # https://github.com/mlcommons/inference/blob/af7f5a0b856402b9f461002cfcad116736a8f8af/vision/classification_and_detection/python/main.py#L37-L39
        # https://github.com/mlcommons/inference/blob/af7f5a0b856402b9f461002cfcad116736a8f8af/vision/classification_and_detection/python/dataset.py#L168-L184
        if type(image) == str:
            image = cv2.imread(image)
            if image is None:
                raise FileNotFoundError(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_with_aspect_ratio(
            image, 224, 224, percent=87.5, interpolation=cv2.INTER_AREA
        )
        image = center_crop(image, 224, 224)
        image = np.asarray(image, dtype=np.float32)
        # https://github.com/mlcommons/inference/blob/af7f5a0b856402b9f461002cfcad116736a8f8af/vision/classification_and_detection/python/dataset.py#L178
        image -= np.array([123.68, 116.78, 103.94], dtype=np.float32)
        image = image.transpose([2, 0, 1])
        return image[np.newaxis, ...], None


class ResNet50PythonPostProcessor(PostProcessor):
    """Convert the outputs of a model to a label string, such as car and cat.

    Args:
        model_outputs: the outputs of the model.
            Please learn more about the output of model,
            please refer to [Outputs](resnet50_v1.5.md#outputs).

    Returns:
        str: A classified label
    """

    def __call__(self, model_outputs: Sequence[npt.ArrayLike], contexts: Any = None) -> str:
        return CLASSES[int(model_outputs[0]) - 1]


class ResNet50NativePostProcessor(PostProcessor):
    def __init__(self, dfg: bytes):
        self._native = native.resnet50.PostProcessor(dfg)

    def __call__(self, model_outputs: Sequence[npt.ArrayLike], contexts: Any = None) -> str:
        return CLASSES[self._native.eval(model_outputs) - 1]


class ResNet50(ImageClassificationModel):
    """MLCommons ResNet50 model"""

    postprocessor_map: Dict[Platform, Type[PostProcessor]] = {
        Platform.PYTHON: ResNet50PythonPostProcessor,
        Platform.RUST: ResNet50NativePostProcessor,
    }

    @staticmethod
    def get_artifact_name():
        return "mlcommons_resnet50_v1.5_int8"

    @classmethod
    def load_aux(cls, artifacts: Dict[str, bytes], use_native: bool = True, *args, **kwargs):
        dfg = artifacts[EXT_DFG]
        if use_native and dfg is None:
            raise ArtifactNotFound(cls.get_artifact_name(), EXT_DFG)
        postproc_type = Platform.RUST if use_native else Platform.PYTHON
        logger.debug(f"Using {postproc_type.name} postprocessor")
        postprocessor = get_field_default(cls, "postprocessor_map")[postproc_type](dfg)
        return cls(
            name="ResNet50",
            source=artifacts[EXT_ONNX],
            dfg=dfg,
            enf=artifacts[EXT_ENF],
            format=Format.ONNX,
            family="ResNet",
            version="v1.5",
            metadata=Metadata(
                description="ResNet50 v1.5 int8 ImageNet-1K",
                publication=Publication(url="https://arxiv.org/abs/1512.03385.pdf"),
            ),
            preprocessor=ResNet50PreProcessor(),
            postprocessor=postprocessor,
        )
