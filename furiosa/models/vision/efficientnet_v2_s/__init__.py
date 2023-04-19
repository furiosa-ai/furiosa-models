from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Type, Union

from PIL import Image, ImageOps
import numpy as np
import numpy.typing as npt

from furiosa.registry.model import Format, Metadata, Publication

from ...types import ImageClassificationModel, Platform, PostProcessor, PreProcessor
from ...utils import EXT_DFG, EXT_ENF, EXT_ONNX, get_field_default
from ..common.datasets import imagenet1k

IMAGENET_DEFAULT_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)[:, np.newaxis, np.newaxis]
IMAGENET_DEFAULT_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)[:, np.newaxis, np.newaxis]

CLASSES: List[str] = imagenet1k.ImageNet1k_CLASSES
INPUT_SIZE = 384


def resize(image: Image.Image, size: int, interpolation: Image.Resampling) -> Image.Image:
    width, height = image.size
    short, long = (width, height) if width <= height else (height, width)
    requested_new_short = size

    new_short, new_long = requested_new_short, int(requested_new_short * long / short)
    new_width, new_height = (new_short, new_long) if width <= height else (new_long, new_short)
    return image.resize((new_width, new_height), interpolation)


def center_crop(image: Image.Image, output_size: int) -> Image.Image:
    image_width, image_height = image.size
    crop_height, crop_width = output_size, output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = (
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        )
        image = ImageOps.expand(image, padding_ltrb, fill=(0, 0, 0))
        image_width, image_height = image.size
        if crop_width == image_width and crop_height == image_height:
            return image

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return image.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


def normalize(image: Image.Image) -> np.ndarray:
    image -= IMAGENET_DEFAULT_MEAN
    image /= IMAGENET_DEFAULT_STD
    return image


class EfficientNetV2sPreProcessor(PreProcessor):
    @staticmethod
    def __call__(image: Union[str, Path, npt.ArrayLike]) -> Tuple[np.ndarray, None]:
        """Read and preprocess an image located at image_path.

        Args:
            image: A path of an image.

        Returns:
            The first element of the tuple is a numpy array that meets the input requirements of the model.
                The second element of the tuple is unused in this model and has no value.
                To learn more information about the output numpy array, please refer to [Inputs](efficientnet_v2_s.md#inputs).

        """

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        image = resize(image, INPUT_SIZE, Image.Resampling.BILINEAR)
        image = center_crop(image, INPUT_SIZE)

        image = np.ascontiguousarray(image)
        image = np.transpose(image, (2, 0, 1))

        image = image.astype(np.float32) / 255

        data = normalize(image)
        return np.expand_dims(data, axis=0), None


class EfficientNetV2sPostProcessor(PostProcessor):
    def __call__(self, model_outputs: Sequence[npt.ArrayLike], contexts: Any = None) -> str:
        """Convert the outputs of a model to a label string, such as car and cat.

        Args:
            model_outputs: the outputs of the model.
                Please learn more about the output of model,
                please refer to [Outputs](efficientnet_b0.md#outputs).

        Returns:
            str: A classified label, e.g., "tabby, tabby cat".
        """

        return CLASSES[int(np.argsort(model_outputs[0], axis=1)[:, ::-1][0, 0])]


class EfficientNetV2s(ImageClassificationModel):
    """EfficientNetV2-s model"""

    postprocessor_map: Dict[Platform, Type[PostProcessor]] = {
        Platform.PYTHON: EfficientNetV2sPostProcessor,
    }

    @staticmethod
    def get_artifact_name():
        return "efficientnet_v2_s"

    @classmethod
    def load_aux(cls, artifacts: Dict[str, bytes], use_native: bool = False, *args, **kwargs):
        postprocessor = get_field_default(cls, "postprocessor_map")[Platform.PYTHON]()
        return cls(
            name="EfficientNetV2s",
            source=artifacts[EXT_ONNX],
            dfg=artifacts[EXT_DFG],
            enf=artifacts[EXT_ENF],
            format=Format.ONNX,
            family="EfficientNetV2",
            version="s",
            metadata=Metadata(
                description="EfficientNetV2s ImageNet-1K",
                publication=Publication(url="https://arxiv.org/abs/2104.00298"),
            ),
            preprocessor=EfficientNetV2sPreProcessor(),
            postprocessor=postprocessor,
        )
