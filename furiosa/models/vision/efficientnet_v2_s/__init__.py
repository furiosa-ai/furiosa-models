from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageOps
import numpy as np
import numpy.typing as npt

from furiosa.registry.model import Format, Metadata, Publication

from ...types import ImageClassificationModel, PostProcessor, PreProcessor
from ...utils import EXT_DFG, EXT_ENF, EXT_ONNX
from ..common.datasets import imagenet1k

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
    image -= np.asarray((0.485, 0.456, 0.406)).reshape(-1, 1, 1)
    image /= np.asarray((0.229, 0.224, 0.225)).reshape(-1, 1, 1)
    return image


class EfficientNetV2sPreProcessor(PreProcessor):
    @staticmethod
    def __call__(image: Path) -> Tuple[np.ndarray, None]:
        """Read and preprocess an image located at image_path."""
        image = Image.open(image).convert("RGB")

        image = resize(image, INPUT_SIZE, Image.Resampling.BILINEAR)
        image = center_crop(image, INPUT_SIZE)

        image = np.ascontiguousarray(image)
        image = np.transpose(image, (2, 0, 1))

        image = image.astype(np.float32) / 255

        data = normalize(image)
        return data[np.newaxis, ...], None


class EfficientNetV2sPostProcessor(PostProcessor):
    def __call__(self, model_outputs: Sequence[npt.ArrayLike], contexts: Any = None) -> str:
        return CLASSES[int(np.argsort(model_outputs[0], axis=1)[:, ::-1][0, 0])]


class EfficientNetV2s(ImageClassificationModel):
    """EfficientNetV2-s model"""

    @staticmethod
    def get_artifact_name():
        return "efficientnet_v2_s"

    @classmethod
    def load_aux(cls, artifacts: Dict[str, bytes], use_native: bool = False, *args, **kwargs):
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
            postprocessor=EfficientNetV2sPostProcessor(),
        )
