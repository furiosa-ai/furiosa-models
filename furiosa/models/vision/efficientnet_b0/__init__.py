import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Type

from PIL import Image
import numpy as np
import numpy.typing as npt

from furiosa.registry.model import Format, Metadata, Publication

from ...types import ImageClassificationModel, Platform, PostProcessor, PreProcessor
from ...utils import EXT_DFG, EXT_ENF, EXT_ONNX
from ..common.datasets import imagenet1k
from ..preprocess import center_crop

CLASSES: List[str] = imagenet1k.ImageNet1k_CLASSES

IMAGENET_DEFAULT_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)[:, np.newaxis, np.newaxis]
IMAGENET_DEFAULT_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)[:, np.newaxis, np.newaxis]


# https://github.com/pytorch/vision/blob/7ba97196757229552cede54639be75e3a0a9959f/torchvision/transforms/functional.py#L386-L392
def resize(image: Image.Image, size: int, resample: Image.Resampling) -> Image.Image:
    """Resize `image` so that a smaller edge of the resulting image will be matched to `size`."""
    # https://github.com/pytorch/vision/blob/0b6b8d42e95a208df85033a46fc7984c74301b35/torchvision/transforms/functional.py#L463
    # https://github.com/pytorch/vision/blob/0b6b8d42e95a208df85033a46fc7984c74301b35/torchvision/transforms/functional.py#L363-L383
    width, height = image.size
    if width <= height:
        new_width = size
        new_height = int(size * height / width)
    else:
        new_width = int(size * width / height)
        new_height = size

    image = image.resize((new_width, new_height), resample=resample)
    return image


# https://github.com/pytorch/vision/blob/7ba97196757229552cede54639be75e3a0a9959f/torchvision/transforms/functional.py#L551
def center_crop(image: Image.Image, cropped_height: int, cropped_width: int) -> Image.Image:
    """Centrally crop `image` into cropped_width x cropped_height."""
    width, height = image.size

    # https://github.com/pytorch/vision/blob/7ba97196757229552cede54639be75e3a0a9959f/torchvision/transforms/functional.py#L587-L588
    top = int(round((height - cropped_height) / 2))
    left = int(round((width - cropped_width) / 2))

    # https://github.com/pytorch/vision/blob/7ba97196757229552cede54639be75e3a0a9959f/torchvision/transforms/functional_pil.py#L237
    image = image.crop((left, top, left + cropped_width, top + cropped_height))
    return image


class EfficientNetB0PreProcessor(PreProcessor):
    @staticmethod
    def __call__(image: Path) -> Tuple[np.ndarray, None]:
        """Read and preprocess an image located at image_path."""

        image = Image.open(Path(image)).convert("RGB")

        scale_size = int(math.floor(224 / 0.875))
        image = resize(image, scale_size, resample=Image.Resampling.BICUBIC)

        image = center_crop(image, 224, 224)

        data = np.asarray(image, dtype=np.float32)
        data = np.transpose(data, axes=(2, 0, 1))
        data /= 255

        data -= IMAGENET_DEFAULT_MEAN
        data /= IMAGENET_DEFAULT_STD

        return data[np.newaxis, ...], None


class EfficientNetB0PostProcessor(PostProcessor):
    def __call__(self, model_outputs: Sequence[npt.ArrayLike], contexts: Any = None) -> str:
        return CLASSES[int(np.argsort(model_outputs[0], axis=1)[:, ::-1][0, 0])]


class EfficientNetB0(ImageClassificationModel):
    """EfficientNet B0 model"""

    postprocessor_map: Dict[Platform, Type[PostProcessor]] = {
        Platform.PYTHON: EfficientNetB0PostProcessor,
    }

    @staticmethod
    def get_artifact_name():
        return "efficientnet_b0"

    @classmethod
    def load_aux(cls, artifacts: Dict[str, bytes], use_native: bool = False, *args, **kwargs):
        return cls(
            name="EfficientNetB0",
            source=artifacts[EXT_ONNX],
            # dfg=artifacts[EXT_DFG],
            enf=artifacts[EXT_ENF],
            format=Format.ONNX,
            family="EfficientNet",
            version="1.0.2",
            metadata=Metadata(
                description="EfficientNetB0 ImageNet-1K",
                publication=Publication(url="https://arxiv.org/abs/1905.11946"),
            ),
            preprocessor=EfficientNetB0PreProcessor(),
            postprocessor=EfficientNetB0PostProcessor(),
        )
