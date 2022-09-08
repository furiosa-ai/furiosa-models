from typing import Any, List

import cv2
import numpy as np

from furiosa.registry import Model

from .common.datasets import imagenet1k
from .preprocess import center_crop, resize_with_aspect_ratio

CLASSES: List[str] = imagenet1k.ImageNet1k_CLASSES


class ResNet50Model(Model):
    """MLCommons ResNet50 model"""

    pass


def preprocess(image_path: str) -> np.array:
    """Read and preprocess an image located at image_path."""
    # https://github.com/mlcommons/inference/blob/af7f5a0b856402b9f461002cfcad116736a8f8af/vision/classification_and_detection/python/main.py#L37-L39
    # https://github.com/mlcommons/inference/blob/af7f5a0b856402b9f461002cfcad116736a8f8af/vision/classification_and_detection/python/dataset.py#L168-L184
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_with_aspect_ratio(image, 224, 224, percent=87.5, interpolation=cv2.INTER_AREA)
    image = center_crop(image, 224, 224)
    image = np.asarray(image, dtype=np.float32)
    # https://github.com/mlcommons/inference/blob/af7f5a0b856402b9f461002cfcad116736a8f8af/vision/classification_and_detection/python/dataset.py#L178
    image -= np.array([123.68, 116.78, 103.94], dtype=np.float32)
    image = image.transpose([2, 0, 1])
    return image[np.newaxis, ...]


def postprocess(output: Any) -> str:
    return CLASSES[int(output[0].numpy()) - 1]
