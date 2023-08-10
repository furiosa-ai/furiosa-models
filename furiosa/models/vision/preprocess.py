import os
from typing import Union

import cv2
import numpy as np
from numpy import ndarray


def center_crop(image: np.ndarray, cropped_height: int, cropped_width: int) -> np.ndarray:
    """Centrally crop `image` into cropped_width x cropped_height."""
    height, width, _ = image.shape
    top = int((height - cropped_height) / 2)
    bottom = int((height + cropped_height) / 2)
    left = int((width - cropped_width) / 2)
    right = int((width + cropped_width) / 2)
    image = image[top:bottom, left:right]
    return image


def resize_with_aspect_ratio(
    image: np.ndarray,
    scaled_height: int,
    scaled_width: int,
    percent: float,
    interpolation: int,
) -> np.ndarray:
    """Resize `image` so that it will be of scaled_width x scaled_height if it is scaled by
    `percent`.
    """
    height, width, _ = image.shape
    new_height = int(100.0 * scaled_height / percent)
    new_width = int(100.0 * scaled_width / percent)
    if height > width:
        new_height = int(new_height * height / width)
    else:
        new_width = int(new_width * width / height)
    image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    return image


def read_image_opencv_if_needed(image: Union[str, os.PathLike, ndarray]):
    if isinstance(image, ndarray):
        return image
    elif isinstance(image, os.PathLike):
        path = image.__fspath__()  # imread only accepts str (opencv/opencv#15731)
    else:
        path = image
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return image
