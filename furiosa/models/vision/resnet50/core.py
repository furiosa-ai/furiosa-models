import os
from typing import Any, List

import cv2
import numpy as np

from furiosa.common.thread import synchronous
from furiosa.models.utils import load_dvc
from furiosa.runtime import session

__classes = [
    label.strip()
    for label in open(os.path.join(os.path.dirname(__file__), "imagenet-1k-label.txt")).readlines()
]

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
    """Resize `image` so that it will be of scaled_width x scaled_height if it is scaled by `percent`."""
    height, width, _ = image.shape
    new_height = int(100.0 * scaled_height / percent)
    new_width = int(100.0 * scaled_width / percent)
    if height > width:
        new_height = int(new_height * height / width)
    else:
        new_width = int(new_width * width / height)
    image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    return image

def load(image_path: str) -> np.array:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess(rgb_image: Any) -> np.array:
    """Read and preprocess an image located at image_path."""
    # https://github.com/mlcommons/inference/blob/af7f5a0b856402b9f461002cfcad116736a8f8af/vision/classification_and_detection/python/main.py#L37-L39
    # https://github.com/mlcommons/inference/blob/af7f5a0b856402b9f461002cfcad116736a8f8af/vision/classification_and_detection/python/dataset.py#L168-L184
    image = resize_with_aspect_ratio(
        rgb_image, 224, 224, percent=87.5, interpolation=cv2.INTER_AREA
    )
    image = center_crop(image, 224, 224)
    image = np.asarray(image, dtype=np.float32)
    # https://github.com/mlcommons/inference/blob/af7f5a0b856402b9f461002cfcad116736a8f8af/vision/classification_and_detection/python/dataset.py#L178
    image -= np.array([123.68, 116.78, 103.94], dtype=np.float32)
    image = image.transpose([2, 0, 1])
    return image[np.newaxis, ...]

def postprocess(output: Any, classes: List[str]) -> str:
    return classes[int(output[0].numpy()) - 1]

async def async_create_session() -> session.Session:
    model_weight = await load_dvc('./models/mlcommons_resnet50_v1.5_int8.onnx')
    print("model-weight:{}", len(model_weight))
    return session.create( bytes(model_weight) )

def create_session() -> session.Session:
    return synchronous(async_create_session())()

def inference(sess, image, post_config={}) -> str:
    pre_image = preprocess(image)
    predict = sess.run(pre_image)
    return postprocess(predict, __classes)
