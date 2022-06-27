import logging
import os
from typing import Any

import aiohttp
import dvc.api

from furiosa.artifacts.vision.models.image_classification import MLCommonsResNet50Model
from furiosa.artifacts.vision.models.object_detection import (
    MLCommonsSSDLargeModel,
    MLCommonsSSDSmallModel,
    YoloV5LargeModel,
)
from furiosa.registry import Format, Metadata, Publication

module_logger = logging.getLogger(__name__)


async def load_dvc(uri: str):
    dvc_repo = os.environ.get("DVC_REPO", None)
    dvc_rev = os.environ.get("DVC_REV", None)
    module_logger.debug(f"dvc_uri={uri}, DVC_REPO={dvc_repo}, DVC_REV={dvc_rev}")
    async with aiohttp.ClientSession() as session:
        async with session.get(
            dvc.api.get_url(
                uri,
                repo=os.environ.get("DVC_REPO", None),
                rev=os.environ.get("DVC_REV", None),
            )
        ) as resp:
            return await resp.read()


async def MLCommonsResNet50(*args: Any, **kwargs: Any) -> MLCommonsResNet50Model:
    return MLCommonsResNet50Model(
        name="MLCommonsResNet50",
        model=await load_dvc("models/mlcommons_resnet50_v1.5_int8.onnx"),
        format=Format.ONNX,
        family="ResNet",
        version="v1.5",
        metadata=Metadata(
            description="ResNet50 v1.5 int8 ImageNet-1K Accuracy 75.982% @ Top1",
            publication=Publication(url="https://arxiv.org/abs/1512.03385.pdf"),
        ),
        *args,
        **kwargs,
    )


# Object detection


async def MLCommonsSSDMobileNet(*args: Any, **kwargs: Any) -> MLCommonsSSDSmallModel:
    return MLCommonsSSDSmallModel(
        name="MLCommonsSSDMobileNet",
        model=await load_dvc("models/mlcommons_ssd_mobilenet_v1_int8.onnx"),
        format=Format.ONNX,
        family="MobileNetV1",
        version="v1.1",
        metadata=Metadata(
            description="MobileNet v1 model for MLCommons v1.1",
            publication=Publication(url="https://arxiv.org/abs/1704.04861.pdf"),
        ),
        *args,
        **kwargs,
    )


async def MLCommonsSSDResNet34(*args: Any, **kwargs: Any) -> MLCommonsSSDLargeModel:
    return MLCommonsSSDLargeModel(
        name="MLCommonsSSDResNet34",
        model=await load_dvc("models/mlcommons_ssd_resnet34_int8.onnx"),
        format=Format.ONNX,
        family="ResNet",
        version="v1.1",
        metadata=Metadata(
            description="ResNet34 model for MLCommons v1.1",
            publication=Publication(
                url="https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection"  # noqa: E501
            ),
        ),
        *args,
        **kwargs,
    )


async def YoloV5Large(*args: Any, **kwargs: Any) -> YoloV5LargeModel:
    return YoloV5LargeModel(
        name="YoloV5Large",
        model=await load_dvc("models/yolov5l_int8.onnx"),
        format=Format.ONNX,
        family="Yolo",
        version="v5",
        metadata=Metadata(
            description="Yolo v5 large model",
            publication=Publication(url="https://github.com/ultralytics/yolov5"),
        ),
        compiler_config={
            "without_quantize": {
                "parameters": [
                    {
                        "input_min": 0.0,
                        "input_max": 1.0,
                        "permute": [0, 2, 3, 1],  # "HWC" to "CHW"
                    }
                ]
            },
        },
        *args,
        **kwargs,
    )
