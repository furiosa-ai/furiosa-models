import io
import os
from typing import Any

import aiohttp
import dvc.api
from furiosa.registry import Format, Metadata, Model, Publication

from furiosa.artifacts.vision.models.image_classification import (
    EfficientNetV2_M as EfficientNetV2_MModel,
)
from furiosa.artifacts.vision.models.image_classification import (
    EfficientNetV2_S as EfficientNetV2_SModel,
)
from furiosa.artifacts.vision.models.image_classification import MLCommonsResNet50Model
from furiosa.artifacts.vision.models.object_detection import (
    MLCommonsSSDLargeModel,
    MLCommonsSSDSmallModel,
)


async def load_dvc(uri: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            dvc.api.get_url(
                uri, repo=os.environ.get("DVC_REPO", None), rev=os.environ.get("DVC_REV", None)
            )
        ) as resp:
            resp.raise_for_status()
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


async def EfficientNetV2_S(*args: Any, **kwargs: Any) -> Model:
    return Model(
        name="EfficientNetV2_S",
        model=EfficientNetV2_SModel().export(io.BytesIO()).getvalue(),
        format=Format.ONNX,
        family="EfficientNet",
        version="v2.0",
        metadata=Metadata(
            description="EfficientNetV2 from Google AutoML",
            publication=Publication(url="https://arxiv.org/abs/2104.00298"),
        ),
        *args,
        **kwargs,
    )


async def EfficientNetV2_M(*args: Any, **kwargs: Any) -> Model:
    return Model(
        name="EfficientNetV2_M",
        model=EfficientNetV2_MModel().export(io.BytesIO()).getvalue(),
        format=Format.ONNX,
        family="EfficientNet",
        version="v2.0",
        metadata=Metadata(
            description="EfficientNetV2 from Google AutoML",
            publication=Publication(url="https://arxiv.org/abs/2104.00298"),
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
