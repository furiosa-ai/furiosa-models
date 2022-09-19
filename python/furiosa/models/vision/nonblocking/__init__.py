from typing import Any

from furiosa.registry import Format, Metadata, Publication

from ...utils import load_dvc, load_dvc_generated
from ...vision import resnet50, ssd_mobilenet, ssd_resnet34
from ...vision.yolov5 import large as yolov5l
from ...vision.yolov5 import medium as yolov5m

__all__ = [
    "ResNet50",
    "SSDMobileNet",
    "SSDResNet34",
    "YOLOv5l",
    "YOLOv5m",
    "resnet50",
    "ssd_mobilenet",
    "ssd_resnet34",
    "yolov5l",
]


_ENF = "enf"
_DFG = "dfg"


def __model_file(relative_path, truncated=True) -> str:
    if truncated:
        return f"{relative_path}_truncated.onnx"
    else:
        return relative_path


async def ResNet50(use_native_post=False, *args: Any, **kwargs: Any) -> resnet50.ResNet50Model:

    source_path = __model_file("models/mlcommons_resnet50_v1.5_int8.onnx", use_native_post)

    return resnet50.ResNet50Model(
        name="ResNet50",
        source=await load_dvc(source_path),
        dfg=await load_dvc_generated(source_path, _DFG),
        enf=await load_dvc_generated(source_path, _ENF),
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
async def SSDMobileNet(
    use_native_post=False, *args: Any, **kwargs: Any
) -> ssd_mobilenet.SSDMobileNetModel:

    source_path = __model_file("models/mlcommons_ssd_mobilenet_v1_int8.onnx", use_native_post)

    return ssd_mobilenet.SSDMobileNetModel(
        name="MLCommonsSSDMobileNet",
        source=await load_dvc(source_path),
        dfg=await load_dvc_generated(source_path, _DFG),
        enf=await load_dvc_generated(source_path, _ENF),
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


async def SSDResNet34(
    use_native_post=False, *args: Any, **kwargs: Any
) -> ssd_resnet34.SSDResNet34Model:

    source_path = __model_file("models/mlcommons_ssd_resnet34_int8.onnx", use_native_post)

    return ssd_resnet34.SSDResNet34Model(
        name="MLCommonsSSDResNet34",
        source=await load_dvc(source_path),
        dfg=await load_dvc_generated(source_path, _DFG),
        enf=await load_dvc_generated(source_path, _ENF),
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


async def YOLOv5l(use_native_post=False, *args: Any, **kwargs: Any) -> yolov5l.YoloV5LargeModel:
    source_path = __model_file("models/yolov5l_int8.onnx", use_native_post)
    return yolov5l.YoloV5LargeModel(
        name="YOLOv5Large",
        source=await load_dvc(source_path),
        dfg=await load_dvc_generated(source_path, _DFG),
        enf=await load_dvc_generated(source_path, _ENF),
        format=Format.ONNX,
        family="YOLOv5",
        version="v5",
        metadata=Metadata(
            description="YOLOv5 large model",
            publication=Publication(url="https://github.com/ultralytics/yolov5"),
        ),
        *args,
        **kwargs,
    )


async def YOLOv5m(use_native_post=False, *args: Any, **kwargs: Any) -> yolov5m.YoloV5MediumModel:
    source_path = __model_file("models/yolov5m_int8.onnx", use_native_post)
    return yolov5m.YoloV5MediumModel(
        name="YOLOv5Medium",
        source=await load_dvc(source_path),
        # FIXME
        # dfg=await load_dvc_generated(source_path, _DFG),
        # enf=await load_dvc_generated(source_path, _ENF),
        format=Format.ONNX,
        family="YOLOv5",
        version="v5",
        metadata=Metadata(
            description="YOLOv5 medium model",
            publication=Publication(url="https://github.com/ultralytics/yolov5"),
        ),
        *args,
        **kwargs,
    )
