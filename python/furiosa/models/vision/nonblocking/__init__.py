from typing import Any

from furiosa.registry import Format, Metadata, Publication

from ...utils import load_dvc, load_dvc_generated
from ...vision import resnet50, ssd_mobilenet, ssd_resnet34
from ...vision.yolov5 import large as yolov5l

__all__ = [
    "ResNet50",
    "SSDMobileNet",
    "SSDResNet34",
    "YOLOv5l",
    "resnet50",
    "ssd_mobilenet",
    "ssd_resnet34",
    "yolov5l",
]


_ENF = "enf"
_DFG = "dfg"


async def ResNet50(
    optimized_postprocess=False, *args: Any, **kwargs: Any
) -> resnet50.MLCommonsResNet50Model:
    if optimized_postprocess:
        source_path = "models/mlcommons_resnet50_v1.5_int8_truncated.onnx"
    else:
        source_path = "models/mlcommons_resnet50_v1.5_int8.onnx"

    return resnet50.MLCommonsResNet50Model(
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
    optimized_postprocess=False, *args: Any, **kwargs: Any
) -> ssd_mobilenet.MLCommonsSSDSmallModel:
    if optimized_postprocess:
        source_path = "models/mlcommons_ssd_mobilenet_v1_int8.onnx_truncated.onnx"
    else:
        source_path = "models/mlcommons_ssd_mobilenet_v1_int8.onnx"

    return ssd_mobilenet.MLCommonsSSDSmallModel(
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
    optimized_postprocess=False, *args: Any, **kwargs: Any
) -> ssd_resnet34.MLCommonsSSDLargeModel:
    if optimized_postprocess:
        source_path = "models/mlcommons_ssd_resnet34_int8.onnx_truncated.onnx"
    else:
        source_path = "models/mlcommons_ssd_resnet34_int8.onnx"

    return ssd_resnet34.MLCommonsSSDLargeModel(
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


async def YOLOv5l(*args: Any, **kwargs: Any) -> yolov5l.YoloV5LargeModel:
    source_path = "models/yolov5l_int8.onnx"
    return yolov5l.YoloV5LargeModel(
        name="YoloV5Large",
        source=await load_dvc(source_path),
        dfg=await load_dvc_generated(source_path, _DFG),
        enf=await load_dvc_generated(source_path, _ENF),
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
