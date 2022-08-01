from typing import Any

from furiosa.models.utils import load_dvc
from furiosa.models.vision import resnet50
from furiosa.models.vision.yolov5 import large as yolov5l
from furiosa.registry import Format, Metadata, Publication

__all__ = [
    "ResNet50",
    "SSDMobileNet",
    "SSDResNet34",
    "YOLOv5l",
    "resnet50",
    "ssd_resnet34",
    "yolov5l",
]


async def ResNet50(*args: Any, **kwargs: Any) -> resnet50.MLCommonsResNet50Model:
    return resnet50.MLCommonsResNet50Model(
        name="ResNet50",
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



async def YOLOv5l(*args: Any, **kwargs: Any) -> yolov5l.YoloV5LargeModel:
    return yolov5l.YoloV5LargeModel(
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
