# Add all published models to make sure if the fetched model image is correct.

import yaml

from furiosa.models._utils import DATA_DIRECTORY_BASE
from furiosa.models.vision import (
    EfficientNetB0,
    EfficientNetV2s,
    ResNet50,
    SSDMobileNet,
    SSDResNet34,
    YOLOv5l,
    YOLOv5m,
)


def sanity_check_for_dvc_file(model, dvc_file_path: str):
    assert model
    assert model.model_source()
    assert model.origin
    assert yaml.safe_load(open(dvc_file_path).read())["outs"][0]["size"] == len(model.origin)


def test_mlcommons_resnet50():
    sanity_check_for_dvc_file(
        ResNet50(),
        next((DATA_DIRECTORY_BASE / "mlcommons_resnet50_v1.5").glob("*.onnx.dvc")),
    )


def test_ssd_mobilenet():
    sanity_check_for_dvc_file(
        SSDMobileNet(),
        next((DATA_DIRECTORY_BASE / "mlcommons_ssd_mobilenet_v1").glob("*.onnx.dvc")),
    )


def test_ssd_resnet34():
    sanity_check_for_dvc_file(
        SSDResNet34(),
        next((DATA_DIRECTORY_BASE / "mlcommons_ssd_resnet34").glob("*.onnx.dvc")),
    )


def test_yolov5_large():
    sanity_check_for_dvc_file(
        YOLOv5l(),
        next((DATA_DIRECTORY_BASE / "yolov5l").glob("*.onnx.dvc")),
    )


def test_yolov5_medium():
    sanity_check_for_dvc_file(
        YOLOv5m(),
        next((DATA_DIRECTORY_BASE / "yolov5m").glob("*.onnx.dvc")),
    )


def test_efficientnet_b0():
    sanity_check_for_dvc_file(
        EfficientNetB0(),
        next((DATA_DIRECTORY_BASE / "efficientnet_b0").glob("*.onnx.dvc")),
    )


def test_efficientnet_v2_s():
    sanity_check_for_dvc_file(
        EfficientNetV2s(),
        next((DATA_DIRECTORY_BASE / "efficientnet_v2_s").glob("*.onnx.dvc")),
    )
