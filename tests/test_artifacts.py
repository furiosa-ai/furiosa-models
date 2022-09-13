# Add all published models to make sure if the fetched model image is correct.

import pytest
import yaml

from furiosa.models.vision import nonblocking


def sanity_check_for_dvc_file(model, dvc_file_path: str):
    assert model
    assert model.dfg
    assert model.enf
    assert yaml.safe_load(open(dvc_file_path).read())["outs"][0]["size"] == len(model.source)


@pytest.mark.asyncio
async def test_mlcommons_resnet50():
    sanity_check_for_dvc_file(
        await nonblocking.ResNet50(),
        "python/furiosa/models/data/mlcommons_resnet50_v1.5_int8.onnx.dvc",
    )


@pytest.mark.asyncio
async def test_mlcommons_ssd_mobilenet():
    sanity_check_for_dvc_file(
        await nonblocking.SSDMobileNet(),
        "python/furiosa/models/data/mlcommons_ssd_mobilenet_v1_int8.onnx.dvc",
    )


@pytest.mark.asyncio
async def test_mlcommons_ssd_resnet34():
    sanity_check_for_dvc_file(
        await nonblocking.SSDResNet34(),
        "python/furiosa/models/data/mlcommons_ssd_resnet34_int8.onnx.dvc",
    )


@pytest.mark.asyncio
async def test_yolov5_large():
    sanity_check_for_dvc_file(
        await nonblocking.YOLOv5l(),
        "python/furiosa/models/data/yolov5l_int8.onnx.dvc",
    )
