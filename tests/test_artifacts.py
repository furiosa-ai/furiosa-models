import pytest
import yaml

import artifacts


def sanity_check_for_dvc_file(model, dvc_file_path: str):
    assert model
    assert yaml.safe_load(open(dvc_file_path).read())["outs"][0]["size"] == len(model.model)


@pytest.mark.asyncio
async def test_mlcommons_resnet50():
    sanity_check_for_dvc_file(
        await artifacts.MLCommonsResNet50(),
        "models/mlcommons_resnet50_v1.5_int8.onnx.dvc",
    )


@pytest.mark.asyncio
async def test_mlcommons_ssd_mobilenet():
    sanity_check_for_dvc_file(
        await artifacts.MLCommonsSSDMobileNet(),
        "models/mlcommons_ssd_mobilenet_v1_int8.onnx.dvc",
    )


@pytest.mark.asyncio
async def test_mlcommons_ssd_resnet34():
    sanity_check_for_dvc_file(
        await artifacts.MLCommonsSSDResNet34(),
        "models/mlcommons_ssd_resnet34_int8.onnx.dvc",
    )


@pytest.mark.asyncio
async def test_efficientnetv2_s():
    assert await artifacts.EfficientNetV2_S()


@pytest.mark.asyncio
async def test_efficientnetv2_m():
    assert await artifacts.EfficientNetV2_M()
