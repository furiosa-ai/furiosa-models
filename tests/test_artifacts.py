import artifacts
import pytest


@pytest.mark.asyncio
async def test_mlcommons_resnet50():
    assert await artifacts.MLCommonsResNet50()


@pytest.mark.asyncio
async def test_mlcommons_ssd_mobilenet():
    assert await artifacts.MLCommonsSSDMobileNet()


@pytest.mark.asyncio
async def test_mlcommons_ssd_resnet34():
    assert await artifacts.MLCommonsSSDResNet34()


@pytest.mark.asyncio
async def test_efficientnetv2_s():
    assert await artifacts.EfficientNetV2_S()


@pytest.mark.asyncio
async def test_efficientnetv2_m():
    assert await artifacts.EfficientNetV2_M()
