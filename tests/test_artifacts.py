import pytest
from furiosa.artifacts.vision import artifacts


@pytest.mark.asyncio
async def test_mlcommons_resnet50():
    assert await artifacts.mlcommons_resnet50()


@pytest.mark.asyncio
async def test_mlcommons_ssd_mobilenet():
    assert await artifacts.mlcommons_ssd_mobilenet()


@pytest.mark.asyncio
async def test_mlcommons_ssd_resnet34():
    assert await artifacts.mlcommons_ssd_resnet34()
