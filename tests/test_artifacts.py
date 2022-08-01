# Add all published models to make sure if the fetched model image is correct.

import pytest
import yaml

from furiosa.models.vision import nonblocking


def sanity_check_for_dvc_file(model, dvc_file_path: str):
    assert model
    assert yaml.safe_load(open(dvc_file_path).read())["outs"][0]["size"] == len(model.model)


@pytest.mark.asyncio
async def test_yolov5_large():
    sanity_check_for_dvc_file(
        await nonblocking.YOLOv5l(),
        "models/yolov5l_int8.onnx.dvc",
    )
