import pytest

from furiosa.models import vision
from furiosa.registry import Model

from .helpers.util import InferenceTestSessionWrapper


@pytest.mark.asyncio
async def test_mlcommons_resnet50_perf():
    m: Model = await vision.MLCommonsResNet50()
    test_image_path = "scripts/assets/cat.jpg"

    with InferenceTestSessionWrapper(m) as sess:
        result = sess.inference(test_image_path)
        assert result == "tabby, tabby cat", "check your result"
