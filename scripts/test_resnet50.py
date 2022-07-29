import numpy as np
import pytest

import furiosa.models.resnet50 as resnet50

from .helpers.util import InferenceTestSessionWrapper, load_image


@pytest.mark.asyncio
async def test_mlcommons_resnet50_perf():
    pipeline = resnet50.load_pipeline()
    test_image_path = "scripts/assets/cat.jpg"
    im = load_image(test_image_path)
    with InferenceTestSessionWrapper(m) as sess:
        result = sess.inference(im)
        assert result == "tabby, tabby cat", "check your result"
