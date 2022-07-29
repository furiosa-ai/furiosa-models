import numpy as np
import pytest

import furiosa.models.vision.resnet50 as resnet50

from .helpers.util import load_image


@pytest.mark.asyncio
async def test_mlcommons_resnet50_perf():
    test_image_path = "scripts/assets/cat.jpg"
    im = load_image(test_image_path)
    with await resnet50.async_create_session() as sess:
        result = resnet50.inference(sess, im)
        print("predicted result", result)
        assert result == "tabby, tabby cat", "check your result"
