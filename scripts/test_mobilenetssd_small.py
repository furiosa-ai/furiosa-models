import numpy as np
import pytest

from furiosa.registry import Model
from .helpers.util import InferenceTestSessionWrapper

import artifacts


@pytest.mark.asyncio
async def test_mlcommons_mobilessd_small_perf():
    m: Model = await artifacts.MLCommonsSSDMobileNet()
    test_image_path = "scripts/assets/cat.jpg"

    assert len(m.classes) == 92, f"Classes is 92, but {len(m.classes)}"
    with InferenceTestSessionWrapper(m) as sess:
        true_bbox = np.array(
            [[187.30786, 88.035324, 763.6886, 655.2937]], dtype=np.float32
        )
        true_classid = np.array([17], dtype=np.int32)
        true_confidence = np.array([0.97390455], dtype=np.float32)
        result = sess.inference(test_image_path)
        assert len(result) == 3, "ssd_resnet34 output shape must be (1, 3)"
        bbox, classid, confidence = result
        assert np.array_equal(
            classid, true_classid
        ), f"wrong classid: {classid}, expected 16(cat)"
        assert (
            np.sum(np.abs(bbox - true_bbox)) < 1e-3
        ), f"bbox is different from expected value"
        assert (
            np.sum(np.abs(confidence - true_confidence)) < 1e-3
        ), "confidence is different from expected value"
