import math

import numpy as np
import pytest

from furiosa.models.vision import ssd_mobilenet_v1_5 as detector


@pytest.mark.asyncio
async def test_mlcommons_mobilessd_small_perf():
    test_image_path = "scripts/assets/cat.jpg"

    with await detector.create_session() as sess:
        result = detector.inference(sess, detector.load_image(test_image_path))

        assert len(result) == 1, "ssd_resnet34 output shape must be 1"

        assert result[0].predicted_class == "cat", f"wrong classid: {result[0].predicted_class}, expected cat"
        true_bbox = np.array([[187.30786, 88.035324, 763.6886, 655.2937]], dtype=np.float32)
        assert np.sum(np.abs(np.array(result[0].boundingbox.numpy(), dtype=np.float32) - true_bbox)) < 1e-3, f"bbox is different from expected value"
        true_score= 0.97390455
        assert (
            math.fabs(result[0].score - true_score) < 1e-3
        ), "score is different from expected value"
        print("finish detected")
