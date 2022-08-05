import numpy as np
import pytest

from furiosa.models.vision import SSDMobileNet, ssd_mobilenet
from furiosa.registry import Model
from furiosa.runtime import session


@pytest.mark.asyncio
async def test_mlcommons_mobilessd_small_perf():
    m: Model = SSDMobileNet()
    test_image_path = "scripts/assets/cat.jpg"

    assert len(ssd_mobilenet.CLASSES) == 92, f"Classes is 92, but {len(ssd_mobilenet.CLASSES)}"
    with session.create(m.model) as sess:
        true_bbox = np.array([[187.30786, 88.035324, 763.6886, 655.2937]], dtype=np.float32)
        true_classid = np.array([17], dtype=np.int32)
        true_confidence = np.array([0.97390455], dtype=np.float32)
        preprocessed = ssd_mobilenet.preprocess(test_image_path)
        outputs = sess.run(preprocessed[0]).numpy()
        result = ssd_mobilenet.postprocess(outputs=outputs, extra_params=preprocessed[1])
        assert len(result) == 3, "ssd_resnet34 output shape must be (1, 3)"
        bbox, classid, confidence = result
        assert np.array_equal(classid, true_classid), f"wrong classid: {classid}, expected 16(cat)"
        assert np.sum(np.abs(bbox - true_bbox)) < 1e-3, f"bbox is different from expected value"
        assert (
            np.sum(np.abs(confidence - true_confidence)) < 1e-3
        ), "confidence is different from expected value"
