import numpy as np
import pytest

from furiosa.models.vision import SSDResNet34, ssd_resnet34
from furiosa.registry import Model
from furiosa.runtime import session


@pytest.mark.asyncio
async def test_mlcommons_ssd_resnet34_perf():
    m: Model = SSDResNet34()
    test_image_path = "scripts/assets/cat.jpg"

    assert len(ssd_resnet34.CLASSES) == 81, f"Classes is 81, but {len(ssd_resnet34.CLASSES)}"
    with session.create(m.model) as sess:
        true_bbox = np.array(
            [
                [264.24792, 259.05603, 699.12964, 474.65332],
                [221.0502, 123.12275, 549.879, 543.1015],
            ],
            dtype=np.float32,
        )
        true_classid = np.array([16, 16], dtype=np.int32)
        true_confidence = np.array([0.37563688, 0.8747512], dtype=np.float32)

        preprocessed = ssd_resnet34.preprocess(test_image_path)
        outputs = sess.run(preprocessed[0]).numpy()
        result = ssd_resnet34.postprocess(
            outputs=outputs, extra_params=preprocessed[1], confidence_threshold=0.3
        )
        assert len(result) == 3, "ssd_resnet34 output shape must be (1, 3)"
        bbox, classid, confidence = result
        assert np.array_equal(classid, true_classid), f"wrong classid: {classid}, expected 16(cat)"
        assert np.sum(np.abs(bbox - true_bbox)) < 1e-3, f"bbox is different from expected value"
        assert (
            np.sum(np.abs(confidence - true_confidence)) < 1e-3
        ), "confidence is different from expected value"
