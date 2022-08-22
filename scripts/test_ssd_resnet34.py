import numpy as np
import pytest

from furiosa.models.vision import SSDResNet34
from furiosa.models.vision.ssd_resnet34 import CLASSES, NUM_OUTPUTS, postprocess, preprocess
from furiosa.registry import Model
from furiosa.runtime import session


@pytest.mark.asyncio
async def test_mlcommons_ssd_resnet34_perf():
    m: Model = SSDResNet34()
    test_image_path = "scripts/assets/cat.jpg"

    assert len(CLASSES) == 81, f"Classes is 81, but {len(CLASSES)}"
    sess = session.create(m.model, batch_size=2)
    true_bbox = np.array(
        [
            [264.24792, 259.05603, 963.37756, 733.70935000000012],
            [221.0502, 123.12275, 770.9292, 666.22425],
        ],
        dtype=np.float32,
    )
    true_confidence = np.array([0.37563688, 0.8747512], dtype=np.float32)

    # For batch mode test, simply read two identical images.
    batch_pre_image, batch_preproc_param = preprocess([test_image_path, test_image_path])
    batch_feat = sess.run(batch_pre_image).numpy()
    detected_result = postprocess(batch_feat, batch_preproc_param, confidence_threshold=0.3)
    assert len(detected_result) == 2, "batch size must be 2"
    detected_result = detected_result[0]  # due to duplicated input image

    assert len(detected_result) == 2, "ssd_resnet34 output shape must be 2"
    assert [detected_result[0].label, detected_result[1].label] == [
        'cat',
        'cat',
    ], f"wrong classid: {detected_result[0].label}, {detected_result[1].label}, expected [cat, cat]"
    assert (
        np.sum(np.abs(np.array(list(detected_result[0].boundingbox)) - true_bbox[0])) < 1e-3
    ), f"bbox is different from expected value: {true_bbox[0]}"
    assert (
        np.sum(np.abs(np.array(list(detected_result[1].boundingbox)) - true_bbox[1])) < 1e-3
    ), f"bbox is different from expected value {true_bbox[1]}"
    assert (
        np.sum(
            np.abs(np.array([detected_result[0].score, detected_result[1].score]) - true_confidence)
        )
        < 1e-3
    ), "confidence is different from expected value"
