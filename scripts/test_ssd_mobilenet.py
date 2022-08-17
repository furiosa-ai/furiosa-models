import numpy as np
import pytest

from furiosa.models.vision import SSDMobileNet
from furiosa.models.vision.postprocess import collate_data
from furiosa.models.vision.ssd_mobilenet import CLASSES, NUM_OUTPUTS, postprocess, preprocess
from furiosa.registry import Model
from furiosa.runtime import session


@pytest.mark.asyncio
async def test_mlcommons_mobilessd_small_perf():
    m: Model = SSDMobileNet()
    test_image_path = "scripts/assets/cat.jpg"

    assert len(CLASSES) == 92, f"Classes is 92, but {len(CLASSES)}"
    sess = session.create(m.model)
    true_bbox = np.array([[187.30786, 88.035324, 950.99646, 743.3290239999999]], dtype=np.float32)
    true_confidence = np.array([0.97390455], dtype=np.float32)
    # For batch mode test, simply read two identical images.
    batch_pre_image, batch_preproc_param = preprocess([test_image_path, test_image_path])
    batch_feat = []
    for pre_image in batch_pre_image:
        feat = sess.run(np.expand_dims(pre_image, axis=0)).numpy()
        assert len(feat) == NUM_OUTPUTS, f"model outputs expteds {NUM_OUTPUTS}"
        batch_feat.append(feat)
    batch_feat = collate_data(batch_feat, NUM_OUTPUTS, axis=0)
    detected_result = postprocess(batch_feat, batch_preproc_param, confidence_threshold=0.3)
    assert len(detected_result) == 2, "ssd_resnet34 output shape must be 2"
    detected_result = detected_result[0]  # due to duplicated input image

    assert len(detected_result) == 1, "detected object must be 1"
    assert np.array_equal(
        detected_result[0].label, 'cat'
    ), f"wrong classid: {detected_result[0].label}, expected cat"
    assert (
        np.sum(np.abs(np.array(list(detected_result[0].boundingbox)) - true_bbox)) < 1e-3
    ), f"bbox is different from expected value"
    assert (
        np.sum(np.abs(detected_result[0].score - true_confidence)) < 1e-3
    ), "confidence is different from expected value"
