# from dataclasses import astuple

import cv2
import numpy as np
import pytest

from furiosa.models.vision import YOLOv5l
from furiosa.models.vision.postprocess import collate
from furiosa.runtime import session


def create_session_scenario(scenario: str, m):
    if scenario == "enf":
        return session.create(model=m.enf)
    elif scenario == "onnx":
        return session.create(model=m.source, compiler_config=m.compiler_config)
    elif scenario == "model":
        return session.create(model=m)


@pytest.mark.parametrize("session_test_scenario", ["enf", "onnx", "model"])
@pytest.mark.parametrize(
    "test_image_path,expected_batch_axis,expected_zero_index_detected_box",
    [("tests/assets/yolov5-test.jpg", 2, 27)],
)
@pytest.mark.asyncio
async def test_yolov5_large(
    session_test_scenario: str,
    test_image_path: str,
    expected_batch_axis: int,
    expected_zero_index_detected_box: int,
):
    print(f"Test Session Create Scenario: {session_test_scenario}")
    m = await YOLOv5l.load_async()
    assert len(m.classes) == 10, "expected CLASS is 10"

    batch_im = [cv2.imread(test_image_path), cv2.imread(test_image_path)]
    sess = create_session_scenario(session_test_scenario, m)
    batch_pre_img, batch_preproc_param = m.preprocess(batch_im, color_format="bgr")
    batch_feat = []
    for pre_image in batch_pre_img:
        batch_feat.append(sess.run(np.expand_dims(pre_image, axis=0)).numpy())

    batch_feat = collate(batch_feat)
    detected_boxes = m.postprocess(batch_feat, batch_preproc_param)
    assert (
        len(detected_boxes) == expected_batch_axis
    ), f"batch axis is expected {expected_batch_axis}"
    assert (
        len(detected_boxes[0]) == expected_zero_index_detected_box
    ), f"detected_boxes is expected {expected_zero_index_detected_box}, got {len(detected_boxes[0])}"
    sess.close()
