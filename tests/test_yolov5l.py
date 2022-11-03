# from dataclasses import astuple

from typing import Callable

import cv2
import numpy as np
import pytest

from furiosa.models.vision import YOLOv5l
from furiosa.models.vision.postprocess import collate
from furiosa.models.vision.yolov5.large import CLASSES, postprocess, preprocess
from furiosa.runtime import session


def create_enf_source_session(m):
    print("create enf source session")
    return session.create(model=m.enf)

def create_onxx_source_session(m):
    print("create onnx source session")
    return session.create(model=m.source, compiler_config=m.compiler_config)

def create_model_type_session(m):
    print("create model type session")
    return session.create(model=m)

@pytest.fixture(params=["enf", "onnx", "model"])
def session_senario(request):
    if request.param == "enf":
        return create_enf_source_session
    elif request.param == "onnx":
        return create_onxx_source_session
    elif request.param == "model":
        return create_model_type_session

@pytest.mark.parametrize("test_image_path,expected_detected_box,expected_zero_index_detected_box", 
                         [("tests/assets/yolov5-test.jpg", 2, 27)])
@pytest.mark.asyncio
async def test_yolov5_large(session_senario: Callable, test_image_path: str, expected_detected_box: int, expected_zero_index_detected_box: int):
    m = await YOLOv5l.load_async()
    assert len(CLASSES) == 10, "expected CLASS is 10"

    batch_im = [cv2.imread(test_image_path), cv2.imread(test_image_path)]
    sess = session_senario(m)
    batch_pre_img, batch_preproc_param = preprocess(batch_im, input_color_format="bgr")
    batch_feat = []
    for pre_image in batch_pre_img:
        batch_feat.append(sess.run(np.expand_dims(pre_image, axis=0)).numpy())

    batch_feat = collate(batch_feat)
    detected_boxes = postprocess(batch_feat, batch_preproc_param)
    assert len(detected_boxes) == expected_detected_box, f"batch axis is expected {expected_detected_box}"
    assert len(detected_boxes[0]) == expected_zero_index_detected_box, f"detected_boxes is expected {expected_zero_index_detected_box}, got {len(detected_boxes[0])}"
    sess.close()
