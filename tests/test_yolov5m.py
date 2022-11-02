# from dataclasses import astuple

import cv2
import numpy as np
import pytest

from furiosa.models.vision import YOLOv5m
from furiosa.models.vision.postprocess import collate
from furiosa.models.vision.yolov5.medium import CLASSES, postprocess, preprocess
from furiosa.runtime import session


@pytest.mark.asyncio
async def test_yolov5_small():
    m = await YOLOv5m.load_async()
    test_image_path = "tests/assets/yolov5-test.jpg"

    assert len(CLASSES) == 10, "expected CLASS is 10"

    batch_im = [cv2.imread(test_image_path), cv2.imread(test_image_path)]
    sess = session.create(m.source, compile_config=m.compile_config(model_input_format='hwc'))
    batch_pre_img, batch_preproc_param = preprocess(batch_im, color_format="bgr")
    batch_feat = []
    for pre_image in batch_pre_img:
        batch_feat.append(sess.run(np.expand_dims(pre_image, axis=0)).numpy())

    batch_feat = collate(batch_feat)
    detected_boxes = postprocess(batch_feat, batch_preproc_param)
    assert len(detected_boxes) == 2, "batch axis is expected 2"
    assert len(detected_boxes[0]) == 18, "detected_boxes must be 18"
    sess.close()
