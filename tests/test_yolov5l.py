# from dataclasses import astuple

import cv2

# from helpers.util import draw_bboxes
import numpy as np
import pytest

from furiosa.models.vision import nonblocking
from furiosa.models.vision.postprocess import collate
from furiosa.models.vision.yolov5.large import CLASSES, postprocess, preprocess
from furiosa.runtime import session


@pytest.mark.asyncio
async def test_yolov5_large():
    m = await nonblocking.YOLOv5l()
    test_image_path = "tests/assets/yolov5-test.jpg"

    assert len(CLASSES) == 10, "expected CLASS is 10"

    batch_im = [cv2.imread(test_image_path), cv2.imread(test_image_path)]
    try:
        sess = session.create(m.source, compile_config=m.compile_config(model_input_format='hwc'))
    except Exception as e:
        print("Exception", e)
    batch_pre_img, batch_preproc_param = preprocess(batch_im, input_color_format="bgr")
    batch_feat = []
    for pre_image in batch_pre_img:
        batch_feat.append(sess.run(np.expand_dims(pre_image, axis=0)).numpy())

    batch_feat = collate(batch_feat)
    detected_boxes = postprocess(batch_feat, batch_preproc_param)
    assert len(detected_boxes) == 2, "batch axis is expected 2"
    # im_out = draw_bboxes(batch_im[0], list(map(astuple, detected_boxes[0])))
    # cv2.imwrite("yolov5l.jpg", im_out)
    assert len(detected_boxes[0]) == 26, f"detected_boxes must be 26, got {len(detected_boxes[0])}"
    sess.close()
