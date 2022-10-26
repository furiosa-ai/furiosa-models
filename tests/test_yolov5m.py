# from dataclasses import astuple

import cv2

# from helpers.util import draw_bboxes
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
    compiler_config = m.compiler_config(model_input_format='hwc')
    sess = session.create(m, compiler_config=compiler_config)
    batch_pre_img, batch_preproc_param = preprocess(batch_im, input_color_format="bgr")
    batch_feat = []
    for pre_image in batch_pre_img:
        batch_feat.append(sess.run(np.expand_dims(pre_image, axis=0)).numpy())

    batch_feat = collate(batch_feat)
    detected_boxes = postprocess(batch_feat, batch_preproc_param)
    assert len(detected_boxes) == 2, "batch axis is expected 2"
    # im_out = draw_bboxes(batch_im[0], list(map(astuple, detected_boxes[0])))
    # cv2.imwrite("yolov5m.jpg", im_out)
    assert len(detected_boxes[0]) == 18, "detected_boxes must be 18"
    sess.close()
