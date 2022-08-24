# from dataclasses import astuple

import cv2
import numpy as np
import pytest

from furiosa.models.vision import nonblocking
from furiosa.models.vision.postprocess import collate_data
from furiosa.models.vision.yolov5.large import (
    CLASSES,
    get_anchor_per_layer_count,
    postprocess,
    preprocess,
)
from furiosa.runtime import session

# from .helpers.util import draw_bboxes


@pytest.mark.asyncio
async def test_mlcommons_mobilessd_small_perf():
    m = await nonblocking.YOLOv5l()
    test_image_path = "scripts/assets/yolov5-test.jpg"

    assert len(CLASSES) == 10, "expected CLASS is 10"

    batch_im = [cv2.imread(test_image_path), cv2.imread(test_image_path)]
    sess = session.create(m.model, compile_config=m.compile_config(model_input_format='hwc'))
    batch_pre_img, batch_preproc_param = preprocess(batch_im, input_color_format="bgr")
    batch_feat = []
    for pre_img in batch_pre_img:
        feat = sess.run(np.expand_dims(pre_img, axis=0)).numpy()
        batch_feat.append(feat)
    batch_feat = collate_data(batch_feat, get_anchor_per_layer_count())
    detected_boxes = postprocess(batch_feat, batch_preproc_param)
    assert len(detected_boxes) == 2, "batch axis is expected 2"
    # im_out = draw_bboxes(batch_im[0], list(map(astuple, detected_boxes[0])))
    # cv2.imwrite("yolov5l.jpg", im_out)
    assert len(detected_boxes[0]) == 27, "detected_boxes must be 27"
    sess.close()
