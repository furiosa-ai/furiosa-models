from pathlib import Path

import cv2
import numpy as np

from furiosa.models.vision import YOLOv5l
from furiosa.models.vision.postprocess import collate
from furiosa.runtime.sync import create_runner

TEST_IMAGE_PATH = str(Path(__file__).parent / "../assets/yolov5-test.jpg")

NUM_BATCHES = 2
NUM_DETECTED_BOXES = 27


def test_yolov5_large_batched():
    m = YOLOv5l()

    batch_im = [cv2.imread(TEST_IMAGE_PATH), cv2.imread(TEST_IMAGE_PATH)]
    with create_runner(m.model_source()) as runner:
        batch_pre_img, batch_preproc_param = m.preprocess(batch_im)
        batch_feat = []
        for pre_image in batch_pre_img:
            batch_feat.append(runner.run(np.expand_dims(pre_image, axis=0)))
        batch_feat = collate(batch_feat)
        detected_boxes = m.postprocess(batch_feat, batch_preproc_param)

        assert len(detected_boxes) == NUM_BATCHES, f"batch axis is expected {NUM_BATCHES}"
        assert (
            len(detected_boxes[0]) == NUM_DETECTED_BOXES
        ), f"detected_boxes is expected {NUM_DETECTED_BOXES}, got {len(detected_boxes[0])}"
