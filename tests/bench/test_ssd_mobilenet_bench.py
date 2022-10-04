from itertools import cycle
import os
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

from furiosa.models.vision import SSDMobileNet
from furiosa.models.vision.postprocess import ObjectDetectionResult
from furiosa.models.vision.ssd_mobilenet import NativePostProcessor, postprocess, preprocess
from furiosa.runtime import session


def load_coco_from_env_variable():
    coco_val_images = os.environ.get('COCO_VAL_IMAGES')
    coco_val_labels = os.environ.get('COCO_VAL_LABELS')

    if coco_val_images is None or coco_val_labels is None:
        raise Exception("Environment variables not set")

    coco = COCO(coco_val_labels)

    return Path(coco_val_images), coco


def workload_python(sess: session, image):
    image, contexts = preprocess([image])
    output = sess.run(image).numpy()
    postprocess(output, contexts, confidence_threshold=0.3)


def test_ssd_mobilenet_benchmark(benchmark):
    model = SSDMobileNet()

    image_directory, coco = load_coco_from_env_variable()

    with session.create(model.enf) as sess:
        image_src_iter = cycle(coco.dataset["images"])

        def read_image():
            image_src = next(image_src_iter)
            image_path = str(image_directory / image_src["file_name"])
            image = cv2.imread(image_path)

            return (sess, image), {}

        benchmark.pedantic(workload_python, setup=read_image, rounds=1000)
