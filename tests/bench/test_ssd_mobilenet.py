import os
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tqdm

from furiosa.models.types import Model
from furiosa.models.vision import SSDMobileNet
from furiosa.runtime import session

EXPECTED_ACCURACY = 0.23199896825191885
EXPECTED_ACCURACY_NATIVE_RUST_PP = 0.23178397430922199


def load_coco_from_env_variable():
    coco_val_images = os.environ.get('COCO_VAL_IMAGES', 'tests/data/coco/val2017')
    coco_val_labels = os.environ.get(
        'COCO_VAL_LABELS', 'tests/data/coco/annotations/instances_val2017.json'
    )
    coco = COCO(coco_val_labels)

    return Path(coco_val_images), coco


def test_mlcommons_ssd_mobilenet_accuracy(benchmark):
    model: Model = SSDMobileNet.load(use_native=False)

    image_directory, coco = load_coco_from_env_variable()
    image_src_iter = iter(tqdm.tqdm(coco.dataset["images"]))
    num_images = len(coco.dataset["images"])

    detections = []

    def read_image():
        image_src = next(image_src_iter)
        image_path = str(image_directory / image_src["file_name"])
        image = cv2.imread(image_path)

        return (image_src["id"], image), {}

    def workload(image_id, image):
        image, contexts = model.preprocess([image])
        outputs = sess.run(image).numpy()
        batch_result = model.postprocess(outputs, contexts, confidence_threshold=0.3)
        result = np.squeeze(batch_result, axis=0)  # squeeze the batch axis

        for res in result:
            detection = {
                "image_id": image_id,
                "category_id": res.index,
                "bbox": [
                    res.boundingbox.left,
                    res.boundingbox.top,
                    (res.boundingbox.right - res.boundingbox.left),
                    (res.boundingbox.bottom - res.boundingbox.top),
                ],
                "score": res.score,
            }
            detections.append(detection)

    sess = session.create(model.enf)
    benchmark.pedantic(workload, setup=read_image, rounds=num_images)
    sess.close()

    coco_detections = coco.loadRes(detections)
    coco_eval = COCOeval(coco, coco_detections, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("mAP:", coco_eval.stats[0])

    assert coco_eval.stats[0] == EXPECTED_ACCURACY, "Accuracy check failed"


def test_mlcommons_ssd_mobilenet_with_native_rust_pp_accuracy(benchmark):
    model = SSDMobileNet.load(use_native=True)

    image_directory, coco = load_coco_from_env_variable()
    image_src_iter = iter(tqdm.tqdm(coco.dataset["images"]))
    num_images = len(coco.dataset["images"])

    detections = []

    def read_image():
        image_src = next(image_src_iter)
        image_path = str(image_directory / image_src["file_name"])
        image = cv2.imread(image_path)

        return (image_src["id"], image), {}

    def workload(image_id, image):
        image, contexts = model.preprocess([image])
        outputs = sess.run(image).numpy()
        result = model.postprocess(outputs, contexts[0])

        for res in result:
            detection = {
                "image_id": image_id,
                "category_id": res.index,
                "bbox": [
                    res.boundingbox.left,
                    res.boundingbox.top,
                    (res.boundingbox.right - res.boundingbox.left),
                    (res.boundingbox.bottom - res.boundingbox.top),
                ],
                "score": res.score,
            }
            detections.append(detection)

    sess = session.create(model.enf)
    benchmark.pedantic(workload, setup=read_image, rounds=num_images)
    sess.close()

    coco_detections = coco.loadRes(detections)
    coco_eval = COCOeval(coco, coco_detections, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("mAP:", coco_eval.stats[0])
    assert coco_eval.stats[0] == EXPECTED_ACCURACY_NATIVE_RUST_PP, "Accuracy check failed"
