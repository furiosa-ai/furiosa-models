import asyncio
import itertools
import os
from pathlib import Path
from typing import Tuple

import tqdm

from furiosa.models.vision import YOLOv5m
from furiosa.runtime import create_runner

from ..bench.test_acc_util import bdd100k

EXPECTED_MAP = 0.27702783413351617
EXPECTED_MAP_RUST = 0.2769884679629229

CONF_THRES = 0.001
IOU_THRES = 0.45


def load_db_from_env_variable() -> Tuple[Path, bdd100k.Yolov5Dataset]:
    MUST_10K_LIMIT = 10_000
    databaset_path = Path(os.environ.get('YOLOV5_DATASET_PATH', "./tests/data/bdd100k_val"))

    db = bdd100k.Yolov5Dataset(databaset_path, mode="val", limit=MUST_10K_LIMIT)

    return databaset_path, db


async def test_yolov5m_accuracy():
    model: YOLOv5m = YOLOv5m(postprocessor_type="Python")

    image_directory, yolov5db = load_db_from_env_variable()

    print(f"dataset_path: {image_directory}")
    metric = bdd100k.MAPMetricYolov5(num_classes=10)

    num_images = len(yolov5db)
    yolov5db = iter(yolov5db)

    async def workload(im):
        batch_im = [im]

        batch_pre_img, batch_preproc_param = model.preprocess(batch_im)
        batch_feat = await runner.run(batch_pre_img)
        detected_boxes = model.postprocess(
            batch_feat, batch_preproc_param, conf_thres=CONF_THRES, iou_thres=IOU_THRES
        )

        return bdd100k.to_numpy(detected_boxes[0])

    async with create_runner(model.model_source(num_pe=1), device="warboy(1)*1") as runner:
        steps = 10
        assert num_images % steps == 0, "cannot divide by step"
        iters = num_images // steps
        for _ in tqdm.tqdm(range(steps), desc="yolov5m accuracy w/ python pp"):
            worklist = []
            bxtargets = []
            clstargets = []
            for im, boxes_target, classes_target in itertools.islice(yolov5db, iters):
                bxtargets.append(boxes_target)
                clstargets.append(classes_target)
                worklist.append(workload(im))
            for det_out, boxes_target, classes_target in zip(
                await asyncio.gather(*worklist), bxtargets, clstargets
            ):
                metric(
                    boxes_pred=det_out[:, :4],
                    scores_pred=det_out[:, 4],
                    classes_pred=det_out[:, 5],
                    boxes_target=boxes_target,
                    classes_target=classes_target,
                )

    result = metric.compute()
    print("YOLOv5Medium mAP:", result['map'])
    print("YOLOv5Medium mAP50:", result['map50'])
    print("YOLOv5Medium ap_class:", result['ap_class'])
    print("YOLOv5Medium ap50_class:", result['ap50_class'])

    assert result['map'] == EXPECTED_MAP, "Accuracy check w/ python failed"


async def test_yolov5m_with_native_rust_pp_accuracy():
    model: YOLOv5m = YOLOv5m(postprocessor_type="Rust")

    image_directory, yolov5db = load_db_from_env_variable()

    print(f"dataset_path: {image_directory}")
    metric = bdd100k.MAPMetricYolov5(num_classes=10)

    num_images = len(yolov5db)
    yolov5db = iter(yolov5db)

    async def workload(im):
        batch_im = [im]

        batch_pre_img, batch_preproc_param = model.preprocessor.__call__(batch_im)
        batch_feat = await runner.run(batch_pre_img)
        detected_boxes = model.postprocessor.__call__(
            batch_feat, batch_preproc_param, conf_thres=CONF_THRES, iou_thres=IOU_THRES
        )

        return bdd100k.to_numpy(detected_boxes[0])

    async with create_runner(model.model_source(num_pe=1), device="warboy(1)*1") as runner:
        steps = 10
        assert num_images % steps == 0, "cannot divide by step"
        iters = num_images // steps
        for _ in tqdm.tqdm(range(steps), desc="yolov5m accuracy w/ rust pp"):
            worklist = []
            bxtargets = []
            clstargets = []
            for im, boxes_target, classes_target in itertools.islice(yolov5db, iters):
                bxtargets.append(boxes_target)
                clstargets.append(classes_target)
                worklist.append(workload(im))
            for det_out, boxes_target, classes_target in zip(
                await asyncio.gather(*worklist), bxtargets, clstargets
            ):
                metric(
                    boxes_pred=det_out[:, :4],
                    scores_pred=det_out[:, 4],
                    classes_pred=det_out[:, 5],
                    boxes_target=boxes_target,
                    classes_target=classes_target,
                )

    result = metric.compute()
    print("YOLOv5Medium mAP:", result['map'])
    print("YOLOv5Medium mAP50:", result['map50'])
    print("YOLOv5Medium ap_class:", result['ap_class'])
    print("YOLOv5Medium ap50_class:", result['ap50_class'])

    assert result['map'] == EXPECTED_MAP_RUST, "Accuracy check w/ rust failed"
