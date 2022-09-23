import os
from pathlib import Path
from typing import Tuple

import numpy as np
from test_acc_util import bdd100k
from tqdm import tqdm

from furiosa.models.vision import YOLOv5m
from furiosa.models.vision.postprocess import collate
from furiosa.models.vision.yolov5 import medium as yolov5m
from furiosa.registry import Model
from furiosa.runtime import session

EXPECTED_MAP = 0.279543358502077  # matches e2e-testing's map exactly


def load_db_from_env_variable() -> Tuple[Path, bdd100k.Yolov5Dataset]:
    MUST_10K_LIMIT = 10000
    databaset_path = os.environ.get('YOLOV5_DATASET_PATH', "./tests/data/bdd100k_val")

    if databaset_path is None:
        raise Exception("Environment variables not set: YOLOV5_DATASET_PATH")

    databaset_path = Path(databaset_path)
    db = bdd100k.Yolov5Dataset(databaset_path, mode="val", limit=MUST_10K_LIMIT)

    return databaset_path, db


def test_yolov5m_accuracy():
    model: Model = YOLOv5m()

    image_directory, yolov5db = load_db_from_env_variable()

    print(f"dataset_path: {image_directory}")
    metric = bdd100k.MAPMetricYolov5(num_classes=len(yolov5m.CLASSES))
    with session.create(
        model.source, compile_config=model.compile_config(model_input_format='hwc')
    ) as sess:
        for im, boxes_target, classes_target in tqdm(yolov5db):
            batch_im = [im]

            batch_pre_img, batch_preproc_param = yolov5m.preprocess(
                batch_im, input_color_format="bgr"
            )  # single-batch
            batch_feat = sess.run(np.expand_dims(batch_pre_img[0], axis=0)).numpy()
            detected_boxes = yolov5m.postprocess(
                batch_feat, batch_preproc_param, conf_thres=0.001, iou_thres=0.6
            )

            det_out = bdd100k.to_numpy(detected_boxes[0])

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
    assert abs(result['map'] - EXPECTED_MAP) < 1e-3, "Accuracy check failed"
