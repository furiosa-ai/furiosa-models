import os
from pathlib import Path
from typing import Tuple

import numpy as np
from test_acc_util import bdd100k
from tqdm import tqdm

from furiosa.models.vision import YOLOv5l
from furiosa.models.vision.postprocess import collate
from furiosa.models.vision.yolov5 import large as yolov5l
from furiosa.registry import Model
from furiosa.runtime import session

EXPECTED_MAP = 0.29551345515187283  # matches e2e-testing's map exactly


def load_db_from_env_variable() -> Tuple[Path, bdd100k.Yolov5Dataset]:
    MUST_LIMIT_TO_VALIDATION = 1000
    databaset_path = os.environ.get('YOLOV5_DATASET_PATH')

    if databaset_path is None:
        raise Exception("Environment variables not set: YOLOV5_DATASET_PATH")

    databaset_path = Path(databaset_path)
    db = bdd100k.Yolov5Dataset(databaset_path, mode="val", limit=MUST_LIMIT_TO_VALIDATION)

    return databaset_path, db


def test_yolov5l_accuracy():
    model: Model = YOLOv5l()

    image_directory, yolov5db = load_db_from_env_variable()

    print(f"dataset_path: {image_directory}")
    metric = bdd100k.MAPMetricYolov5(num_classes=len(yolov5l.CLASSES))
    with session.create(
        model.source, compile_config=model.compile_config(model_input_format='hwc')
    ) as sess:
        for im, boxes_target, classes_target in tqdm(yolov5db):
            batch_im = [im]

            batch_pre_img, batch_preproc_param = yolov5l.preprocess(
                batch_im, input_color_format="bgr"
            )
            batch_feat = []
            for pre_image in batch_pre_img:
                batch_feat.append(sess.run(np.expand_dims(pre_image, axis=0)).numpy())

            batch_feat = collate(batch_feat)
            detected_boxes = yolov5l.postprocess(
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
    print("YOLOv5Large mAP:", result['map'])
    print("YOLOv5Large mAP50:", result['map50'])
    print("YOLOv5Large ap_class:", result['ap_class'])
    print("YOLOv5Large ap50_class:", result['ap50_class'])
    assert abs(result['map'] - EXPECTED_MAP) < 1e-2, "Accuracy check failed"
