from pathlib import Path

import cv2
import numpy as np

from furiosa.models.vision import YOLOv7w6Pose
from furiosa.runtime.sync import create_runner

cwd = Path(__file__).parent
yolo_pose = YOLOv7w6Pose()

with create_runner(yolo_pose.model_source()) as runner:
    image = cv2.imread("tests/assets/yolov5-test.jpg")
    inputs, contexts = yolo_pose.preprocess([image])
    output = runner.run(np.expand_dims(inputs[0], axis=0))
    yolo_pose.postprocess(output, contexts=contexts)
