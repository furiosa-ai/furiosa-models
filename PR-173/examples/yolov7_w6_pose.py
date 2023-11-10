import cv2
import numpy as np

from furiosa.models.vision import YOLOv7w6Pose
from furiosa.runtime.sync import create_runner

yolov5m = YOLOv7w6Pose()

with create_runner(yolov5m.model_source()) as runner:
    image = cv2.imread("tests/assets/yolov5-test.jpg")
    inputs, contexts = yolov5m.preprocess([image])
    output = runner.run(np.expand_dims(inputs[0], axis=0))
    yolov5m.postprocess(output, contexts=contexts)
