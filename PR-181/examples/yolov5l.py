import cv2
import numpy as np

from furiosa.models.vision import YOLOv5l
from furiosa.runtime.sync import create_runner

yolov5l = YOLOv5l()

with create_runner(yolov5l.model_source()) as runner:
    image = cv2.imread("tests/assets/yolov5-test.jpg")
    inputs, contexts = yolov5l.preprocess([image])
    output = runner.run(np.expand_dims(inputs[0], axis=0))
    yolov5l.postprocess(output, contexts=contexts)
