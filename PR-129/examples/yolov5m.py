import cv2
import numpy as np

from furiosa.models.vision import YOLOv5m
from furiosa.runtime import session

yolov5m = YOLOv5m.load()

with session.create(yolov5m) as sess:
    image = cv2.imread("tests/assets/yolov5-test.jpg")
    inputs, contexts = yolov5m.preprocess([image])
    output = sess.run(np.expand_dims(inputs[0], axis=0)).numpy()
    yolov5m.postprocess(output, contexts=contexts)
