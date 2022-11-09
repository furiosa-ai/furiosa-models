import cv2
import numpy as np

from furiosa.models.vision import YOLOv5l
from furiosa.runtime import session

yolov5l = YOLOv5l.load()

with session.create(yolov5l) as sess:
    image = cv2.imread("tests/assets/yolov5-test.jpg")
    inputs, contexts = yolov5l.preprocess([image])
    output = sess.run(np.expand_dims(inputs[0], axis=0)).numpy()
    yolov5l.postprocess(output, contexts=contexts)
