import cv2
import numpy as np

from furiosa.models.vision import YOLOv5m
from furiosa.models.vision.yolov5.medium import postprocess, preprocess
from furiosa.runtime import session

model = YOLOv5m.load()

with session.create(model) as sess:
    image = cv2.imread("tests/assets/yolov5-test.jpg")
    inputs, context = preprocess([image], color_format="bgr")
    output = sess.run(np.expand_dims(inputs[0], axis=0)).numpy()
    postprocess(output, context=context)
