import cv2

from furiosa.models.vision import YOLOv5l
from furiosa.models.vision.yolov5.medium import postprocess, preprocess
from furiosa.runtime import session

model = YOLOv5l.load()

with session.create(model) as sess:
    image = cv2.imread("tests/assets/yolov5-test.jpg")
    input, context = preprocess([image], color_format="bgr")
    output = sess.run(input).numpy()
    postprocess(output, context=context)
