import cv2

from furiosa.models.vision import YOLOv7w6Pose
from furiosa.runtime.sync import create_runner

yolo_pose = YOLOv7w6Pose()

with create_runner(yolo_pose.model_source()) as runner:
    image = cv2.imread("tests/assets/yolov5-test.jpg")
    inputs, contexts = yolo_pose.preprocess([image])
    output = runner.run(inputs)
    yolo_pose.postprocess(output, contexts=contexts)
