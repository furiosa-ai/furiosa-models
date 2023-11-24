import cv2

from furiosa.models.vision import YOLOv7w6Pose
from furiosa.runtime.sync import create_runner

yolo_pose = YOLOv7w6Pose()

with create_runner(yolo_pose.model_source()) as runner:
    image = cv2.imread("tests/assets/pose_demo.jpg")
    inputs, contexts = yolo_pose.preprocess([image])
    output = runner.run(inputs)
    results = yolo_pose.postprocess(output, contexts=contexts)
    yolo_pose.visualize(image, results[0])
    cv2.imwrite("./pose_result.jpg", image)
