# YOLOv7w6Pose

YOLOv7w6 Pose Estimation Model.
You can find more details at [https://github.com/WongKinYiu/yolov7#pose-estimation](https://github.com/WongKinYiu/yolov7#pose-estimation).

## Overall

* Framework: PyTorch
* Model format: ONNX
* Model task: Pose Estimation
* Source: [https://github.com/WongKinYiu/yolov7#pose-estimation](https://github.com/WongKinYiu/yolov7#pose-estimation).

## <a name="YOLOv7w6Pose_Usage"></a>
!!!Usage
    ```python
    --8<-- "docs/examples/yolov7_w6_pose.py"
    ```

## Inputs
The input is a 3-channel image of 384, 640 (height, width).

* Data Type: `numpy.uint8`
* Tensor Shape: `[1, 3, 384, 640]`
* Memory Format: NHWC, where
    * N - batch size
    * C - number of channels
    * H - image height
    * W - image width
* Color Order: RGB
* Optimal Batch Size (minimum: 1): <= 4

## Outputs
The outputs are 3 `numpy.float32` tensors in various shapes as the following.
You can refer to `postprocess()` function to learn how to decode boxes, classes, and confidence scores.

| Tensor | Shape             | Data Type | Data Type | Description |
|--------|-------------------|-----------|-----------|-------------|
| 0      | (1, 18, 48, 80)   | float32   | NCHW      |             |
| 1      | (1, 153, 48, 80)  | float32   | NCHW      |             |
| 2      | (1, 18, 24, 40)   | float32   | NCHW      |             |
| 3      | (1, 153, 24, 40)  | float32   | NCHW      |             |
| 4      | (1, 18, 12, 20)   | float32   | NCHW      |             |
| 5      | (1, 153, 12, 20)  | float32   | NCHW      |             |
| 6      | (1, 18, 6, 10)    | float32   | NCHW      |             |
| 7      | (1, 153, 6, 10)   | float32   | NCHW      |             |

## Pre/Postprocessing

`furiosa.models.vision.YOLOv7w6Pose` class provides `preprocess` and `postprocess` methods.
`preprocess` method converts input images to input tensors, and `postprocess` method converts
model output tensors to a list of `PoseEstimationResult`.
You can find examples at [YOLOv7w6Pose Usage](#YOLOv7w6Pose_Usage).

### `furiosa.models.vision.YOLOv7w6Pose.preprocess`

::: furiosa.models.vision.yolov7_w6_pose.YOLOv7w6PosePreProcessor.__call__
    options:
        show_source: false


### `furiosa.models.vision.YOLOv7w6Pose.postprocess`

::: furiosa.models.vision.yolov7_w6_pose.postprocess.YOLOv7w6PosePostProcessor.__call__
    options:
        show_source: false

Postprocess output tensors to a list of `PoseEstimationResult`. It transforms the model's output into a list of `PoseEstimationResult` instances.
Each `PoseEstimationResult` contains information about the overall pose, including a bounding box, confidence score, and keypoint details such as nose, eyes, shoulders, etc.
Please refer to the followings for more details.

### `Keypoint`

The `Keypoint` class represents a keypoint detected by the YOLOv7W6 Pose Estimation model. It contains the following attributes:

| Attribute      | Description                                                               |
|----------------|---------------------------------------------------------------------------|
| `x`            | The x-coordinate of the keypoint as a floating-point number.              |
| `y`            | The y-coordinate of the keypoint as a floating-point number.              |
| `confidence`   | Confidence score associated with the keypoint as a floating-point number. |


See the source code for more details.

::: furiosa.models.vision.yolov7_w6_pose.postprocess.Keypoint

### `PoseEstimationResult`

The `PoseEstimationResult` class represents the overall result of the YOLOv7W6 Pose Estimation model. It includes the following attributes:

| Attribute          | Description                                                                                           |
|--------------------|-------------------------------------------------------------------------------------------------------|
| `bounding_box`     | A list of four floating-point numbers representing the bounding box coordinates of the detected pose. |
| `confidence`       | Confidence score associated with the overall pose estimation as a floating-point number.              |
| `nose`             | Instance of the `Keypoint` class representing the nose keypoint.                                      |
| `left_eye`         | Instance of the `Keypoint` class representing the left eye keypoint.                                  |
| `right_eye`        | Instance of the `Keypoint` class representing the right eye keypoint.                                 |
| `left_ear`         | Instance of the `Keypoint` class representing the left ear keypoint.                                  |
| `right_ear`        | Instance of the `Keypoint` class representing the right ear keypoint.                                 |
| `left_shoulder`    | Instance of the `Keypoint` class representing the left shoulder keypoint.                             |
| `right_shoulder`   | Instance of the `Keypoint` class representing the right shoulder keypoint.                            |
| `left_elbow`       | Instance of the `Keypoint` class representing the left elbow keypoint.                                |
| `right_elbow`      | Instance of the `Keypoint` class representing the right elbow keypoint.                               |
| `left_wrist`       | Instance of the `Keypoint` class representing the left wrist keypoint.                                |
| `right_wrist`      | Instance of the `Keypoint` class representing the right wrist keypoint.                               |
| `left_hip`         | Instance of the `Keypoint` class representing the left hip keypoint.                                  |
| `right_hip`        | Instance of the `Keypoint` class representing the right hip keypoint.                                 |
| `left_knee`        | Instance of the `Keypoint` class representing the left knee keypoint.                                 |
| `right_knee`       | Instance of the `Keypoint` class representing the right knee keypoint.                                |
| `left_ankle`       | Instance of the `Keypoint` class representing the left ankle keypoint.                                |
| `right_ankle`      | Instance of the `Keypoint` class representing the right ankle keypoint.                               |

See the source code for more details.

::: furiosa.models.vision.yolov7_w6_pose.postprocess.PoseEstimationResult
    options:
        show_source: true
