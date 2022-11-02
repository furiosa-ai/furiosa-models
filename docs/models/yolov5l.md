# YOLOv5L

YOLOv5 is the one of the most popular object detection models developed by [Ultralytics](https://ultralytics.com/).
You can find more details at https://github.com/ultralytics/yolov5.

## Overall
* Framework: PyTorch
* Model format: ONNX
* Model task: Object Detection
* Source: This model is originated from https://github.com/ultralytics/yolov5

!!!Usage
    ```python
    --8<-- "docs/examples/yolov5l.py"
    ```

## Model inputs
The input is a 3-channel image of 640, 640 (height, width).

* Data Type: `numpy.uint8`
* Tensor Shape: `[1, 640, 640, 3]`
* Memory Format: NCHW, where
    * N - batch size
    * C - number of channels
    * H - image height
    * W - image width
* Optimal Batch Size: 1

## Outputs
The outputs are 3 `numpy.float32` tensors in various shapes as the following. 
You can refer to `postprocess()` function to learn how to decode boxes, classes, and confidence scores.

| Tensor | Shape             | Data Type | Data Type | Description |
|--------|-------------------|-----------|-----------|-------------|
| 0      | (1, 45, 80, 80)   | float32   | NCHW      |             |
| 1      | (1, 45, 40, 40)   | float32   | NCHW      |             |
| 2      | (1, 45, 20, 20)   | float32   | NCHW      |             |

## Pre/Post processing
`furiosa.models.vision.yolov5.large` module provides a set of utilities 
to convert images to input tensors and the model outputs to object detection results.
  
### `furiosa.models.vision.yolov5.large.preprocess`
::: furiosa.models.vision.yolov5.large.preprocess
    options:
        show_root_heading: false
        show_sources: false
### `furiosa.models.vision.yolov5.large.postprocess`
::: furiosa.models.vision.yolov5.large.postprocess
    options:
        show_root_heading: false
        show_sources: false
