# YOLOv5L

YOLOv5 is the one of the most popular object detection models.
You can find more details at [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5).

<div align="center">
    <img src="../../images/yolov5l_demo.jpg" title="YOLOv5l Example" width="640" />
</div>

## Overall
* Framework: PyTorch
* Model format: ONNX
* Model task: Object Detection
* Source: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

## <a name="YOLOv5l_Usage"></a>
!!!Usage
    ```python
    --8<-- "docs/examples/yolov5l.py"
    ```

## Inputs
The input is a 3-channel image of 640, 640 (height, width).

* Data Type: `numpy.uint8`
* Tensor Shape: `[1, 640, 640, 3]`
* Memory Format: NHWC, where
    * N - batch size
    * H - image height
    * W - image width
    * C - number of channels
* Color Order: RGB
* Optimal Batch Size (minimum: 1): <= 2

## Outputs
The outputs are 3 `numpy.float32` tensors in various shapes as the following. 
You can refer to `postprocess()` function to learn how to decode boxes, classes, and confidence scores.

| Tensor | Shape             | Data Type | Data Type | Description |
|--------|-------------------|-----------|-----------|-------------|
| 0      | (1, 45, 80, 80)   | float32   | NCHW      |             |
| 1      | (1, 45, 40, 40)   | float32   | NCHW      |             |
| 2      | (1, 45, 20, 20)   | float32   | NCHW      |             |
 

## Pre/Postprocessing
`furiosa.models.vision.YOLOv5l` class provides `preprocess` and `postprocess` methods.
`preprocess` method converts input images to input tensors, and `postprocess` method converts 
model output tensors to a list of bounding boxes, scores and labels. 
You can find examples at [YOLOv5l Usage](#YOLOv5l_Usage).
 
### `furiosa.models.vision.YOLOv5l.preprocess`
::: furiosa.models.vision.yolov5.core.YOLOv5PreProcessor.__call__
    options:
        show_source: false
    
### `furiosa.models.vision.YOLOv5l.postprocess`
::: furiosa.models.vision.yolov5.core.YOLOv5PostProcessor.__call__
    options:
        show_source: false
