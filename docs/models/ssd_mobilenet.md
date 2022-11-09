# SSD MobileNet v1

SSD MobileNet v1 backbone model trained on COCO (300x300).
This model has been used since MLCommons v0.5.

## Overall
* Framework: PyTorch
* Model format: ONNX
* Model task: Object detection
* Source: This model is originated from SSD MobileNet v1 in ONNX available at
[MLCommons - Supported Models](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#supported-models).

## <a name="SSDMobileNet_Usage"></a>
!!! Usages
    === "Python Postprocessor"
        ```python
        --8<-- "docs/examples/ssd_mobilenet.py"
        ```
     
    === "Native Postprocessor"
        ```python
        --8<-- "docs/examples/ssd_mobilenet_native.py"
        ```

## Inputs 
The input is a 3-channel image of 300x300 (height, width).

* Data Type: `numpy.float32`
* Tensor Shape: `[1, 3, 300, 300]`
* Memory Format: NCHW, where:
    * N - batch size
    * C - number of channels
    * H - image height
    * W - image width
* Color Order: RGB
* Optimal Batch Size (minimum: 1): <= 8

## Outputs
The outputs are 12 `numpy.float32` tensors in various shapes as the following.
You can refer to `postprocess()` function to learn how to decode boxes, classes, and confidence scores.

| Tensor | Shape            | Data Type | Data Type | Description |
|--------|------------------|-----------|-----------|-------------|
| 0      | (1, 273, 19, 19) | float32   | NCHW      |             |
| 1      | (1, 12, 19, 19)  | float32   | NCHW      |             |
| 2      | (1, 546, 10, 10) | float32   | NCHW      |             |
| 3      | (1, 24, 10, 10)  | float32   | NCHW      |             |
| 4      | (1, 546, 5, 5)   | float32   | NCHW      |             |
| 5      | (1, 24, 5, 5)    | float32   | NCHW      |             |
| 6      | (1, 546, 3, 3)   | float32   | NCHW      |             |
| 7      | (1, 24, 3, 3)    | float32   | NCHW      |             |
| 8      | (1, 546, 2, 2)   | float32   | NCHW      |             |
| 9      | (1, 24, 2, 2)    | float32   | NCHW      |             |
| 10     | (1, 546, 1, 1)   | float32   | NCHW      |             |
| 11     | (1, 24, 1, 1)    | float32   | NCHW      |             |

## Pre/Postprocessing
`furiosa.models.vision.SSDMobileNet` class provides `preprocess` and `postprocess` methods.
`preprocess` method converts input images to input tensors, and `postprocess` method converts 
model output tensors to a list of bounding boxes, scores and labels. 
You can find examples at [SSDMobileNet Usage](#SSDMobileNet_Usage).
 
### `furiosa.models.vision.SSDMobileNet.preprocess`
::: furiosa.models.vision.ssd_mobilenet.SSDMobileNetPreProcessor.__call__
    options:
        show_source: false
    
### `furiosa.models.vision.SSDMobileNet.postprocess`
::: furiosa.models.vision.ssd_mobilenet.SSDMobileNetPythonPostProcessor.__call__
    options:
        show_source: false

### <a href="NativePostProcessor"></a>Native Postprocessor
This class provides another version of the postprocessing implementation
which is highly optimized for NPU. The implementation leverages the NPU IO architecture and runtime.

To use this implementation, when this model is loaded, the parameter `use_native=True`
should be passed to `load()` or `load_aync()`. The following is an example:

!!! Example
    ```python
    --8<-- "docs/examples/ssd_mobilenet_native.py"
    ```
