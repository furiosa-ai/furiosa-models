# SSD ResNet34

SSD ResNet34 backbone model trained on COCO (1200x1200).
This model has been used since MLCommons v0.5.

## Overall
* Framework: PyTorch
* Model format: ONNX
* Model task: Object detection
* Source: This model is originated from SSD ResNet34 in ONNX available at
[MLCommons - Supported Models](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#supported-models).

## Usage

## Usages
=== "Default"
    ```python
    --8<-- "docs/examples/ssd_resnet34.py"
    ```
 
=== "Native Postprocessor"
    ```python
    --8<-- "docs/examples/ssd_resnet34_native.py"
    ```

## Model inputs
The input is a 3-channel image of 300x300 (height, width).

* Data Type: `numpy.float32`
* Tensor Shape: `[1, 3, 1200, 1200]`
* Memory Format: NCHW, where
    * N - batch size
    * C - number of channels
    * H - image height
    * W - image width
* Optimal Batch Size: 1

## Outputs
The outputs are 12 `numpy.float32` tensors in various shapes as the following. 
You can refer to `postprocess()` function to learn how to decode boxes, classes, and confidence scores.

| Tensor | Shape            | Data Type | Data Type | Description |
|--------|------------------|-----------|-----------|-------------|
| 0      | (1, 324, 50, 50) | float32   | NCHW      |             |
| 1      | (1, 486, 25, 25) | float32   | NCHW      |             |
| 2      | (1, 486, 13, 13) | float32   | NCHW      |             |
| 3      | (1, 486, 7, 7)   | float32   | NCHW      |             |
| 4      | (1, 324, 3, 3)   | float32   | NCHW      |             |
| 5      | (1, 324, 3, 3)   | float32   | NCHW      |             |
| 6      | (1, 16, 50, 50)  | float32   | NCHW      |             |
| 7      | (1, 24, 25, 25)  | float32   | NCHW      |             |
| 8      | (1, 24, 13, 13)  | float32   | NCHW      |             |
| 9      | (1, 24, 7, 7)    | float32   | NCHW      |             |
| 10     | (1, 16, 3, 3)    | float32   | NCHW      |             |
| 11     | (1, 16, 3, 3)    | float32   | NCHW      |             |

## Pre/Post processing
`furiosa.models.vision.ssd_resnet34` module provides a set of utilities 
to convert images to input tensors and the model outputs to labels.
  
### `furiosa.models.vision.ssd_resnet34.preprocess`
::: furiosa.models.vision.ssd_resnet34.preprocess
    options:
        show_root_heading: false
### `furiosa.models.vision.ssd_resnet34.postprocess`
::: furiosa.models.vision.ssd_resnet34.postprocess
    options:
        show_root_heading: false
 
### `furiosa.models.vision.ssd_resnet34.NativePostProcess`
::: furiosa.models.vision.ssd_resnet34.NativePostProcessor
    options:
        show_root_heading: false
        show_bases: false
        show_source: false