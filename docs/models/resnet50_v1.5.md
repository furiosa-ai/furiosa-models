# ResNet50 v1.5

ResNet50 v1.5 backbone model trained on ImageNet (224x224).
This model has been used since MLCommons v0.5.

## Overall
* Framework: PyTorch
* Model format: ONNX
* Model task: Image classification
* Source: This model is originated from ResNet50 v1.5 in ONNX available at
[MLCommons - Supported Models](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#supported-models).
 
## Usages
=== "Default"
    ```python
    --8<-- "docs/examples/resnet50.py"
    ```
 
=== "Native Postprocessor"
    ```python
    --8<-- "docs/examples/resnet50_native.py"
    ```
 
## Inputs of Model
The input is a 3-channel image of 224x224 (height, width).

* Data Type: `numpy.float32`
* Tensor Shape: `[1, 3, 224, 224]`
* Memory Format: NCHW, where:
    * N - batch size
    * C - number of channels
    * H - image height
    * W - image width
* NPU Optimal Batch Size: <= 8

## Output of Model
The output is a `numpy.float32` tensor with the shape (`[1,]`), including
a class id. `postprocess()` can transform the class id to a single label.

## Pre/Post processing
`furiosa.models.vision.resnet50` module provides a set of utilities 
to convert images to input tensors and the model outputs to labels.
  
### `furiosa.models.vision.resnet50.preprocess`
::: furiosa.models.vision.resnet50.preprocess
    options:
        show_root_heading: false
### `furiosa.models.vision.resnet50.postprocess`
::: furiosa.models.vision.resnet50.postprocess
    options:
        show_root_heading: false
 
### `furiosa.models.vision.resnet50.NativePostProcess`
::: furiosa.models.vision.resnet50.NativePostProcessor
    options:
        show_root_heading: false
        show_bases: false
        show_source: false
