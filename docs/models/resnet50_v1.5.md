# ResNet50 v1.5

ResNet50 v1.5 backbone model trained on ImageNet (224x224).
This model has been used since MLCommons v0.5.

## Overall
* Framework: PyTorch
* Model format: ONNX
* Model task: Image classification
* Source: This model is originated from ResNet50 v1.5 in ONNX available at
[MLCommons - Supported Models](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#supported-models).


## <a name="ResNet50_Usage"></a>
!!! Usages
    === "Postprocessor"
        ```python
        --8<-- "docs/examples/resnet50.py"
        ```

## Inputs
The input is a 3-channel image of 224x224 (height, width).

* Data Type: `numpy.float32`
* Tensor Shape: `[1, 3, 224, 224]`
* Memory Format: NCHW, where:
    * N - batch size
    * C - number of channels
    * H - image height
    * W - image width
* Color Order: BGR
* Optimal Batch Size (minimum: 1): <= 8

## Outputs
The output is a `numpy.float32` tensor with the shape (`[1,]`), including
a class id. `postprocess()` transforms the class id to a label string.

## Pre/Postprocessing
`furiosa.models.vision.ResNet50` class provides `preprocess` and `postprocess` methods that
convert input images to input tensors and the model outputs to labels respectively.
You can find examples at [ResNet50 Usage](#ResNet50_Usage).

### `furiosa.models.vision.ResNet50.preprocess`
::: furiosa.models.vision.resnet50.ResNet50PreProcessor.__call__
    options:
        show_source: false

### `furiosa.models.vision.ResNet50.postprocess`
::: furiosa.models.vision.resnet50.ResNet50PostProcessor.__call__
    options:
        show_source: false
