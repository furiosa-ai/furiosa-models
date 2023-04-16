# EfficientNetB0

The EfficientNet originates from the ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"](https://arxiv.org/abs/1905.11946) paper, which proposes a new compound scaling method that enables better performance on image classification tasks with fewer parameters. EfficientNet B0 is the smallest and most efficient model in the EfficientNet family, achieving state-of-the-art results on various image classification benchmarks with just 5.3 million parameters.

## Overall
* Framework: PyTorch
* Model format: ONNX
* Model task: Image classification
* Source: [github](https://github.com/rwightman/gen-efficientnet-pytorch)


## <a name="EfficientNetB0_Usage"></a>
!!! Usages
    === "Python Postprocessor"
        ```python
        --8<-- "docs/examples/efficientnet_b0.py"
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
* Optimal Batch Size (minimum: 1): <= 16

## Outputs
The output is a `numpy.float32` tensor with the shape (`[1,]`), including
a class id. `postprocess()` transforms the class id to a label string.

## Pre/Postprocessing
`furiosa.models.vision.EfficientNetB0` class provides `preprocess` and `postprocess` methods that
convert input images to input tensors and the model outputs to labels respectively.
You can find examples at [EfficientNetB0 Usage](#EfficientNetB0_Usage).

### `furiosa.models.vision.EfficientNetB0.preprocess`
::: furiosa.models.vision.efficientnet_b0.EfficientNetB0PreProcessor.__call__
    options:
        show_source: false

### `furiosa.models.vision.EfficientNetB0.postprocess`
::: furiosa.models.vision.efficientnet_b0.EfficientNetB0PostProcessor.__call__
    options:
        show_source: false

## Notes on the source field of this model
There is a significant update in sdk version 0.9.0, which involves that the Furiosa's quantization tool adopts DFG(Data Flow Graph) as its output format instead of onnx. DFG is an IR of FuriosaAI that supports more diverse quantization schemes than onnx and is more specialized for FuriosaAI’s Warboy.

The EfficientNetB0 we offer has been quantized with furiosa-sdk 0.9.0 and thus formatted in DFG. The ONNX file in the source field is the original f32 model, not yet quantized.

In case you need to use a different batch size or start from scratch, you can either start from the DFG or use the original ONNX file (and repeat the quantization process).

