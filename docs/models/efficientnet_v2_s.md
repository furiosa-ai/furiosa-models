# EfficientNetV2-S

EfficientNetV2-S is the smallest and most efficient model in the EfficientNetV2 family. Introduced in the paper ["EfficientNetV2: Smaller Models and Faster Training"](https://arxiv.org/abs/2104.00298), EfficientNetV2-S achieves state-of-the-art performance on image classification tasks, and it can be trained much faster and has a smaller model size of up to 6.8 times when compared to previous state-of-the-art models. It uses a combination of advanced techniques such as Swish activation function, Squeeze-and-Excitation blocks, and efficient channel attention to optimize its performance and efficiency.

## Overall
* Framework: PyTorch
* Model format: ONNX
* Model task: Image classification
* Source: [torchvision](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html)

## <a name="EfficientNetV2-S_Usage"></a>
!!! Usages
    === "Python Postprocessor"
        ```python
        --8<-- "docs/examples/efficientnet_v2_s.py"
        ```

## Inputs
The input is a 3-channel image of 384x384 (height, width).

* Data Type: `numpy.float32`
* Tensor Shape: `[1, 3, 384, 384]`
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
`furiosa.models.vision.EfficientNetV2s` class provides `preprocess` and `postprocess` methods that
convert input images to input tensors and the model outputs to labels respectively.
You can find examples at [EfficientNetV2-S Usage](#EfficientNetV2-S_Usage).

### `furiosa.models.vision.EfficientNetV2s.preprocess`
::: furiosa.models.vision.efficientnet_v2_s.EfficientNetV2sPreProcessor.__call__
    options:
        show_source: false

### `furiosa.models.vision.EfficientNetV2s.postprocess`
::: furiosa.models.vision.efficientnet_v2_s.EfficientNetV2sPostProcessor.__call__
    options:
        show_source: false

## Notes on the source field of this model
There is a significant update in sdk version 0.9.0, which involves that the Furiosa's quantization tool adopts DFG(Data Flow Graph) as its output format instead of onnx. DFG is an IR of FuriosaAI that supports more diverse quantization schemes than onnx and is more specialized for FuriosaAI’s Warboy.

The EfficientNetV2-S we offer has been quantized with furiosa-sdk 0.9.0 and thus formatted in DFG. The ONNX file in the source field is the original f32 model, not yet quantized.

In case you need to use a different batch size or start from scratch, you can either start from the DFG or use the original ONNX file (and repeat the quantization process).
