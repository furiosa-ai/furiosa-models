# ResNet50 v1.5

ResNet50 v1.5 backbone model trained on ImageNet (224x224). 
This model has been used since MLCommons v0.5.

## Usage

### Using Furiosa SDK
```python
from furiosa.models.vision import ResNet50
from furiosa.models.vision import resnet50
from furiosa.runtime import session

resnet50 = ResNet50()

with session.create(resnet50.bytes) as sess:
    image = resnet50.preprocess("image/car.jpeg")
    output = sess.run(image)
    resnet50.postprocess(output)
```

## Model inputs
The input is a 3-channel image of 224x224 (height, width).

* Data Type: `numpy.float32`
* Tensor Shape: `[1, 3, 224, 224]`
* Memory Layout: NCHW
* Optimal Batch Size: <= 8

## Outputs
The output is a `numpy.float32` tensor with the shape (`[1,]`), including
a class id. `postprocess()` can transform the class id to a single label.

## Source
This model is originated from ResNet50 v1.5 in ONNX available at
[MLCommons - Supported Models](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#supported-models).