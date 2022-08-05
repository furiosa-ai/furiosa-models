# SSD MobileNet v1

SSD MobileNet v1 backbone model trained on COCO (300x300).
This model has been used since MLCommons v0.5.

## Usage

### Using Furiosa SDK

```python
from furiosa.models.vision import SSDMobileNet
from furiosa.models.vision import ssd_mobilenet
from furiosa.runtime import session

ssd_mobilenet = SSDMobileNet()

with session.create(ssd_mobilenet.bytes) as sess:
    image = ssd_mobilenet.preprocess("image/car.jpeg")
    output = sess.run(image)
    ssd_mobilenet.postprocess(output)
```

## Model inputs
The input is a 3-channel image of 300x300 (height, width).

* Data Type: `numpy.float32`
* Tensor Shape: `[1, 3, 300, 300]`
* Memory Layout: NCHW
* Optimal Batch Size: <= 8

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

## Source
This model is originated from SSD MobileNet v1 in ONNX available at
[MLCommons - Supported Models](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#supported-models).