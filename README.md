FuriosaAI Models
======================
Furiosa Models provides DNN models including quantized pre-trained weights, model metadata, and 
runtime configurations for FuriosaAI SDK and NPU. Basically, all models are specifically optimized for FuriosaAI NPU, 
but the models are based on standard ONNX format. You can feel free to use all models for even CPU and GPU.

## Available models

| Model                               | Task                 | Size | Accuracy | Latency (NPU) | Latency (CPU) |
|-------------------------------------|----------------------|------|----------|---------------|---------------|
| [ResNet50](models/resnet50_v1.5.md) | Image Classification | 25M  |          |               |               |
| SSDMobileNet                        | Object Detection     | 7.2M |          |               |               |
| SSDResNet35                         | Object Detection     | 20M  |          |               |               |
| YOLOv5M                             | Object Detection     | 21M  |          |               |               |
| YOLOv5L                             | Object Detection     | 46M  |          |               |               |

## Installation
You can quickly install `furiosa-models` by using `pip`.
```sh
pip install furiosa-models
```

Or you can build from the source code as following:

```
git clone https://github.com/furiosa-ai/furiosa-models-experimental
pip install .
```

## Example

```python
from furiosa.registry import Model
from furiosa.models.vision import ResNet50

model: Model = ResNet50()
```

## See Also
* [Furiosa Models - Github](https://github.com/furiosa-ai/furiosa-models-experimental)
* [Furiosa SDK Documentation](https://furiosa-ai.github.io/docs/latest/en/)