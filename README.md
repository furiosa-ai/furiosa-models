Furiosa Models
======================
Furiosa Models provides DNN models including quantized pre-trained weights, model metadata, and 
runtime configurations for FuriosaAI SDK and NPU. Basically, all models are specifically optimized for FuriosaAI NPU, 
but the models are based on standard ONNX format. You can feel free to use all models for even CPU and GPU.

## <a name="AvailableModels"></a>Available Models

| Model                                        | Task                 | Size | Accuracy | Latency (NPU) | Latency (CPU) |
|----------------------------------------------|----------------------|------|----------|---------------|---------------|
| [ResNet50](docs/models/resnet50_v1.5.md)     | Image Classification | 25M  |          |               |               |
| [SSDMobileNet](docs/models/ssd_mobilenet.md) | Object Detection     | 7.2M |          |               |               |
| [SSDResNet34](docs/models/ssd_resnet34.md)   | Object Detection     | 20M  |          |               |               |
| YOLOv5M                                      | Object Detection     | 21M  |          |               |               |
| YOLOv5L                                      | Object Detection     | 46M  |          |               |               |

## Installation
You can quickly install Furiosa Models by using `pip` as following:
```sh
pip install 'furiosa-sdk[models]'
```

Or you can build from the source code as following:

```
git clone https://github.com/furiosa-ai/furiosa-models
pip install .
```

## Usage
You can simply access each model as following:
```python
from furiosa.models.vision import ResNet50

model = ResNet50()
```

Each model in [available models](#AvailableModels) also provides the details 
including how to access the model, input and output tensors, and pre/post processings.

If you want to learn more about Furiosa SDK, you can refer to 
[Furiosa SDK - Tutorial and Code Examples](https://furiosa-ai.github.io/docs/latest/en/software/tutorials.html)

Also, you can learn about [Blocking and Non-blocking APIs](blocking_and_nonblocking_api.md) 
if you want to access the models with Asynchronous (AsyncIO) client library.

## See Also
* [Furiosa Models - Github](https://github.com/furiosa-ai/furiosa-models)
* [Furiosa SDK Documentation](https://furiosa-ai.github.io/docs/latest/en/)
