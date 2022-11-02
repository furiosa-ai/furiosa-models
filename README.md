Furiosa Models
======================
Furiosa Models provides a set of public pre-trained, pre-quantized models with various metadata.
You can use Furiosa Models for learning and demo purposes or for developing your applications.

Basically, all models are specifically optimized for FuriosaAI NPU,
In addition, it includes pre-packaged post/processing functions and runtime configurations 
optimized for FuriosaAI SDK and NPU. However, the models are standard ONNX or tflite models, 
and they can run on even CPU and GPU.

[https://github.com/furiosa-ai/furiosa-models](https://github.com/furiosa-ai/furiosa-models)

## Online Documentation
You can find the latest furiosa-ai documentation 
including programming guide, API reference, examples from the followings:

* [Furiosa Models - Online Documentation](https://furiosa-ai.github.io/furiosa-models/latest/)
* [Getting Started](https://furiosa-ai.github.io/furiosa-models/latest/getting_started.md)
* [Model object and its Examples](https://furiosa-ai.github.io/furiosa-models/models_and_examples.md)
* [Available Models](https://furiosa-ai.github.io/furiosa-models/latest/#available_models)
* [Furiosa SDK - Tutorial and Code Examples](https://furiosa-ai.github.io/docs/latest/en/software/tutorials.html)


## <a name="Models"></a>Models
The table summarizes all models in Furisoa Models. If you visit each model link, 
you can find details about how to load the model, input and output tensors, and pre/post processings, and examples.

| Model                                   | Task                 | Size | Accuracy |
|-----------------------------------------|----------------------|------|----------|
| [ResNet50](models/resnet50_v1.5.md)     | Image Classification | 25M  | 76.002%  |
| [SSDMobileNet](models/ssd_mobilenet.md) | Object Detection     | 7.2M | mAP 0.228|
| [SSDResNet34](models/ssd_resnet34.md)   | Object Detection     | 20M  | mAP 0.220|
| [YOLOv5M](models/yolov5m.md)            | Object Detection     | 21M  | mAP 0.280|
| [YOLOv5L](models/yolov5l.md)            | Object Detection     | 46M  | mAP 0.295|

## See Also
* [Furiosa Models - Documentation](https://furiosa-ai.github.io/furiosa-models/latest/)
* [Furiosa Models - Github](https://github.com/furiosa-ai/furiosa-models)
* [Furiosa SDK Documentation](https://furiosa-ai.github.io/docs/latest/en/)