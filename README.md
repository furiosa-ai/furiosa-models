Furiosa Models
======================
`furiosa-models` is an open model zoo project for FuriosaAI NPU. 
It provides a set of public pre-trained, pre-quantized models for learning and demo purposes or 
for developing your applications.

`furiosa-models` also includes pre-packaged post/processing utilities, compiler configurations optimized
for FuriosaAI NPU. However, all models are standard ONNX or tflite models, 
and they can run even on CPU and GPU as well.

## Releases
* [v0.8.0](https://furiosa-ai.github.io/furiosa-models/changelog/)

## Online Documentation
If you are new, you can start from [Getting Started](https://furiosa-ai.github.io/furiosa-models/latest/getting_started.md).
You can also find the latest online documents, 
including programming guide, API reference, examples from the followings:

* [Furiosa Models - Documentation](https://furiosa-ai.github.io/furiosa-models/latest/)
* [Model object and its Examples](https://furiosa-ai.github.io/furiosa-models/models_and_examples.md)
* [Model List](https://furiosa-ai.github.io/furiosa-models/latest/#model_list)
* [Furiosa SDK - Tutorial and Code Examples](https://furiosa-ai.github.io/docs/latest/en/software/tutorials.html)


## <a name="model_list"></a>Model List
The table summarizes all models available in `furiosa-models`. If you visit each model link, 
you can find details about loading a model, their input and output tensors, and pre/post processings, and examples.

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