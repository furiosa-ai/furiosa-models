Furiosa Models
======================
`furiosa-models` is an open model zoo project for FuriosaAI NPU. 
It provides a set of public pre-trained, pre-quantized models for learning and demo purposes or 
for developing your applications.

`furiosa-models` also includes pre-packaged post/processing utilities, compiler configurations optimized
for FuriosaAI NPU. However, all models are standard ONNX or tflite models, 
and they can run even on CPU and GPU as well.

## Releases
* [v0.8.0](https://furiosa-ai.github.io/furiosa-models/v0.8.0/changelog/) - 2022-11-10

## Online Documentation
If you are new, you can start from [Getting Started](getting_started.md).
You can also find the latest online documents, 
including programming guides, API references, and examples from the followings:

* [Furiosa Models - Latest Documentation](https://furiosa-ai.github.io/furiosa-models/latest/)
* [Model object](model_object.md)
* [Model List](#model_list)
* [Command Tool](command_tool.md)
* [Furiosa SDK - Tutorial and Code Examples](https://furiosa-ai.github.io/docs/latest/en/software/tutorials.html)


## <a name="model_list"></a>Model List
The table summarizes all models available in `furiosa-models`. If you visit each model link, 
you can find details about loading a model, their input and output tensors, pre/post processings, and usage examples.

| Model                                   | Task                 | Size | Accuracy                  |
|-----------------------------------------| -------------------- | ---- |---------------------------|
| [ResNet50](models/resnet50_v1.5.md)     | Image Classification | 25M  | 76.002% (ImageNet1K-val)  |
| [SSDMobileNet](models/ssd_mobilenet.md) | Object Detection     | 7.2M | mAP 0.228 (COCO 2017-val) |
| [SSDResNet34](models/ssd_resnet34.md)   | Object Detection     | 20M  | mAP 0.220 (COCO 2017-val) |
| [YOLOv5M](models/yolov5m.md)            | Object Detection     | 21M  | mAP 0.280                 |
| [YOLOv5L](models/yolov5l.md)            | Object Detection     | 46M  | mAP 0.295                 |

## See Also
* [Furiosa Models - Latest Documentation](https://furiosa-ai.github.io/furiosa-models/latest/)
* [Furiosa Models - Github](https://github.com/furiosa-ai/furiosa-models)
* [Furiosa SDK Documentation](https://furiosa-ai.github.io/docs/latest/en/)