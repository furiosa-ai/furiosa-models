Furiosa Models
======================
`furiosa-models` is an open model zoo project for FuriosaAI NPU.
It provides a set of public pre-trained, pre-quantized models for learning and demo purposes or
for developing your applications.

`furiosa-models` also includes pre-packaged post/processing utilities, and compiler configurations optimized
for FuriosaAI NPU. However, all models are standard ONNX or tflite models,
and they can run even on CPU and GPU as well.

## Releases
* [v0.10.1](https://furiosa-ai.github.io/furiosa-models/v0.10.1/changelog/) - 2023-11-25
* [v0.10.0](https://furiosa-ai.github.io/furiosa-models/v0.10.0/changelog/) - 2023-08-28
* [v0.9.1](https://furiosa-ai.github.io/furiosa-models/v0.9.1/changelog/) - 2023-05-26
* [v0.9.0](https://furiosa-ai.github.io/furiosa-models/v0.9.0/changelog/) - 2023-05-12
* [v0.8.0](https://furiosa-ai.github.io/furiosa-models/v0.8.0/changelog/) - 2022-11-10

## Online Documentation
If you are new, you can start from [Getting Started](https://furiosa-ai.github.io/furiosa-models/v0.10.1/getting_started/).
You can also find the latest online documents,
including programming guides, API references, and examples from the following:

* [Furiosa Models - Latest Documentation](https://furiosa-ai.github.io/furiosa-models/latest/)
* [Model Object](https://furiosa-ai.github.io/furiosa-models/v0.10.1/model_object/)
* [Model List](https://furiosa-ai.github.io/furiosa-models/v0.10.1/#model_list)
* [Command Line Tool](https://furiosa-ai.github.io/furiosa-models/v0.10.1/command_line_tool/)
* [Furiosa SDK - Tutorial and Code Examples](https://furiosa-ai.github.io/docs/v0.10.1/en/software/tutorials.html)


## <a name="model_list"></a>Model List
The table summarizes all models available in `furiosa-models`. If you visit each model link,
you can find details about loading a model, their input and output tensors, pre/post-processing, and usage examples.

| Model                                                                                             | Task                 | Size | Accuracy                  |
| ------------------------------------------------------------------------------------------------- | -------------------- | ---- | ------------------------- |
| [ResNet50](https://furiosa-ai.github.io/furiosa-models/v0.10.1/models/resnet50_v1.5/)             | Image Classification | 25M  | 75.618% (ImageNet1K-val)  |
| [EfficientNetB0](https://furiosa-ai.github.io/furiosa-models/v0.10.1/models/efficientnet_b0/)     | Image Classification | 6.4M | 72.44% (ImageNet1K-val)   |
| [EfficientNetV2-S](https://furiosa-ai.github.io/furiosa-models/v0.10.1/models/efficientnet_v2_s/) | Image Classification | 26M  | 83.532% (ImageNet1K-val)  |
| [SSDMobileNet](https://furiosa-ai.github.io/furiosa-models/v0.10.1/models/ssd_mobilenet/)         | Object Detection     | 7.2M | mAP 0.232 (COCO 2017-val) |
| [SSDResNet34](https://furiosa-ai.github.io/furiosa-models/v0.10.1/models/ssd_resnet34/)           | Object Detection     | 20M  | mAP 0.220 (COCO 2017-val) |
| [YOLOv5M](https://furiosa-ai.github.io/furiosa-models/v0.10.1/models/yolov5m/)                    | Object Detection     | 21M  | mAP 0.272 (Bdd100k-val) <sup>[[1]](#footnote_1)</sup> |
| [YOLOv5L](https://furiosa-ai.github.io/furiosa-models/v0.10.1/models/yolov5l/)                    | Object Detection     | 46M  | mAP 0.284 (Bdd100k-val) <sup>[[1]](#footnote_1)</sup> |
| [YOLOv7w6Pose](https://furiosa-ai.github.io/furiosa-models/v0.10.1/models/yolov7_w6_pose/)         | Pose Estimation      | 80M  | N/A                       |

_<a name="footnote_1">[1]</a>: The accuracy of the yolov5 f32 model trained with bdd100k-val dataset, is mAP 0.295 (for yolov5m) and mAP 0.316 (for yolov5l)._

## See Also
* [Furiosa Models - Latest Documentation](https://furiosa-ai.github.io/furiosa-models/latest/)
* [Furiosa Models - Github](https://github.com/furiosa-ai/furiosa-models)
* [Furiosa SDK Documentation](https://furiosa-ai.github.io/docs/latest/en/)
