# Command Tool
We provide a simple command line tool called `furiosa-models` to allow users to
evaluate or run quickly one of models with FuriosaAI NPU.

## Installing
To install `furiosa-models` command, please refer to [Installing](getting_started.md#installing).
Then, `furiosa-models` command will be available.

## Synopsis
```
furiosa-models [-h] {list, desc, bench} ...
```

`furiosa-models` command has three subcommands: `list`, `desc`, and `bench`.

## Subcommand: list

`list` subcommand prints out the list of models with attributes.
You will be able to figure out what models are available.

*Example*
```
$ furiosa-models list

+-----------------+------------------------------+----------------------+-------------------------+
|   Model name    |      Model description       |      Task type       | Available postprocesses |
+-----------------+------------------------------+----------------------+-------------------------+
|    ResNet50     |   MLCommons ResNet50 model   | Image Classification |         Python          |
|  SSDMobileNet   | MLCommons MobileNet v1 model |   Object Detection   |      Python, Rust       |
|   SSDResNet34   | MLCommons SSD ResNet34 model |   Object Detection   |      Python, Rust       |
|     YOLOv5l     |      YOLOv5 Large model      |   Object Detection   |         Python          |
|     YOLOv5m     |     YOLOv5 Medium model      |   Object Detection   |         Python          |
| EfficientNetB0  |    EfficientNet B0 model     | Image Classification |         Python          |
| EfficientNetV2s |    EfficientNetV2-s model    | Image Classification |         Python          |
+-----------------+------------------------------+----------------------+-------------------------+
```

## Subcommand: bench

`bench` subcommand runs a specific model with a given path where the input sample data are located.
It will print out the performance benchmark results like QPS.

*Example*
```
$ furiosa-models bench ResNet50 .
libfuriosa_hal.so --- v0.11.0, built @ 43c901f
Running 4 input samples ...
----------------------------------------------------------------------
WARN: the benchmark results may depend on the number of input samples,
sizes of the images, and a machine where this benchmark is running.
----------------------------------------------------------------------

----------------------------------------------------------------------
Preprocess -> Inference -> Postprocess
----------------------------------------------------------------------
Total elapsed time: 618.86050 ms
QPS: 6.46349
Avg. elapsed time / sample: 154.71513 ms

----------------------------------------------------------------------
Inference -> Postprocess
----------------------------------------------------------------------
Total elapsed time: 5.40490 ms
QPS: 740.06865
Avg. elapsed time / sample: 1.35123 ms

----------------------------------------------------------------------
Inference
----------------------------------------------------------------------
Total elapsed time: 5.05762 ms
QPS: 790.88645
Avg. elapsed time / sample: 1.26440 ms
```

## Subcommand: desc

`desc` subcommand shows the details of a specific model.

*Example*
```
$ furiosa-models desc ResNet50
family: ResNet
format: onnx
metadata:
  description: ResNet50 v1.5 int8 ImageNet-1K
  publication:
    authors: null
    date: null
    publisher: null
    title: null
    url: https://arxiv.org/abs/1512.03385.pdf
name: ResNet50
version: v1.5
task type: Image Classification
available postprocess versions: Python
```