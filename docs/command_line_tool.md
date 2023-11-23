# Command Line Tool

`furiosa-models` is a command line tool provided by FuriosaAI to allow users to
evaluate or quickly run models with FuriosaAI NPU.

## Installing

To install `furiosa-models` command, please refer to the [installation guide](getting_started.md#installing).
Once installed, the `furiosa-models` command will be available.

## Synopsis

```text
$ furiosa-models --help

Usage: furiosa-models [OPTIONS] COMMAND [ARGS]...

  FuriosaAI Model Zoo CLI --- v0.10.0

Options:
  --help  Show this message and exit.

Commands:
  bench  Run benchmark on a model
  desc   Describe a model
  list   List available models

Examples:
  # List available models
  `furiosa-models list`

  # List Object Detection models
  `furiosa-models list -t detect`

  # Describe SSDResNet34 model
  `furiosa-models desc SSDResNet34`

  # Run SSDResNet34 for images in `./input` directory
  `furiosa-models bench ssd-resnet34 ./input/`
```


## Subcommand: `list`

The `list` subcommand prints out the list of available models with their attributes.
It helps users to identify the models available for use.

*Help Message*
```text
$ furiosa-models list --help

Usage: furiosa-models list [OPTIONS] [FILTER_TYPE]

  List available models

Arguments:
  [FILTER_TYPE]  Limits the task type (ex. classify, detect, pose)

Options:
  --help  Show this message and exit.
```

*Example*
```
$ furiosa-models list

+-----------------+---------------------------------+----------------------+-------------------------+
|   Model name    |        Model description        |      Task type       | Available postprocesses |
+-----------------+---------------------------------+----------------------+-------------------------+
|    ResNet50     |    MLCommons ResNet50 model     | Image Classification |         Python          |
|  SSDMobileNet   |  MLCommons MobileNet v1 model   |   Object Detection   |      Python, Rust       |
|   SSDResNet34   |  MLCommons SSD ResNet34 model   |   Object Detection   |      Python, Rust       |
|     YOLOv5l     |       YOLOv5 Large model        |   Object Detection   |          Rust           |
|     YOLOv5m     |       YOLOv5 Medium model       |   Object Detection   |          Rust           |
| EfficientNetB0  |      EfficientNet B0 model      | Image Classification |         Python          |
| EfficientNetV2s |     EfficientNetV2-s model      | Image Classification |         Python          |
|  YOLOv7w6Pose   | YOLOv7 w6 Pose Estimation model |   Pose Estimation    |         Python          |
+-----------------+---------------------------------+----------------------+-------------------------+
```

## Subcommand: `desc`

The `desc` subcommand provides detailed information about a specific model.

*Help Message*
```text
$ furiosa-models desc --help

Usage: furiosa-models desc [OPTIONS] MODEL_NAME

  Describe a model

Arguments:
  MODEL_NAME  [required]

Options:
  --help  Show this message and exit.
```


*Example*
```
$ furiosa-models desc ResNet50

name: ResNet50
format: ONNX
family: ResNet
version: v1.5
metadata:
  description: ResNet50 v1.5 int8 ImageNet-1K
  publication:
    url: https://arxiv.org/abs/1512.03385.pdf
task type: Image Classification
available postprocess versions: Python
```

## Subcommand: `bench`

The `bench` subcommand runs a specific model with input data and prints out performance benchmark results
such as queries per second (QPS) and average latency.

*Help Message*
```text
$ furiosa-models bench --help

Usage: furiosa-models bench [OPTIONS] MODEL INPUT_PATH

  Run benchmark on a model

Arguments:
  MODEL       [required]
  INPUT_PATH  [required]

Options:
  --postprocess TEXT  Specifies a postprocess implementation
  --help              Show this message and exit.
```

*Example*
```
$ furiosa-models bench ResNet50 ./

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

The benchmark results include information about preprocessing, inference,
postprocessing, and overall performance metrics.
