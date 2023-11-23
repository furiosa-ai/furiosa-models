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
  serve  Open a REST API server for a model

Examples:
  # List available models
  `furiosa-models list`

  # List Object Detection models
  `furiosa-models list -t detect`

  # Describe SSDResNet34 model
  `furiosa-models desc SSDResNet34`

  # Run SSDResNet34 for images in `./input` directory
  `furiosa-models bench ssd-resnet34 ./input/`

  # Run YOLOv7w6Pose REST API server for `0.0.0.0:8080`
  `furiosa-models serve yolov7w6pose --host 0.0.0.0 --port 8080`
```


## Subcommand: `list`

The `list` subcommand prints out the list of available models with their attributes.
It helps users to identify the models available for use.

### Help Message

```text
$ furiosa-models list --help

Usage: furiosa-models list [OPTIONS] [FILTER_TYPE]

  List available models

Arguments:
  [FILTER_TYPE]  Limits the task type (ex. classify, detect, pose)

Options:
  --help  Show this message and exit.
```

### Example

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

### Help Message

```text
$ furiosa-models desc --help

Usage: furiosa-models desc [OPTIONS] MODEL_NAME

  Describe a model

Arguments:
  MODEL_NAME  [required]

Options:
  --help  Show this message and exit.
```

### Example

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

For detailed information on device specification for `--devices` argument, please refer to the
[device specification documentation](https://furiosa-ai.github.io/docs/latest/en/api/python/furiosa.runtime.html#device-specification)

### Help Message

```text
$ furiosa-models bench --help

Usage: furiosa-models bench [OPTIONS] MODEL INPUT_PATH

  Run benchmark on a model

Arguments:
  MODEL       [required]
  INPUT_PATH  [required]

Options:
  --postprocess TEXT  Specifies a postprocess implementation
  --devices TEXT      Specifies devices to run the model (ex. 'warboy(2)*1')
  --help              Show this message and exit.
```

### Example

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

## Subcommand: `serve`

The `serve` command in furiosa-models allows you to deploy a machine learning model as a REST API server,
enabling users to perform inference on input data through HTTP requests.

!!! note
    The serve command requires the `furiosa-serving` library.
    Please make sure to install it before using this command.
    ```shell
    pip install furiosa-sdk[serving]
    ```
    The `furiosa-serving` leverages `FastAPI` and `uvicorn` under the hood.

For detailed information on device specification for `--devices` argument, please refer to the
[device specification documentation](https://furiosa-ai.github.io/docs/latest/en/api/python/furiosa.runtime.html#device-specification)

### Help Message

```text
$ furiosa-models serve --help

Usage: furiosa-models serve [OPTIONS] MODEL

  Open a REST API server for a model

Arguments:
  MODEL  [required]

Options:
  --postprocess TEXT  Specifies a postprocess implementation
  --host TEXT         Specifies a host address  [default: 0.0.0.0]
  --port INTEGER      Specifies a port number  [default: 8000]
  --devices TEXT      Specifies devices to run the model (ex. 'warboy(2)*1')
  --help              Show this message and exit.
```

### Example

Start a server for the YOLOv5m model with Rust postprocessing on port 1234:
```
furiosa-models serve yolov5m --postprocess rust --port 1234
```

Once the server is running, perform inference with a sample curl command:
```
$ curl -X 'POST' \
  'http://0.0.0.0:1234/infer' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@./tests/assets/yolov5-test.jpg;type=image/jpeg'
```

Response example
```json
{"result": [[{"boundingbox": {"left": 3682.03, "top": 2128.83, "right": 3922.49, "bottom": 2363.04}, "score": 0.89, "label": "traffic sign", "index": 9}, {"boundingbox": {"left": 821.35, "top": 2012.27, "right": 1061.16, "bottom": 2392.25}, "score": 0.85, "label": "car", "index": 2}, ...]]}
```
