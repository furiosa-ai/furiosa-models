# Command Tool
We provide a simple command line too called `furiosa-models` to allow users to 
evaluate or run quickly one of models with FuriosaAI NPU. 

## Installing
To install `furiosa-models` command, please refer to [Installing](getting_started.md#installing). 
Then, `furiosa-models` command will be available.

## Synopsis
```
furiosa-models [-h] {list, desc, run} ...
```

`furiosa-models` command has two subcommands: `list` and `run`.

## Subcommand: List

`list` subcommand prints out the list of models with attributes. 
You will be able to figure out what models are available. 

*Example*
```
$ furiosa-models list

+--------------+------------------------------+----------------------+-------------------------+
|  Model name  |      Model description       |      Task type       | Available postprocesses |
+--------------+------------------------------+----------------------+-------------------------+
|   ResNet50   |   MLCommons ResNet50 model   | Image Classification |      Python, Rust       |
| SSDMobileNet | MLCommons MobileNet v1 model |   Object Detection   |    Python, Rust, Cpp    |
| SSDResNet34  | MLCommons SSD ResNet34 model |   Object Detection   |    Python, Rust, Cpp    |
|   YOLOv5l    |      YOLOv5 Large model      |   Object Detection   |         Python          |
|   YOLOv5m    |     YOLOv5 Medium model      |   Object Detection   |         Python          |
+--------------+------------------------------+----------------------+-------------------------+
```

## Subcommand: Run

`run` subcommand runs a specific model with a given path where the input sample data are located.
It also prints out the performance benchmark results like QPS.

*Example*
```
$ furiosa-models run ResNet50 tests/assets/
```

## Subcommand: Desc

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
available postprocess versions: Python, Rust
```