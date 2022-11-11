# Command Tool
We provide a simple command line too called `furiosa-models` to allow users to 
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

## Subcommand: bench

`bench` subcommand runs a specific model with a given path where the input sample data are located.
It will print out the performance benchmark results like QPS.

*Example*
```
$ furiosa-models bench ResNet50 ./samples

Running 10 input samples ...
----------------------------------------------------------------------
WARN: the benchmark results may depend on the number of input samples,
sizes of the images, and a machine where this benchmark is running.
----------------------------------------------------------------------

----------------------------------------------------------------------
Preprocess -> Inference -> Postprocess
----------------------------------------------------------------------
Total elapsed time: 1.04471 sec
QPS: 9.57201
Avg. elapsed time / sample: 104.47126 ms

----------------------------------------------------------------------
Inference -> Postprocess
----------------------------------------------------------------------
Total elapsed time: 24.98687 ms
QPS: 400.21017
Avg. elapsed time / sample: 2.49869 ms

----------------------------------------------------------------------
Inference
----------------------------------------------------------------------
Total elapsed time: 558.66595 us
QPS: 17899.78418
Avg. elapsed time / sample: 55.86660 us
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
available postprocess versions: Python, Rust
```