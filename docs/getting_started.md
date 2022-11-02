# Getting Started
This documentation explains how to install furiosa-models, how to use available models in furisoa-models, and
how to explore the documents.

## Prerequisites
`furiosa-models` can be installed on various Linux distributions, but it has been tested on the followings:

* CentOS 7 or higher
* Debian buster or higher
* Ubuntu 18.04 or higher

The following packages should be installed, but the followings are installed by default in most systems.
So, only when you have any dependency issue, you need to install the following packages:

* libstdc++6
* libgomp

## Installing
You can quickly install Furiosa Models by using `pip` as following:

```sh
pip install 'furiosa-models'
```

Or you can build from the source code as following:

```
git clone https://github.com/furiosa-ai/furiosa-models
pip install .
```

## Quick example and Guides

You can simply load a model and run through furiosa-sdk as following:
```python
--8<-- "docs/examples/ssd_mobilenet.py"
```

This example 1) fetches the [SSDMobileNet](models/ssd_mobilenet.md) model, 2) create a `session`, 
which is the main class of Furiosa Runtime which actually loads an ONNX/tflite model to NPU and run inferences,
and 3) run an inference with pre/post processings.

A `Model` instance is a Python object, including model artifacts and metadata.
You can learn more about `Model` object at [Model object](model_object.md).
Each mode has its own pre/post processing steps. 
To learn about them, please refer to [Pre/Post processing](pre_postprocess.md).

Also, you can find all available models at 
[Available Models](https://furiosa-ai.github.io/furiosa-models/#available_models).
Each model page includes the model information, input and output tensors, and pre/post processings, 
and API reference.

If you want to learn more about `furiosa.runtime.session` in Furiosa Runtime, please refer to
[Furiosa SDK - Tutorial and Code Examples](https://furiosa-ai.github.io/docs/latest/en/software/tutorials.html).
