# Getting Started
This documentation explains how to install furiosa-models, how to use available models in furiosa-models, and
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

??? "Building from Source Code (click to see)"
    Or you can build from the source code as following:

    Building `furiosa-models` requires additional prerequisites:

    * rust-toolchain (please refer to [rustup](https://rustup.rs/))
    
    ```
    git clone https://github.com/furiosa-ai/furiosa-models
    pip install .
    ```

## Quick example and Guides

You can simply load a model and run through furiosa-sdk as the following:

!!!Info
    If you want to learn more about the installation of furiosa-sdk and how to use it, please follow the followings:

    * [Driver, Firmware, and Runtime Installation](https://furiosa-ai.github.io/docs/latest/en/software/installation.html)
    * [Python SDK Installation and User Guide](https://furiosa-ai.github.io/docs/latest/en/software/python-sdk.html)
    * [Furiosa SDK - Tutorial and Code Examples](https://furiosa-ai.github.io/docs/latest/en/software/tutorials.html)

```python
--8<-- "docs/examples/ssd_mobilenet.py"
```

This example does:

1. Loads the [SSDMobileNet](models/ssd_mobilenet.md) model
1. Creates a `session` which is the main class of Furiosa Runtime which actually loads an ONNX/tflite model to NPU and run inferences
1. Runs an inference with pre/post processings.

A `Model` instance is a Python object, including model artifacts, metadata, and its pre/postprocessors.
You can learn more about `Model` object at [Model object](model_object.md).

Also, you can find all available models at 
[Available Models](https://furiosa-ai.github.io/furiosa-models/#available_models).
Each model page includes the details of the model, input and output tensors, and pre/post processings, 
and API reference.

If you want to learn more about `furiosa.runtime.session` in Furiosa Runtime, please refer to
[Furiosa SDK - Tutorial and Code Examples](https://furiosa-ai.github.io/docs/latest/en/software/tutorials.html).
