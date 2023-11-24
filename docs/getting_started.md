# Getting Started

This documentation explains how to install furiosa-models, how to use available models in furiosa-models, and
how to explore the documents.

## Prerequisites

`furiosa-models` is compatible with various Linux distributions, and it has been tested on the following:

- CentOS 7 or higher
- Debian bullseye or higher
- Ubuntu 20.04 or higher

Ensure that the following packages are installed. In most systems, these are installed by default.
However, if you encounter any dependency issues, manually install the following:

- libstdc++6
- libgomp

## Installing

### APT Dependencies

To install the APT dependencies, you need to set up authentication through [FuriosaAI IAM](https://iam.furiosa.ai).
After authentication, install the following packages:

```shell
sudo apt-get update && sudo apt-get install furiosa-libhal-warboy furiosa-compiler
```

### Installing the `furiosa-models` package

You can quickly install Furiosa Models by using `pip` as following:

```shell
pip install --upgrade pip setuptools wheel
pip install 'furiosa-models'
```

!!!Info
    Older versions of wheel may reject the native-build wheels of `furiosa-models`.
    Please make sure of installing & upgrading Python packaging tools before
    installing `furiosa-models`.


??? "Building from Source Code (click to see)"
    Or you can build from the source code as following:

    ```
    git clone https://github.com/furiosa-ai/furiosa-models
    pip install ./furiosa-models
    ```

### Extra packages

To enable the use of the `furiosa models serve` command, you need to install the `furiosa-serving` Python package.
You can achieve this by specifying the serving extra package when installing `furiosa-models`.

```shell
pip install 'furiosa-models[serving]'
```

If you require additional packages related to compilation, you can install the full extra package, which includes all relevant dependencies.

```shell
pip install 'furiosa-models[full]'
```

These commands ensure that you have the necessary packages for serving models with furiosa-serving or any other compiling-related functionalities provided by the full extra package.

## Quick example and Guides

You can simply load a model and run through furiosa-sdk as the following:

!!!Info
    If you want to learn more about the installation of furiosa-sdk and how to use it, please refer to the followings:

    * [Driver, Firmware, and Runtime Installation](https://furiosa-ai.github.io/docs/latest/en/software/installation.html)
    * [Python SDK Installation and User Guide](https://furiosa-ai.github.io/docs/latest/en/software/python-sdk.html)
    * [Furiosa SDK - Tutorial and Code Examples](https://furiosa-ai.github.io/docs/latest/en/software/tutorials.html)

```python
--8<-- "docs/examples/ssd_mobilenet.py"
```

This example does:

1. Load the [SSDMobileNet](models/ssd_mobilenet.md) model
2. Create a `Runner`, which is one of the main classes of Furiosa Runtime, that can load an ONNX/tflite model onto NPU and run inferences.
3. Run an inference with pre/post process functions.

A `Model` instance is a Python object, including model artifacts, metadata, and its pre/postprocessors.
You can learn more about `Model` object at [Model object](model_object.md).

Also, you can find all available models at
[Available Models](https://furiosa-ai.github.io/furiosa-models/#available_models).
Each model page includes the details of the model, input and output tensors, and pre/post processings,
and API reference.

If you want to learn more about `Runner` in Furiosa Runtime, please refer to below links.

- [Furiosa SDK - furiosa.runtime API Reference](https://furiosa-ai.github.io/docs/latest/en/api/python/furiosa.runtime.html)
- [Furiosa SDK - furiosa.runtime.sync.create_runner Reference](https://furiosa-ai.github.io/docs/latest/en/api/python/furiosa.runtime.html#furiosa.runtime.sync.Runtime)
- [Furiosa SDK - Tutorial and Code Examples](https://furiosa-ai.github.io/docs/latest/en/software/tutorials.html).
