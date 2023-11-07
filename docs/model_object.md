# Model Object

In the `furiosa-models` project, the `Model` is the primary class object, representing a neural network model. This document elucidates the offerings and uses of the [`Model`][furiosa.models.types.Model] object.


## Loading a pre-trained model

To load a pre-trained neural network model, you need to invoke the `Model` object.
As the sizes of pre-trained model weights can range from tens to hundreds of megabytes,
the model images are not included in the Python package. The first time the model object is called,
a pre-trained model will be fetched over the network. This process takes some time,
typically a few seconds, depending on the models and network conditions.
Once the model images are fetched, they will be cached on your local disk.

=== "Load module"
    ```python
    --8<-- "docs/examples/loading_model.py"
    ```

<a name="accessing_artifacts_and_metadata"></a>

## Accessing artifacts and metadata

A Model object encompasses model artifacts, such as ONNX, TFLite, mapping from a tensor name to the tensor's min and max, and ENF.

ENF is the serialization format of a compiled binary used in Furiosa SDK.
Once you have the ENF file, you can reuse it to skip the compilation process, which can take up to several minutes.
You can acquire the ENF binary from the model_source() method.
In addition, a Model object contains various metadata attributes.

!!!Info
    If you want to learn more about the ENF please visit
    [Furiosa SDK - Compiler - Using ENF files](https://furiosa-ai.github.io/docs/latest/en/software/compiler.html#using-enf-files)


::: furiosa.models.types.Model
    options:
        show_root_heading: true
        heading_level: 4
        show_source: false


## Inferencing with Runner API

To create a Runner, you need to pass the ENF binary obtained from the `model_source()` method of the model object to the `furiosa.runtime.sync.create_runner` function. If you prefer an asynchronous Runner, you can use the `furiosa.runtime.create_runner` function instead. Passing the pre-compiled ENF binary allows you to perform inference directly without the compilation process. Alternatively, you can also manually quantize and compile the original f32 model with the provided calibration range.


!!!Info
    If you want to learn more about the installation of furiosa-sdk and how to use it, please follow the followings:

    * [Driver, Firmware, and Runtime Installation](https://furiosa-ai.github.io/docs/latest/en/software/installation.html)
    * [Python SDK Installation and User Guide](https://furiosa-ai.github.io/docs/latest/en/software/python-sdk.html)
    * [Furiosa SDK - Tutorial and Code Examples](https://furiosa-ai.github.io/docs/latest/en/software/tutorials.html)

Passing `Model.origin` to `create_runner()` allows users to start from source models in ONNX or tflite and customize models to their specific use-cases. This customization includes options such as specifying batch sizes and compiler configurations for optimization purposes. For additional information on Model.origin, please refer to [Accessing artifacts and metadata](#accessing_artifacts_and_metadata).

To work with f32 source models, calibration and quantization are essential steps.
You can access pre-calibrated data directly from furiosa-models, simplifying the quantization process.
If you prefer a manual quantization step for a model, you can install the `furiosa-quantizer` package, available at this [package link](https://furiosa-ai.github.io/docs/latest/en/software/python-sdk.html#quantizer).
The `tensor_name_to_range` field within the model class contains this pre-calibrated data.

Upon quantization, the output will be in FuriosaAI's Intermediate Representation (IR) format, which can then be passed to the Runner. At this stage, you have the option to specify the compiler configuration.

After quantization, the output will be in the form of FuriosaAI's Intermediate Representation (IR) which
can then be passed to the session. At this stage, the compiler configuration can be specified.


<a name="Examples"></a>
!!! Example
    === "Using pre-compiled ENF binary"
        ```python
        --8<-- "docs/examples/ssd_mobilenet.py"
        ```


!!! Example
    === "From ONNX"
        ```python
        --8<-- "docs/examples/ssd_mobilenet_onnx.py"
        ```


### Pre/Postprocessing

There are gaps between model input/outputs and user applications' desired input and output data.
In general, inputs and outputs of a neural network model are tensors. In applications,
user sample data are images in standard formats like PNG or JPEG, and
users also need to convert the output tensors to struct data for user applications.

A `Model` object also provides both `preprocess()` and `postprocess()` methods.
They are utilities to convert easily user inputs to the model's input tensors and output tensors
to struct data which can be easily accessible by applications.
If using pre-built pre/postprocessing methods, users can quickly start using `furiosa-models`.

In sum, typical steps of a single inference is as the following, as also shown at [examples](#Examples).

1. Call `preprocess()` with user inputs (e.g., image files)
2. Pass an output of `preprocess()` to `Session.run()`
3. Pass the output of the model to `postprocess()`


!!!Info
    Default postprocessing implementations are in Python.
    However, some models have the native postprocessing implemented in Rust and C++ and
    optimized for FuriosaAI Warboy and Intel/AMD CPUs.
    Python implementations can run on CPU and GPU as well, whereas
    the native postprocessor implementations works with only FuriosaAI NPU.
    Native implementations are designed to leverage FuriosaAI NPU's characteristics even for post-processing
    and maximize the latency and throughput by using modern CPU architecture,
    such as CPU cache, SIMD instructions and CPU pipelining.
    According to our benchmark, the native implementations show at most 70% lower latency.

    To use native post processor, please pass `postprocessor_type=Platform.RUST` to `Model()`.

    The following is an example to use native post processor for [SSDMobileNet](models/ssd_mobilenet.md).
    You can find more details of each model page.

    !!!Example
        ```python
        --8<-- "docs/examples/ssd_mobilenet_native.py"
        ```

## See Also

* [Furiosa SDK Documentation](https://furiosa-ai.github.io/docs/latest/en/)
