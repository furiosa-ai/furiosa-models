# Model object

In `furiosa-models` project, `Model` is the first class object, and it represents a neural network model. 
This document explains what [`Model`][furiosa.registry.Model] object offers and their usages.

## Loading a pre-trained model
To load a pre-trained neural-network model, you need to call `load()` method.
Since the sizes of pre-trained model weights vary from tens to hundreds megabytes, 
the model images are not included in Python package. When `load()` method is called, a pre-trained model will be 
fetched over network. It takes some time (usually few seconds) depending on models and environments. 
Once the model images are fetched, they will be cached on a local disk.

Non-blocking API `load_async()` also is available, and it can be used 
if your application is running through asynchronous executors (e.g., asyncio).

=== "Blocking API"
    ```python
    --8<-- "docs/examples/loading_model.py"
    ```

=== "Non-blocking API"
    ```python
    --8<-- "docs/examples/loading_model_nonblocking.py"
    ```

<a name="accessing_artifacts_and_metadata"></a>
## Accessing artifacts and metadata
A `Model` object includes model artifacts, such as ONNX, tflite, DFG, and ENF.

DFG and ENF are [FuriosaAI Compiler](https://furiosa-ai.github.io/docs/latest/en/software/compiler.html) specific formats.
Both formats are used for pre-compiled binary, and they are used to skip compilation times that take up to minutes.
In addition, a `Model` object has various metadata. The followings are all attributes belonging to a single `Model` object.

### `furiosa.registry.Model`
::: furiosa.registry.Model
    options:
        show_source: true


## Inferencing with Session API
To load a model to FuriosaAI NPU, you need to create a session instance with a `Model` object 
through Furiosa SDK. As we mentioned above, even a single `Model` object has multiple model artifacts, such as 
a ONNX model and an ENF (FuriosaAI's compiled program binary).

If an `Model` object is passed to `session.create()`, Session API chooses the ENF (FuriosaAI's Executable NPU Format) 
by default. In this case, `session.create()` doesn't involve any compilation because it uses the pre-compiled ENF binary.

!!!Info
    If you want to learn more about the installation of furiosa-sdk and how to use it, please follow the followings:

    * [Driver, Firmware, and Runtime Installation](https://furiosa-ai.github.io/docs/latest/en/software/installation.html)
    * [Python SDK Installation and User Guide](https://furiosa-ai.github.io/docs/latest/en/software/python-sdk.html)
    * [Furiosa SDK - Tutorial and Code Examples](https://furiosa-ai.github.io/docs/latest/en/software/tutorials.html)

Users still can compile source models like ONNX or tflite if passing `Model.source` to `session.create()`. 
Compiling models will take some time up to minutes, but it allows to specify batch size and compiler configs, 
leading to more optimizations depending on user use-cases. To learn more about `Model.source`, 
please refer to [Accessing artifacts and metadata](#accessing_artifacts_and_metadata).

<a name="Examples"></a>
!!! Example
    === "Using ENF binary"
        ```python
        --8<-- "docs/examples/ssd_mobilenet.py"
        ```
    
    === "Using ONNX model"
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

    To use native post processor, please pass `use_native=True` to `Model.load()` or `Model.load_async()`.
    The following is an example to use native post processor for [SSDMobileNet](models/ssd_mobilenet.md).
    You can find more details of each mode page.

    !!!Example
        ```python
        --8<-- "docs/examples/ssd_mobilenet_native.py"
        ```



# See Also
* [Furiosa SDK Documentation](https://furiosa-ai.github.io/docs/latest/en/)