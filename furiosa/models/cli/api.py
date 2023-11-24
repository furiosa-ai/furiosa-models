from functools import reduce
import os
from time import perf_counter
from typing import Any, Callable, List, Optional, Sequence, Type

from tqdm import tqdm

from furiosa import device
from furiosa.common.thread import asynchronous
from furiosa.device import furiosa_native_device_sync as device_sync
from furiosa.runtime.sync import create_runner

from .. import vision
from ..types import Model


def normalize(text: str) -> str:
    """Returns only casefolded alphabets and numbers

    Args:
        text: string to be normalized

    Returns:
        Normalized text
    """
    text = "".join(filter(lambda c: c.isalnum(), text))
    return text.casefold()


def prettified_task_type(model: Type[Model]):
    """Returns pretty string for model's task type

    Args:
        model: Model class instance

    Returns:
        Prettified string for model's task type
    """
    task_type = model.task_type.name
    return " ".join(map(lambda x: x.capitalize(), task_type.split("_")))


def get_model_list(filter_func: Optional[Callable[..., bool]] = None) -> List[List[str]]:
    """Get list of available model names and descriptions

    Args:
        filter_func: Filter function for model class

    Returns:
        List of available model names and descriptions
    """
    filter_func = filter_func or (lambda _: True)
    model_list = []
    for model_name in vision.__all__:
        model_cls = getattr(vision, model_name)
        if not filter_func(model_cls):
            continue
        model = model_cls()
        postproc_map = model.postprocessor_map
        if not postproc_map:
            raise ValueError(f"No postprocessor map found for {model_name.capitalize()}")
        postprocs = ', '.join(map(lambda x: x.name.capitalize(), postproc_map.keys()))
        # Model name, description, task type, available post process implementations
        model_list.append([model_name, model.__doc__, prettified_task_type(model), postprocs])
    return model_list


def get_model(model_name: str) -> Optional[Type[Model]]:
    """Get model for given model name string

    Args:
        model_name: Model name

    Returns:
        A model which have name of the given argument
    """
    for name in vision.__all__:
        if normalize(name) == normalize(model_name):
            return getattr(vision, name)
    return None


def get_device_mode(device_str: str) -> device.DeviceMode:
    """Get device type for given device type string

    Args:
        device_type: Device type string

    Returns:
        Device type
    """
    device_config = device.DeviceConfig.from_str(device_str)
    device_list = device_sync.find_device_files(device_config)
    if not device_list:
        raise ValueError(f"No device found for {device_str}")
    assert reduce(
        lambda x, y: x == y, map(lambda x: x.mode(), device_list)
    ), "All devices must have same device type"
    return device_list[0].mode()


def device_file_to_pe_count(device_mode: device.DeviceMode) -> int:
    """Get number of PEs for given device mode

    Args:
        device_mode: DeviceMode

    Returns:
        Number of PEs
    """
    if device_mode == device.DeviceMode.Single:
        return 1
    elif device_mode == device.DeviceMode.Fusion:
        return 2
    else:
        raise ValueError(f"Unsupported device mode: {device_mode}")


def get_pe_count_from_device_str(device_str: Optional[str]) -> int:
    """Get number of PEs for given device string

    Args:
        device_str: Device string

    Returns:
        Number of PEs
    """
    device_str = device_str or os.environ.get(
        "FURIOSA_DEVICES", os.environ.get("NPU_DEVNAME", None)
    )
    if device_str:
        device_mode = get_device_mode(device_str)
        return device_file_to_pe_count(device_mode)
    else:
        return 2


def decorate_with_bar(string: str) -> str:
    """Decorate given string with bar

    Args:
        string: String to decorate

    Returns:
        Decorated string
    """

    bar = "----------------------------------------------------------------------"
    return "\n".join([bar, string, bar])


def time_with_proper_suffix(t: float, digits: int = 5) -> str:
    """Returns time with proper suffix

    Args:
        t: Time in seconds
        digits: Number of digits after decimal point

    Returns:
        Time with proper suffix
    """

    units = iter(["sec", "ms", "us", "ns"])
    while t < 1:
        t *= 1_000
        next(units)
    return f"{t:.{digits}f} {next(units)}"


def decorate_result(
    total_time: float, queries: int, header: str = "", digits: int = 5, newline: bool = True
) -> str:
    """Decorate benchmark result

    Args:
        total_time: Total elapsed time
        queries: Number of queries
        header: Header string
        digits: Number of digits after decimal point
        newline: Whether to add newline at the end

    Returns:
        Decorated benchmark result
    """

    result = []
    result.append(decorate_with_bar(header))
    result.append(f"Total elapsed time: {time_with_proper_suffix(total_time, digits)}")
    result.append(f"QPS: {queries / total_time:.{digits}f}")
    result.append(
        f"Avg. elapsed time / sample: {time_with_proper_suffix(total_time / queries, digits)}"
    )
    if newline:
        result.append("")
    return "\n".join(result)


def run_inferences(
    model_cls: Type[Model],
    input_paths: Sequence[str],
    postprocess: Optional[str],
    device_str: Optional[str],
):
    """Run inferences on given model

    Args:
        model_cls: Model class
        input_paths: Input paths
        postprocess: Postprocess implementation
        device_str: Device string
    """

    warning = """WARN: the benchmark results may depend on the number of input samples,
sizes of the images, and a machine where this benchmark is running."""
    num_pe = get_pe_count_from_device_str(device_str)
    model = model_cls(postprocessor_type=postprocess) if postprocess else model_cls()
    queries = len(input_paths)
    print(f"Run benchmark on {queries} input samples ...")

    print(decorate_with_bar(warning))
    with create_runner(model.model_source(num_pe=num_pe), device=device_str) as runner:
        model_inputs, model_outputs = [], []
        initial_time = perf_counter()
        for input_path in tqdm(input_paths, desc="Preprocess"):
            model_inputs.append(model.preprocess(input_path))
        after_preprocess = perf_counter()
        for contexted_model_input in tqdm(model_inputs, desc="Inference"):
            model_input, context = contexted_model_input
            model_outputs.append([runner.run(model_input), context])
        after_npu = perf_counter()
        for contexted_model_output in tqdm(model_outputs, desc="Postprocess"):
            model_output, context = contexted_model_output
            model.postprocess(model_output, context)
        all_done = perf_counter()

    print(
        decorate_result(all_done - initial_time, queries, "Preprocess -> Inference -> Postprocess")
    )
    print(decorate_result(all_done - after_preprocess, queries, "Inference -> Postprocess"))
    print(decorate_result(after_npu - after_preprocess, queries, "Inference", newline=False))


def serve_model(
    model_cls: Type[Model],
    postprocess: Optional[str],
    host: str,
    port: int,
    device_str: Optional[str],
) -> Any:
    """Serve model

    Args:
        model_cls: Model class
        postprocess: Postprocess implementation
        host: Host address
        port: Port number
        device_str: Device string
    """

    num_pe = get_pe_count_from_device_str(device_str)

    try:
        from fastapi import FastAPI, File, UploadFile

        from furiosa.serving import ServeAPI, ServeModel
    except ImportError:
        raise ImportError("Please install `furiosa-serving` to use this command")

    from tempfile import NamedTemporaryFile

    import numpy as np
    import uvicorn

    from furiosa.common.thread import synchronous

    serve = ServeAPI()
    app: FastAPI = serve.app
    model = model_cls(postprocessor_type=postprocess) if postprocess else model_cls()

    # ServeModel does not support in-memory model binary for now,
    # so we write model into temp file and pass its path
    model_file = NamedTemporaryFile()
    model_file.write(model.model_source(num_pe=num_pe))
    model_file_path = model_file.name

    serve_model: ServeModel = synchronous(serve.model("furiosart"))(
        model.name, location=model_file_path, npu_device=device_str
    )

    @serve_model.post("/infer")
    async def infer(image: UploadFile = File(...)):
        # Model Zoo's preprocesses do not consider in-memory image file for now
        # (note that it's different from in-memory tensor)
        # so we write in-memory image into temp file and pass its path
        image_file_path = NamedTemporaryFile()
        image_file_path.write(await image.read())

        tensors, ctx = await asynchronous(model.preprocess)(image_file_path.name)

        # Infer from ServeModel
        result: List[np.ndarray] = await serve_model.predict(tensors)

        return {"result": model.postprocess(result, ctx)}

    return uvicorn.run(app, host=host, port=port)
