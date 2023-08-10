from time import perf_counter
from typing import Callable, List, Optional, Sequence, Type

from tqdm import tqdm

from .. import vision
from ..types import Model, Platform


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


def decorate_with_bar(string: str) -> str:
    bar = "----------------------------------------------------------------------"
    return "\n".join([bar, string, bar])


def time_with_proper_suffix(t: float, digits: int = 5) -> str:
    units = iter(["sec", "ms", "us", "ns"])
    while t < 1:
        t *= 1_000
        next(units)
    return f"{t:.{digits}f} {next(units)}"


def decorate_result(
    total_time: float, queries: int, header: str = "", digits: int = 5, newline: bool = True
) -> str:
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


def run_inferences(model_cls: Type[Model], input_paths: Sequence[str], postprocess: Optional[str]):
    from furiosa.runtime.sync import create_runner

    warning = """WARN: the benchmark results may depend on the number of input samples,
sizes of the images, and a machine where this benchmark is running."""
    if postprocess:
        model = model_cls(postprocessor_type=postprocess)
    else:
        model = model_cls()
    queries = len(input_paths)
    use_native = model.postprocessor_type != Platform.PYTHON
    print(f"Running {queries} input samples ...")
    print(decorate_with_bar(warning))
    with create_runner(model.model_source()) as runner:
        model_inputs, model_outputs = [], []
        initial_time = perf_counter()
        for input_path in tqdm(input_paths, desc="Preprocess"):
            model_inputs.append(model.preprocess(input_path))
        after_preprocess = perf_counter()
        for model_input in tqdm(model_inputs, desc="Inference"):
            model_outputs.append([runner.run(model_input[0]), model_input[1]])
        after_npu = perf_counter()
        for contexted_model_output in tqdm(model_outputs, desc="Postprocess"):
            model_output, context = contexted_model_output
            # FIXME: Only YOLO can handle multiple contexts
            use_native = (
                False if isinstance(model, (vision.YOLOv5m, vision.YOLOv5l)) else use_native
            )
            context = context[0] if context is not None and use_native else context
            model.postprocess(model_output, context)
        all_done = perf_counter()

    print(
        decorate_result(all_done - initial_time, queries, "Preprocess -> Inference -> Postprocess")
    )
    print(decorate_result(all_done - after_preprocess, queries, "Inference -> Postprocess"))
    print(decorate_result(after_npu - after_preprocess, queries, "Inference", newline=False))
