from time import perf_counter
from typing import Callable, List, Optional, Sequence, Type

from tqdm import tqdm

from .. import vision
from ..types import Model
from ..utils import get_field_default


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
    task_type = get_field_default(model, "task_type").name
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
        model = getattr(vision, model_name)
        if not filter_func(model):
            continue
        postprocs = ', '.join(
            map(lambda x: x.name.capitalize(), get_field_default(model, "postprocessor_map").keys())
        )
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
    from furiosa.runtime import session

    warning = """WARN: the benchmark results may depend on the number of input samples,
sizes of the images, and a machine where this benchmark is running."""
    postprocess = postprocess and postprocess.lower()
    use_native = postprocess != "python"
    if issubclass(model_cls, (vision.YOLOv5l, vision.YOLOv5m)):
        use_native = False
    model = model_cls.load(use_native=use_native)
    queries = len(input_paths)
    print(f"Running {queries} input samples ...")
    print(decorate_with_bar(warning))
    sess, queue = session.create_async(model)
    model_inputs, model_outputs = [], []
    initial_time = perf_counter()
    for input_path in tqdm(input_paths, desc="Preprocess"):
        model_inputs.append(model.preprocess(input_path))
    after_preprocess = perf_counter()
    for idx, (model_input, ctx) in enumerate(model_inputs):
        sess.submit(model_input, context=idx)
    for _ in tqdm(range(queries), desc="Inference"):
        model_outputs.append(queue.recv())
    after_npu = perf_counter()
    for ctx, model_output in tqdm(model_outputs, desc="Postprocess"):
        contexts = model_inputs[ctx][1]
        contexts = contexts[0] if contexts is not None and use_native else contexts
        model.postprocess(model_output.numpy(), contexts)
    all_done = perf_counter()
    sess.close()

    print(
        decorate_result(all_done - initial_time, queries, "Preprocess -> Inference -> Postprocess")
    )
    print(decorate_result(all_done - after_preprocess, queries, "Inference -> Postprocess"))
    print(decorate_result(after_npu - after_preprocess, queries, "Inference", newline=False))
