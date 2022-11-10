from time import perf_counter
from typing import Callable, List, Optional, Sequence, Type

from tqdm import tqdm

from furiosa.runtime import session

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


def run_inferences(model_cls: Type[Model], input_paths: Sequence[str], postprocess: Optional[str]):
    postprocess = postprocess and postprocess.lower()
    use_native = postprocess != "python"
    model = model_cls.load(use_native=use_native, version=postprocess)
    queries = len(input_paths)
    print(f"Running {queries} inferences")
    sess, queue = session.create_async(model)
    model_inputs, model_outputs = [], []
    initial_time = perf_counter()
    for input_path in tqdm(input_paths, desc="Preprocessing"):
        model_inputs.append(model.preprocess(input_path))
    after_preprocess = perf_counter()
    for idx, (model_input, ctx) in enumerate(model_inputs):
        sess.submit(model_input, context=idx)
    for _ in tqdm(range(queries), desc="Run inferences on NPU"):
        model_outputs.append(queue.recv())
    after_npu = perf_counter()
    for ctx, model_output in tqdm(model_outputs, desc="Postprocessing"):
        contexts = model_inputs[ctx][1]
        contexts = contexts[0] if contexts is not None and use_native else contexts
        model.postprocess(model_output.numpy(), contexts)
    all_done = perf_counter()
    sess.close()

    print(f"Ran total {queries} queries")
    print(f"Preprocess times: {(after_preprocess - initial_time):.5f} sec")
    print(f"NPU inference times: {(after_npu - after_preprocess):.5f} sec")
    print(f"Postprocess times: {(all_done - after_npu):.5f} sec")
    print(f"Overall w/ preprocessing: {(all_done - initial_time):.5f} sec")
    print(f"Overall w/o preprocessing: {(all_done - after_preprocess):.5f} sec")
    print(f"Overall qps w/o preprocessing: {queries / (all_done - after_preprocess):.5f} qps")
