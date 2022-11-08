from time import perf_counter
from typing import Callable, List, Optional

from tqdm import tqdm

from furiosa.runtime import session

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


def pretty_task_type(model: Model):
    """Returns pretty string for model's task type

    Args:
        model: Model class instance

    Returns:
        Prettified string for model's task type
    """
    return " ".join(
        map(lambda x: x.capitalize(), model.__fields__["task_type"].default.name.split("_"))
    )


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
        # Model name, description, task type
        model_list.append([model_name, model.__doc__, pretty_task_type(model)])
    return model_list


def get_model(model_name: str) -> Optional[Model]:
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


def run_inference(model: Model, input_paths: List[str]):
    net_inference_times = []
    model = model.load()
    print(f"Running {len(input_paths)} inferences")
    with session.create(model) as sess:
        initial_time = perf_counter()
        for input_path in tqdm(input_paths):
            input, context = model.preprocess(input_path)
            start_time = perf_counter()
            model_output = sess.run(input)
            net_inference_times.append(perf_counter() - start_time)
            model_output = model_output.numpy()  # To avoid calling __getitem__
            _final_output = model.postprocess(model_output, context)
        total_time_elapsed = perf_counter() - initial_time
    print(f"Total time elapsed: {total_time_elapsed:.5f} sec")
    average = sum(net_inference_times) / len(net_inference_times)
    print(f"Average inference time: {average * 1000:.5f} msec")
    print(f"Throughput: {1 / average:.5f} inferences/sec")
