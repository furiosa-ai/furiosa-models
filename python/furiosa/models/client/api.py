import time
from typing import Callable, List, Optional

from furiosa.models import vision
from furiosa.models.vision import resnet50
from furiosa.runtime import session

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


def get_model_list(filter_func: Optional[Callable[..., bool]] = None) -> List[List[str]]:
    """Returns list of available model names and descriptions

    Args:
        filter_func: Filter function, called on each models and filter

    Returns:
        List of available model names and descriptions
    """
    filter_func = filter_func or (lambda _: True)
    model_list = []
    for model_name in vision.__all__:
        model = getattr(vision, model_name)
        if not filter_func(model):
            continue
        task_type = " ".join(
            map(lambda x: x.capitalize(), model.__fields__["task_type"].default.name.split("_"))
        )
        # Model name, description, task type
        model_list.append([model_name, model.__doc__, task_type])
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
    model_inner = model.load()
    with session.create(model_inner) as sess:
        for input_path in input_paths:
            # FIXME: Get pre/post processes gracefully?
            input = resnet50.preprocess(input_path)
            start_time = time.perf_counter()
            output = sess.run(input)
            net_inference_times.append(time.perf_counter() - start_time)
            output = output.numpy()  # To avoid calling __getitem__
            output = resnet50.postprocess(output)
            print(output)
    average = sum(net_inference_times) / len(net_inference_times)
    print(f"Average inference time: {average * 1000} ms")
    print(f"Throughput: {1 / average}")
