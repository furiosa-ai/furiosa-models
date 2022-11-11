import argparse
import logging
from pathlib import Path
import sys
from typing import Callable, List, Optional

from tabulate import tabulate
import yaml

from . import api
from ..types import ImageClassificationModel, Model, ObjectDetectionModel

logger = logging.getLogger(__name__)

EXAMPLE: str = """example:
    # List available models
    furiosa-models list

    # List Object Detection models
    furiosa-models list -t detect

    # Describe SSDResNet34 model
    furiosa-models desc SSDResNet34

    # Run SSDResNet34 for images in `./input` directory
    furiosa-models run ssd-resnet34 ./input/
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="furiosa-models",
        epilog=EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        conflict_handler="resolve",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    list_parser = subparsers.add_parser("list", help="Lists available models")
    list_parser.add_argument(
        "-t", "--type", type=str, help="Limits the task type (ex. classify, detect)"
    )

    desc_parser = subparsers.add_parser("desc", help="Prints out a description of a model")
    desc_parser.add_argument("model", type=str, help="Model name (ignore case)")

    inference_parser = subparsers.add_parser(
        "bench", help="Benchmark a given model by running inferences"
    )
    inference_parser.add_argument("-v", "--verbose", action="store_true", help="Set verbose")
    inference_parser.add_argument(
        "-post", "--postprocess", type=str, help="Specifies a postprocess implementation"
    )
    inference_parser.add_argument("model", type=str, help="Model name (ignore case)")
    inference_parser.add_argument("input", type=str, help="Input path (file or directory)")

    return parser.parse_args()


def get_model_list(table: List[List[str]]):
    header = ["Model name", "Model description", "Task type", "Available postprocesses"]
    return tabulate(table, headers=header, tablefmt="pretty")


def resolve_input_paths(input_path: Path) -> List[str]:
    """Create input file list"""
    if input_path.is_file():
        return [str(input_path)]
    elif input_path.is_dir():
        # Directory may containing image files
        image_extensions = {".jpg", ".jpeg", ".png"}
        return [
            str(p.resolve())
            for p in input_path.glob("**/*")
            if p.suffix.lower() in image_extensions
        ]
    else:
        logger.warning(f"Invalid input path '{str(input_path)}'")
        sys.exit(1)


def get_filter(filter_type: Optional[str]) -> Callable[..., bool]:
    if filter_type is None:
        return lambda _: True
    elif "detect" in filter_type.lower():
        return lambda x: issubclass(x, ObjectDetectionModel)
    elif "classif" in filter_type.lower():
        return lambda x: issubclass(x, ImageClassificationModel)
    else:
        logger.warning(f"Unknown type filter '{filter_type}', showing all models...")
        return lambda _: True


def get_model_or_exit(model_name: str) -> Model:
    model = api.get_model(model_name)
    if model is None:
        logger.warning(f"Model name '{model_name}' not found")
        sys.exit(1)
    return model


def describe_model(model_cls: Model) -> str:
    # TODO: Make dry load (to avoid resolving heavy artifacts)
    model = model_cls.load()
    include = {'name', 'format', 'family', 'version', 'metadata'}
    output = []
    output.append(yaml.dump(model.dict(include=include)))
    output.append(f"task type: {api.prettified_task_type(model)}\n")
    available_postprocs = ', '.join(
        map(lambda x: x.name.capitalize(), model.postprocessor_map.keys())
    )
    output.append(f"available postprocess versions: {available_postprocs}")
    return ''.join(output)


def main():
    args = parse_args()
    command: str = args.command

    if command == "list":
        filter_type: str = args.type
        print(get_model_list(api.get_model_list(filter_func=get_filter(filter_type))))
    elif command == "desc":
        model_name: str = args.model
        model_cls = get_model_or_exit(model_name)
        print(describe_model(model_cls))
    elif command == "bench":
        model_name: str = args.model
        verbose: bool = args.verbose
        _input_paths: str = args.input
        postprocess: Optional[str] = args.postprocess
        if verbose:
            logging.root.setLevel(logging.DEBUG)
        input_paths = resolve_input_paths(Path(_input_paths))
        logger.debug(f"Collected input paths: {input_paths}")
        model_cls = get_model_or_exit(model_name)
        api.run_inferences(model_cls, input_paths, postprocess)
