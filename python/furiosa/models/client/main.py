import argparse
import logging
from pathlib import Path
import sys
from typing import List

from tabulate import tabulate

from . import api
from ..types import ImageClassificationModel, ObjectDetectionModel

logger = logging.getLogger(__name__)

EXAMPLE: str = """example:
    # List available models
    furiosa-models list

    # List Object Detection models
    furiosa-models list -t detect

    # List available pre/post-processes for SSDResNet34
    furiosa-models list --model SSDResNet34

    # Run SSDResNet34 for images in `./input` directory
    furiosa-models run -m ssdresnet34 ./input/
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

    list_parser = subparsers.add_parser(
        "list", help="See available models or pre/post-processes list"
    )
    list_parser.add_argument(
        "-t", "--type", type=str, help="Limits the task type (ex. classify, detect)"
    )
    list_parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model name: if this argument is given, it will show the available pre/post processes",
    )

    inference_parser = subparsers.add_parser("run", help="Run Inference")
    inference_parser.add_argument("-pre", "--preprocess", type=str, help="Set preprocess type")
    inference_parser.add_argument("-post", "--postprocess", type=str, help="Set postprocess type")
    inference_parser.add_argument(
        "-m", "--model", type=str, help="Model name (ignore case)", required=True
    )
    inference_parser.add_argument("input", type=str, help="Input path (file or directory)")

    return parser.parse_args()


def print_model_list(table: List[List[str]]):
    header = ["Model name", "Model description", "Task type"]
    print(tabulate(table, headers=header, tablefmt="pretty"))


def resolve_input_paths(input_path: Path) -> List[str]:
    """Create input file list"""
    if input_path.is_file():
        return [str(input_path)]
    elif input_path.is_dir():
        # Directory may containing image files
        image_extensions = {".jpg", ".jpeg", ".png"}
        return [str(p) for p in input_path.glob("**/*") if p.suffix.lower() in image_extensions]
    else:
        # Error
        print("error: invalid input path")
        return sys.exit(1)


def main():
    args = parse_args()
    command: str = args.command
    filter_type: str = args.type
    model_name: str = args.model

    if command == "list":
        if filter_type is None:
            filter_func = lambda _: True
        elif "detect" in filter_type.lower():
            filter_func = lambda x: issubclass(x, ObjectDetectionModel)
        elif "classif" in filter_type.lower():
            filter_func = lambda x: issubclass(x, ImageClassificationModel)
        else:
            filter_func = lambda _: True
            logger.warn(f"Unknwon type filter {filter_type}, showing all models...")
        print_model_list(api.get_model_list(filter_func=filter_func))
    elif command == "run":
        input_path: str = args.input
        input_paths = resolve_input_paths(Path(input_path))
        model = api.get_model(model_name)
        if model is None:
            logging.warn(f"Can't find model named {model}")
        api.run_inference(model, input_paths)
