import argparse
from pathlib import Path
import sys

EXAMPLE: str = """example:
    # List available models
    furiosa-models list

    # List Object Detection models
    furiosa-models list -t detect

    # List available pre/post-processes for ResNet34
    furiosa-models list -n ResNet34

    # Run ResNet34 for images in `input` directory
    furiosa-models classify -n ResNet34 -i input/
"""

parser = argparse.ArgumentParser(
    prog="furiosa-models",
    epilog=EXAMPLE,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
subparsers = parser.add_subparsers(dest="command", required=True)

# Note that common_parser is not used for `list` command
common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument(
    "-i", "--input", type=str, help="REQUIRED: Input path (file or directory)", required=True
)
common_parser.add_argument("-n", "--name", type=str, help="REQUIRED: Model name", required=True)
common_parser.add_argument("-pre", "--preprocess", type=str, help="Set preprocess type")
common_parser.add_argument("-post", "--postprocess", type=str, help="Set postprocess type")

list_parser = subparsers.add_parser(
    "list", help="See available model/preprocesses/postprocesses list"
)
list_parser.add_argument(
    "-t", "--type", type=str, help="Limits the task type (ex. classify, detect)"
)
list_parser.add_argument(
    "-n",
    "--name",
    type=str,
    help="Model name: if this argument is given, it will show the available pre/post processes",
)

classification_parser = subparsers.add_parser(
    "classify", parents=[common_parser], help="Run Image Classification"
)

detection_parser = subparsers.add_parser(
    "detect", parents=[common_parser], help="Run Object Detection"
)


def main():
    args = parser.parse_args()
    if args.command == "list":
        print("list")
        return

    # Process common arguments for running furiosa-models
    # Create input file lists
    input_path = Path(args.input)
    if input_path.is_file():
        input_paths = [input_path]
    elif input_path.is_dir():
        # Directory may containing image files
        image_extensions = {".jpg", ".jpeg", ".png"}
        input_paths = (p for p in input_path.glob("**/*") if p.suffix.lower() in image_extensions)
    else:
        # Error
        print("error: invalid input path")
        return sys.exit(1)

    if args.command == "classify":
        print("classify")
    elif args.command == "detect":
        print("detect")
