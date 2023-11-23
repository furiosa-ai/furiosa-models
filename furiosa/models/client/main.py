from pathlib import Path
from typing import Callable, List, Optional, Type

from tabulate import tabulate
import typer
import yaml

from . import api
from .. import __version__ as models_version
from ..types import ImageClassificationModel, Model, ObjectDetectionModel, PoseEstimationModel

EXAMPLE: str = """Examples:\n\n\n
# List available models\n
`furiosa-models list`\n\n\n
# List Object Detection models\n
`furiosa-models list -t detect`\n\n\n
# Describe SSDResNet34 model\n
`furiosa-models desc SSDResNet34`\n\n\n
# Run SSDResNet34 for images in `./input` directory\n
`furiosa-models bench ssd-resnet34 ./input/`
"""

app = typer.Typer(
    help=f"FuriosaAI Model Zoo CLI --- v{models_version}", epilog=EXAMPLE, add_completion=False
)


def get_model_list(table: List[List[str]]):
    header = ["Model name", "Model description", "Task type", "Available postprocesses"]
    return tabulate(table, headers=header, tablefmt="pretty")


def resolve_input_paths(input_path: Path) -> List[str]:
    """Create input file list"""
    if input_path.is_file():
        return [str(input_path)]
    elif input_path.is_dir():
        # Directory may contain image files
        image_extensions = {".jpg", ".jpeg", ".png"}
        return [
            str(p.resolve())
            for p in input_path.glob("**/*")
            if p.suffix.lower() in image_extensions
        ]
    else:
        typer.echo(f"Invalid input path '{str(input_path)}'")
        raise typer.Exit(1)


def get_filter(filter_type: Optional[str]) -> Callable[..., bool]:
    if filter_type is None:
        return lambda _: True
    elif "detect" in filter_type.lower():
        return lambda x: issubclass(x, ObjectDetectionModel)
    elif "classif" in filter_type.lower():
        return lambda x: issubclass(x, ImageClassificationModel)
    elif "pose" in filter_type.lower():
        return lambda x: issubclass(x, PoseEstimationModel)
    else:
        typer.echo(f"Unknown type filter '{filter_type}', showing all models...")
        return lambda _: True


def get_model_or_exit(model_name: str) -> Model:
    model = api.get_model(model_name)
    if model is None:
        typer.echo(f"Model name '{model_name}' not found")
        raise typer.Exit(1)
    return model


def describe_model(model_cls: Type[Model]) -> str:
    model = model_cls()
    include = {"name", "format", "family", "version", "metadata", "tags"}
    output = []
    output.append(yaml.dump(model.model_dump(include=include, exclude_none=True), sort_keys=False))
    output.append(f"task type: {api.prettified_task_type(model)}\n")
    available_postprocs = ', '.join(
        map(lambda x: x.name.capitalize(), model.postprocessor_map.keys())
    )
    output.append(f"available postprocess versions: {available_postprocs}")
    return ''.join(output)


@app.command("list", help="List available models")
def list_models(
    filter_type: Optional[str] = typer.Argument(
        None, help="Limits the task type (ex. classify, detect, pose)"
    )
):
    typer.echo(get_model_list(api.get_model_list(filter_func=get_filter(filter_type))))


@app.command("desc", help="Describe a model")
def describe_model_cmd(model_name: str):
    model_cls = get_model_or_exit(model_name)
    typer.echo(describe_model(model_cls))


@app.command("bench", help="Run benchmark on a model")
def benchmark_model(
    model: str,
    input_path: Path,
    postprocess: Optional[str] = typer.Option(
        None, "--postprocess", help="Specifies a postprocess implementation"
    ),
):
    input_paths = resolve_input_paths(Path(input_path))
    if len(input_paths) == 0:
        typer.echo(f"No input files found in '{input_path}'")
        raise typer.Exit(code=1)
    typer.echo(f"Collected input paths: {input_paths}")
    model_cls = get_model_or_exit(model)
    api.run_inferences(model_cls, input_paths, postprocess)


if __name__ == "__main__":
    app()
