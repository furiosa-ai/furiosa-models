import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Collection, Optional, Tuple, Union

if TYPE_CHECKING:
    from .types import Platform

import requests
import yaml

from . import errors

EXT_CALIB_YAML = "calib_range.yaml"
EXT_ONNX = "onnx"
DATA_DIRECTORY_BASE = Path(__file__).parent / "data"
CACHE_DIRECTORY_BASE = Path(
    os.getenv(
        "FURIOSA_MODELS_CACHE_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"), "furiosa/models"),
    )
)
DVC_PUBLIC_HTTP_ENDPOINT = (
    "https://furiosa-public-artifacts.s3-accelerate.amazonaws.com/furiosa-artifacts/"
)

module_logger = logging.getLogger(__name__)


def get_version_info() -> Optional[str]:
    try:
        from furiosa.native_runtime import __git_short_hash__, __version__

        return f"{__version__}_{__git_short_hash__}"
    except ImportError:
        return None


def find_dvc_cache_directory(path: Path) -> Optional[Path]:
    if path is None or path == path.parent:
        return None
    if (path / ".dvc").is_dir():
        return path / ".dvc" / "cache"
    return find_dvc_cache_directory(path.parent)


def parse_dvc_file(file_path: Path) -> Tuple[str, str, int]:
    info_dict = yaml.safe_load(open(f"{file_path}.dvc").read())["outs"][0]
    md5sum = info_dict["md5"]
    return md5sum[:2], md5sum[2:], info_dict["size"]


def get_from_url(path: str, uri: Path, is_legacy_path: bool = False) -> bytes:
    url = f"{DVC_PUBLIC_HTTP_ENDPOINT}{path}"
    module_logger.debug(f"Fetching from remote: {url}")
    with requests.get(url) as resp:
        if resp.status_code != 200:
            if not is_legacy_path:
                # New dvc now stores data into /files/md5
                return get_from_url(f"files/md5/{path}", uri, True)
            raise errors.NotFoundInDVCRemote(uri, path)
        data = resp.content
        caching_path = CACHE_DIRECTORY_BASE / get_version_info() / (uri.name)
        module_logger.debug(f"caching to {caching_path}")
        caching_path.parent.mkdir(parents=True, exist_ok=True)
        with open(caching_path, mode="wb") as f:
            f.write(data)
        return data


class ArtifactResolver:
    def __init__(self, uri: Union[str, Path]):
        self.uri = Path(uri)
        # Note: DVC_REPO is to locate local DVC directory not remote git repository
        self.dvc_cache_path = os.environ.get("DVC_REPO", find_dvc_cache_directory(Path.cwd()))
        if self.dvc_cache_path is not None:
            self.dvc_cache_path = Path(self.dvc_cache_path)
            if self.dvc_cache_path.is_symlink():
                self.dvc_cache_path = self.dvc_cache_path.readlink()
            module_logger.debug(f"Found DVC cache directory: {self.dvc_cache_path}")

    def _read(self, directory: str, filename: str) -> bytes:
        # Try to find local cached file
        local_cache_path = CACHE_DIRECTORY_BASE / get_version_info() / (self.uri.name)
        if local_cache_path.exists():
            module_logger.debug(f"Local cache exists: {local_cache_path}")
            with open(local_cache_path, mode="rb") as f:
                return f.read()

        # Try to find real file along with DVC file (no DVC)
        if Path(self.uri).exists():
            module_logger.debug(f"Local file exists: {self.uri}")
            with open(self.uri, mode="rb") as f:
                return f.read()

        module_logger.debug(f"{self.uri} not exists, resolving DVC")
        if self.dvc_cache_path is not None:
            cached: Path = self.dvc_cache_path / directory / filename
            if cached.exists():
                module_logger.debug(f"DVC cache hit: {cached}")
                with open(cached, mode="rb") as f:
                    return f.read()
            else:
                module_logger.debug(f"DVC cache directory exists, but not having {self.uri}")

        # Fetching from remote
        return get_from_url(f"{directory}/{filename}", self.uri)

    def read(self) -> bytes:
        directory, filename, size = parse_dvc_file(self.uri)
        data = self._read(directory, filename)
        assert len(data) == size
        return data


def resolve_artifact(src_name: str, full_path: Path) -> bytes:
    try:
        return ArtifactResolver(full_path).read()
    except Exception as e:
        raise errors.ArtifactNotFound(f"{src_name}:{full_path}") from e


def resolve_source(src_name: str, extension: str) -> bytes:
    full_path = next((DATA_DIRECTORY_BASE / src_name).glob(f'*.{extension}.dvc'))
    # Remove `.dvc` suffix
    full_path = full_path.with_suffix('')
    return resolve_artifact(src_name, full_path)


def resolve_model_source(src_name: str, num_pe: int = 2) -> bytes:
    version_info = get_version_info()
    if version_info is None:
        raise errors.VersionInfoNotFound()
    generated_path_base = DATA_DIRECTORY_BASE / f"generated/{version_info}"
    if not generated_path_base.exists():
        module_logger.warning("ENF does not exist. Trying to generate from source..")

        try:
            import onnx
            import yaml

            from furiosa.quantizer import ModelEditor, TensorType, get_pure_input_names, quantize
        except ImportError:
            raise errors.ExtraPackageRequired()
        module_logger.warning(f"Returning quantized ONNX for {src_name}")
        onnx_model = onnx.load_from_string(resolve_source(src_name, EXT_ONNX))
        calib_range = yaml.full_load(resolve_source(src_name, EXT_CALIB_YAML))
        editor = ModelEditor(onnx_model)
        for input_name in get_pure_input_names(onnx_model):
            editor.convert_input_type(input_name, TensorType.UINT8)
        return quantize(onnx_model, calib_range)
    file_name = f'{src_name}_warboy_{num_pe}pe.enf'
    return resolve_artifact(src_name, generated_path_base / file_name)


def validate_postprocessor_type(
    postprocessor_type: "Platform", postprocessor_map: Collection["Platform"]
):
    if postprocessor_type not in postprocessor_map:
        raise ValueError(
            f"Not supported postprocessor type: {postprocessor_type}, "
            f"Available choices: {', '.join(postprocessor_map)}"
        )
