from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple, Type, Union

import aiofiles
import aiohttp
from pydantic import BaseModel
import yaml

from furiosa.common.native import DEFAULT_ENCODING, find_native_libs
from furiosa.common.thread import synchronous

from . import errors

EXT_CALIB_YAML = "calib_range.yaml"
EXT_ENF = "enf"
EXT_ONNX = "onnx"
DATA_DIRECTORY_BASE = Path(__file__).parent / "data"
CACHE_DIRECTORY_BASE = Path(
    os.getenv(
        "FURIOSA_MODELS_CACHE_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"), "furiosa/models"),
    )
)
DVC_PUBLIC_HTTP_ENDPOINT = (
    "https://furiosa-public-artifacts.s3-accelerate.amazonaws.com/furiosa-artifacts"
)

module_logger = logging.getLogger(__name__)


@dataclass
class CompilerVersion:
    version: str
    revision: str


def get_field_default(model: Type[BaseModel], field: str) -> Any:
    """Returns field's default value from BaseModel cls

    Args:
        model: A pydantic BaseModel cls

    Returns:
        Pydantic class' default field value
    """
    return model.__fields__[field].default


def get_nux_version() -> Optional[CompilerVersion]:
    # TODO - hacky version. Eventually,
    #  it should find a compiler version being used by runtime.
    libnux = find_native_libs("nux")
    if libnux is None:
        return None
    return CompilerVersion(
        libnux.version().decode(DEFAULT_ENCODING),
        libnux.git_short_hash().decode(DEFAULT_ENCODING),
    )


def get_version_info() -> Optional[str]:
    version_info = get_nux_version()
    if not version_info:
        return None
    return f"{version_info.version}_{version_info.revision}"


def removesuffix(base: str, suffix: str) -> str:
    # Copied from https://github.com/python/cpython/blob/6dab8c95/Tools/scripts/deepfreeze.py#L105-L108
    if base.endswith(suffix):
        return base[: len(base) - len(suffix)]
    return base


class ArtifactResolver:
    def __init__(self, uri: Union[str, Path]):
        self.uri = Path(uri)
        # Note: DVC_REPO is to locate local DVC directory not remote git repository
        self.dvc_cache_path = os.environ.get("DVC_REPO", self.find_dvc_cache_directory(Path.cwd()))
        if self.dvc_cache_path is not None:
            self.dvc_cache_path = Path(self.dvc_cache_path)
            if self.dvc_cache_path.is_symlink():
                self.dvc_cache_path = self.dvc_cache_path.readlink()
            module_logger.debug(f"Found DVC cache directory: {self.dvc_cache_path}")

    @classmethod
    def find_dvc_cache_directory(cls, path: Path) -> Optional[Path]:
        if path is None or path == path.parent:
            return None
        if (path / ".dvc").is_dir():
            return path / ".dvc" / "cache"
        return cls.find_dvc_cache_directory(path.parent)

    @staticmethod
    def parse_dvc_file(file_path: Path) -> Tuple[str, str, int]:
        info_dict = yaml.safe_load(open(f"{file_path}.dvc").read())["outs"][0]
        md5sum = info_dict["md5"]
        return md5sum[:2], md5sum[2:], info_dict["size"]

    @staticmethod
    def get_url(
        directory: str, filename: str, http_endpoint: str = DVC_PUBLIC_HTTP_ENDPOINT
    ) -> str:
        return f"{http_endpoint}/{directory}/{filename}"

    async def _read(self, directory: str, filename: str) -> bytes:
        # Try to find local cached file
        local_cache_path = CACHE_DIRECTORY_BASE / get_version_info() / (self.uri.name)
        if local_cache_path.exists():
            module_logger.debug(f"Local cache exists: {local_cache_path}")
            async with aiofiles.open(local_cache_path, mode="rb") as f:
                return await f.read()

        # Try to find real file along with DVC file (no DVC)
        if Path(self.uri).exists():
            module_logger.debug(f"Local file exists: {self.uri}")
            async with aiofiles.open(self.uri, mode="rb") as f:
                return await f.read()

        module_logger.debug(f"{self.uri} not exists, resolving DVC")
        if self.dvc_cache_path is not None:
            cached: Path = self.dvc_cache_path / directory / filename
            if cached.exists():
                module_logger.debug(f"DVC cache hit: {cached}")
                async with aiofiles.open(cached, mode="rb") as f:
                    return await f.read()
            else:
                module_logger.debug(f"DVC cache directory exists, but not having {self.uri}")

        # Fetching from remote
        async with aiohttp.ClientSession() as session:
            url = self.get_url(directory, filename)
            module_logger.debug(f"Fetching from remote: {url}")
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise errors.NotFoundInDVCRemote(self.uri, f"{directory}{filename}")
                data = await resp.read()
                caching_path = CACHE_DIRECTORY_BASE / get_version_info() / (self.uri.name)
                module_logger.debug(f"caching to {caching_path}")
                caching_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(caching_path, mode="wb") as f:
                    await f.write(data)
                return data

    async def read(self) -> bytes:
        directory, filename, size = self.parse_dvc_file(self.uri)
        data = await self._read(directory, filename)
        assert len(data) == size
        return data


def resolve_file(src_name: str, extension: str, num_pe: int = 2) -> bytes:
    # First check whether it is generated file or not
    if extension == EXT_ENF:
        version_info = get_version_info()
        if version_info is None:
            raise errors.VersionInfoNotFound()
        generated_path_base = f"generated/{version_info}"
        file_name = f'{src_name}_warboy_{num_pe}pe.{extension}'
        full_path = DATA_DIRECTORY_BASE / f'{generated_path_base}/{file_name}'
    else:
        full_path = next((DATA_DIRECTORY_BASE / src_name).glob(f'*.{extension}.dvc'))
        # Remove `.dvc` suffix
        full_path = full_path.with_suffix('')

    try:
        return synchronous(ArtifactResolver(full_path).read)()
    except Exception as e:
        raise errors.ArtifactNotFound(f"{src_name}:{full_path}") from e
