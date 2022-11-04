from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import aiofiles
import aiohttp
import yaml

from furiosa.common.native import DEFAULT_ENCODING, find_native_lib_path, find_native_libs
from furiosa.common.thread import asynchronous

from . import errors

EXT_ONNX = "onnx"
EXT_ENF = "enf"
EXT_DFG = "dfg"

GENERATED_EXTENSIONS = (EXT_DFG, EXT_ENF)
DATA_DIRECTORY_BASE = Path(__file__).parent / "data"
CACHE_DIRECTORY_BASE = Path(
    os.getenv(
        "FURIOSA_MODELS_CACHE_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "furiosa/models"),
    )
)

DVC_PUBLIC_HTTP_ENDPOINT = "https://furiosa-public-artifacts.s3.amazonaws.com/furiosa-artifacts"

module_logger = logging.getLogger(__name__)


@dataclass
class CompilerVersion:
    version: str
    revision: str


def compiler_version() -> Optional[CompilerVersion]:
    # TODO - hacky version. Eventually,
    #  it should find a compiler version being used by runtime.
    if find_native_lib_path("nux") is None:
        return None
    else:
        libnux = find_native_libs("nux")
        return CompilerVersion(
            libnux.version().decode(DEFAULT_ENCODING),
            libnux.git_short_hash().decode(DEFAULT_ENCODING),
        )


def _generated_path_base() -> Optional[str]:
    version_info = compiler_version()
    if not version_info:
        return None
    return f"generated/{version_info.version}_{version_info.revision}"


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
    def parse_dvc_file(file_path: Path) -> Tuple[str, str]:
        md5sum = yaml.safe_load(open(f"{file_path}.dvc").read())["outs"][0]["md5"]
        return md5sum[:2], md5sum[2:]

    @staticmethod
    def get_url(
        directory: str, filename: str, http_endpoint: str = DVC_PUBLIC_HTTP_ENDPOINT
    ) -> str:
        return f"{http_endpoint}/{directory}/{filename}"

    async def read(self):
        # Try to find real file (no DVC)
        if Path(self.uri).exists():
            module_logger.debug(f"Local file exists: {self.uri}")
            async with aiofiles.open(self.uri, mode="rb") as f:
                return await f.read()

        module_logger.debug(f"{self.uri} not exists, resolving DVC")
        directory, filename = self.parse_dvc_file(self.uri)
        if self.dvc_cache_path is not None:
            # DVC Cache hit
            cached: Path = self.dvc_cache_path / directory / filename
            if cached.exists():
                module_logger.debug(f"DVC Cache hit: {cached}")
                async with aiofiles.open(cached, mode="rb") as f:
                    return await f.read()

        # Fetch from remote
        async with aiohttp.ClientSession() as session:
            url = self.get_url(directory, filename)
            module_logger.debug(f"Fetching from remote: {url}")
            async with session.get(url) as resp:
                return await resp.read()


def model_file_name(relative_path, truncated=True) -> str:
    suffix = "_truncated" if truncated else ""
    return relative_path + suffix


async def resolve_file(src_name: str, extension: str, generated_suffix="_warboy_2pe") -> bytes:
    # First check whether it is generated file or not
    if extension.lower() in GENERATED_EXTENSIONS:
        generated_path_base = _generated_path_base()
        if generated_path_base is None:
            # FIXME: Uncovered code path. Can we assume libnux is always installed?
            raise errors.VersionInfoNotFound()
        file_name = f'{src_name}{generated_suffix}.{extension}'
        file_subpath = f'{generated_path_base}/{file_name}'
    else:
        file_subpath = f'{src_name}.{extension}'
    full_path = (DATA_DIRECTORY_BASE / file_subpath).resolve()

    try:
        return await ArtifactResolver(full_path).read()
    except Exception as e:
        raise errors.ArtifactNotFound(src_name, extension) from e


async def load_artifacts(name: str) -> Dict[str, bytes]:
    exts = [EXT_ONNX, EXT_DFG, EXT_ENF]
    resolvers = map(partial(resolve_file, name), exts)
    return dict((x, y) for x, y in zip(exts, await asyncio.gather(*resolvers)))
