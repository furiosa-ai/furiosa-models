from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Optional, Union

import aiohttp
import dvc.api

from furiosa.common.native import DEFAULT_ENCODING, find_native_lib_path, find_native_libs

from . import errors

GENERATED_EXTENSIONS = ('dfg', 'enf')
DATA_DIRECTORY_BASE = Path(__file__).parent / "data"
CACHE_DIRECTORY_BASE = Path(
    os.getenv(
        "FURIOSA_MODELS_CACHE_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "furiosa/models"),
    )
)
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


def is_onnx_file(uri: Union[str, Path]) -> bool:
    return str(uri).lower().endswith(".onnx")


class ResolvedFile(ABC):
    @abstractmethod
    async def read(self):
        ...


class LocalFile(ResolvedFile):
    def __init__(self, uri: Path):
        self.uri = uri

    async def read(self):
        # Maybe we can use https://github.com/Tinche/aiofiles later
        return self.uri.read_bytes()


# TODO: Implement Cache
class LocalCachedFile(ResolvedFile):
    pass


class DVCFile(ResolvedFile):
    def __init__(self, uri: Union[str, Path]):
        self.uri = removesuffix(str(uri), ".dvc")

    async def read(self):
        dvc_repo = os.environ.get("DVC_REPO", None)
        dvc_rev = os.environ.get("DVC_REV", None)
        module_logger.debug(f"DVC_URI={self.uri}, DVC_REPO={dvc_repo}, DVC_REV={dvc_rev}")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    dvc.api.get_url(
                        self.uri,
                        repo=os.environ.get("DVC_REPO", None),
                        rev=os.environ.get("DVC_REV", None),
                    )
                ) as resp:
                    return await resp.read()
            except Exception as e:
                if is_onnx_file(self.uri):
                    raise e
                else:
                    # It can happen in development phase
                    module_logger.warning(f"{self.uri} is missing")
                    module_logger.warning(e)
                    return None


def resolve_file(src_name: str, extension: str, generated_suffix="_warboy_2pe") -> ResolvedFile:
    # First check whether it is generated file or not
    if extension.lower() in GENERATED_EXTENSIONS:
        basepath = DATA_DIRECTORY_BASE / _generated_path_base()
        file_name = f'{src_name}{generated_suffix}.{extension}'
    else:
        basepath = DATA_DIRECTORY_BASE
        file_name = f'{src_name}.{extension}'
    full_path = basepath / file_name

    # Find real file in data folder
    if full_path.exists():
        module_logger.debug(f"{full_path} exists, making LocalFile class")
        return LocalFile(full_path.resolve())

    # Check it has been cached
    # cached_file = CACHE_DIRECTORY_BASE / file_name
    # if cached_file.exists():
    #     return LocalCachedFile(cached_file)

    # .dvc file
    dvc_path = f'{str(full_path)}.dvc'
    if Path(dvc_path).exists():
        module_logger.debug(f"{dvc_path} exists, making DVCFile class")
        return DVCFile(Path(dvc_path))

    raise errors.ArtifactNotFound(src_name, extension)
