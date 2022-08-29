from dataclasses import dataclass
import logging
import os
from typing import Optional

import aiohttp
import dvc.api
from furiosa.common.native import find_native_libs, find_native_lib_path, DEFAULT_ENCODING

module_logger = logging.getLogger(__name__)


async def load_dvc(uri: str):
    dvc_repo = os.environ.get("DVC_REPO", None)
    dvc_rev = os.environ.get("DVC_REV", None)
    module_logger.debug(f"dvc_uri={uri}, DVC_REPO={dvc_repo}, DVC_REV={dvc_rev}")
    async with aiohttp.ClientSession() as session:
        async with session.get(
            dvc.api.get_url(
                uri,
                repo=os.environ.get("DVC_REPO", None),
                rev=os.environ.get("DVC_REV", None),
            )
        ) as resp:
            return await resp.read()


async def load_dvc_generated(uri: str, extension: str):
    """Return the generated artifacts identified by the original source model"""
    artifact_path = generated_artifact_path(uri, extension)
    if not artifact_path:
        return None

    return await load_dvc(artifact_path)


def generated_artifact_path(src_path: str, extension: str) -> str:
    rel_dir, full_filename = src_path.split("/")
    ext_index = full_filename.rfind(".")
    filename = full_filename[:ext_index]
    # FIXME: warboy_2pe should be parameterized
    return f"{rel_dir}/{_generated_path_base()}/{filename}_warboy_2pe.{extension}"


def _generated_path_base() -> Optional[str]:
    version_info = compiler_version()
    if version_info:
        return f"generated/{version_info.version}_{version_info.revision}"
    else:
        return None


@dataclass
class CompilerVersion:
    version: str
    revision: str


def compiler_version() -> Optional[CompilerVersion]:
    if find_native_lib_path("nux") is None:
        return None
    else:
        libnux = find_native_libs("nux")
        return CompilerVersion(libnux.version().decode(DEFAULT_ENCODING),
                               libnux.git_short_hash().decode(DEFAULT_ENCODING))
