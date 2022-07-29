import logging
from typing import Callable

import aiohttp
import dvc.api

module_logger = logging.getLogger(__name__)

async def load_dvc(uri: str, dvc_repo: str = None, dvc_rev: str = None):
    module_logger.debug(f"dvc_uri={uri}, DVC_REPO={dvc_repo}, DVC_REV={dvc_rev}")
    async with aiohttp.ClientSession() as session:
        async with session.get(
            dvc.api.get_url(
                uri,
                repo=dvc_repo,
                rev=dvc_rev,
            )
        ) as resp:
            return await resp.read()
