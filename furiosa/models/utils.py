import logging
import os

import aiohttp
import dvc.api

from artifacts import module_logger

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