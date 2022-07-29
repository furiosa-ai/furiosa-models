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

class LazyPipeLine:
    def __init__(self, value: object):
        if isinstance(value, Callable):
            self.compute = value
        else:

            def return_val():
                return value

            self.compute = return_val

    def bind(self, f: Callable, *args, kwargs={}) -> "LazyPipeLine":
        def f_compute():
            computed_result = self.compute()
            if type(computed_result) == tuple:
                return f(*computed_result, *args, **kwargs)
            return f(computed_result, *args, **kwargs)

        return LazyPipeLine(f_compute)
