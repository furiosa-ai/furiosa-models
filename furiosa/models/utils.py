import aiohttp
import dvc.api


async def load_dvc(uri: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(dvc.api.get_url(uri)) as resp:
            return await resp.read()
