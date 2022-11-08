import asyncio

from furiosa.models.types import Model
from furiosa.models.vision import ResNet50

model: Model = asyncio.run(ResNet50.load_async())
