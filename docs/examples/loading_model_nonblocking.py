import asyncio

from furiosa.models.vision import ResNet50
from furiosa.registry import Model

model: Model = asyncio.run(ResNet50.load_async())

print(model.name)
print(model.format)
print(model.metadata.description)
