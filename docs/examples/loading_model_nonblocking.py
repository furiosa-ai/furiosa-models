import asyncio

from furiosa.models.vision.nonblocking import ResNet18
from furiosa.registry import Model

model: Model = asyncio.run(ResNet18())

print(model.name)
print(model.format)
print(model.metadata.description)
