import asyncio

from furiosa.models.nonblocking.vision import ResNet18
from furiosa.registry import Model

model: Model = asyncio.run(ResNet18(pretrained=True))

print(model.name)
print(model.format)
print(model.metadata.description)
