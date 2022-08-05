from furiosa.models.vision import ResNet18
from furiosa.registry import Model

model: Model = ResNet18()

print(model.name)
print(model.format)
print(model.metadata.description)
