from furiosa.models.vision import ResNet50
from furiosa.registry import Model

model: Model = ResNet50.load()

print(model.name)
print(model.format)
print(model.metadata.description)
