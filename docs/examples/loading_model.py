from furiosa.models.types import Model
from furiosa.models.vision import ResNet50

model: Model = ResNet50.load()

print(model.name)
print(model.format)
print(model.metadata.description)
