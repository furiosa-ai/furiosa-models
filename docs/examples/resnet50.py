from furiosa.models.vision import ResNet50, resnet50
from furiosa.runtime import session

model = ResNet50.load()

with session.create(model.enf) as sess:
    image = resnet50.preprocess("tests/assets/cat.jpg")
    output = sess.run(image).numpy()
    resnet50.postprocess(output)
