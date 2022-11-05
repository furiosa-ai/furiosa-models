from furiosa.models.vision import ResNet50
from furiosa.runtime import session

image = "tests/assets/cat.jpg"

resnet50 = ResNet50.load()
with session.create(resnet50) as sess:
    inputs, _ = resnet50.preprocess(image)
    outputs = sess.run(inputs).numpy()
    outputs = resnet50.postprocess(outputs)
    print(outputs)
