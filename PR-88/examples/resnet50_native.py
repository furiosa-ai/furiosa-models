from furiosa.models.vision import ResNet50
from furiosa.models.vision.resnet50 import NativePostProcessor, preprocess
from furiosa.runtime import session

model = ResNet50.load(use_native_post=True)

postprocessor = NativePostProcessor(model)
with session.create(model) as sess:
    image = preprocess("tests/assets/cat.jpg")
    output = sess.run(image).numpy()
    postprocessor.eval(output)
