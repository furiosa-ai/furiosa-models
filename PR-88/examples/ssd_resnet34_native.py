from furiosa.models.vision import SSDResNet34
from furiosa.models.vision.ssd_resnet34 import NativePostProcessor, preprocess
from furiosa.runtime import session

model = SSDResNet34.load(use_native_post=True)
postprocessor = NativePostProcessor(model)

with session.create(model) as sess:
    image, context = preprocess(["tests/assets/cat.jpg"])
    output = sess.run(image).numpy()
    postprocessor.eval(output, context=context)
