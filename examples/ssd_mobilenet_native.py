from furiosa.models.vision import SSDMobileNet
from furiosa.models.vision.ssd_mobilenet import NativePostProcessor, preprocess
from furiosa.runtime import session

model = SSDMobileNet.load(use_native_post=True)

postprocessor = NativePostProcessor(model)
with session.create(model.enf) as sess:
    image, context = preprocess(["tests/assets/cat.jpg"])
    output = sess.run(image).numpy()
    postprocessor.eval(output, context=context[0])
