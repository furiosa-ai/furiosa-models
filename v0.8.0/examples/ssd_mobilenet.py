from furiosa.models.vision import SSDMobileNet
from furiosa.models.vision.ssd_mobilenet import postprocess, preprocess
from furiosa.runtime import session

model = SSDMobileNet.load()

with session.create(model) as sess:
    image, context = preprocess(["tests/assets/cat.jpg"])
    output = sess.run(image).numpy()
    postprocess(output, context=context)
