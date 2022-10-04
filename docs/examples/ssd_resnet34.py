from furiosa.models.vision import SSDResNet34
from furiosa.models.vision.ssd_resnet34 import postprocess, preprocess
from furiosa.runtime import session

ssd_resnet34 = SSDResNet34.load()

with session.create(ssd_resnet34.enf) as sess:
    image, context = preprocess(["tests/assets/cat.jpg"])
    output = sess.run(image).numpy()
    postprocess(output, batch_preproc_params=context)
