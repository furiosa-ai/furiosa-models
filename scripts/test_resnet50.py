import pytest

from furiosa.models.vision import ResNet50
from furiosa.models.vision.resnet50 import postprocess, preprocess
from furiosa.runtime import session


def test_mlcommons_resnet50_perf():
    resnet50 = ResNet50()
    test_image_path = "scripts/assets/cat.jpg"

    with session.create(resnet50.model) as sess:
        image = preprocess(test_image_path)
        output = postprocess(sess.run(image))
        assert output == "tabby, tabby cat", "check your result"
