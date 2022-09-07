from typing import List

import numpy as np
import pytest

from furiosa.models.vision import SSDMobileNet
from furiosa.models.vision.postprocess import ObjectDetectionResult
from furiosa.models.vision.ssd_mobilenet import (
    CLASSES,
    CppPostProcessor,
    RustPostProcessor,
    postprocess,
    preprocess,
)
from furiosa.registry import Model
from furiosa.runtime import session

test_image_path = "tests/assets/cat.jpg"
expected_bbox = np.array([[187.30786, 88.035324, 950.99646, 743.3290239999999]], dtype=np.float32)
expected_score = np.array([0.97390455], dtype=np.float32)


def assert_results(results: List[ObjectDetectionResult]):
    assert len(CLASSES) == 92, f"Classes is 92, but {len(CLASSES)}"
    assert len(results) == 1, "detected object must be 1"
    assert np.array_equal(
        results[0].label, 'cat'
    ), f"wrong classid: {results[0].label}, expected cat"
    assert (
        np.sum(np.abs(np.array(list(results[0].boundingbox)) - expected_bbox)) < 9
    ), f"bbox is different from expected value"
    assert (
        np.sum(np.abs(results[0].score - expected_score)) < 1e-3
    ), "confidence is different from expected value"


@pytest.mark.asyncio
async def test_mlcommons_mobilessd_small_perf():
    model: Model = SSDMobileNet()

    # For testing batch modes, simply read two identical images.
    images, contexts = preprocess([test_image_path, test_image_path])

    with session.create(model.source, batch_size=2) as sess:
        outputs = sess.run(images).numpy()
        results = postprocess(outputs, contexts, confidence_threshold=0.3)
        assert len(results) == 2, "the number of outputs must be 2"
        assert_results(results[0])


def test_rust_post_processor():
    model = SSDMobileNet(optimized_postprocess=True)
    processor = RustPostProcessor(model)

    images, context = preprocess([test_image_path])

    with session.create(model.enf) as sess:
        outputs = sess.run(images).numpy()
        results = processor.eval(outputs, context=context)
        assert len(results) == 1, "the number of outputs must be 1"
        assert_results(results)


def test_cpp_post_processor():
    model = SSDMobileNet(optimized_postprocess=True)
    processor = CppPostProcessor(model)

    images, context = preprocess([test_image_path])

    with session.create(model.enf) as sess:
        outputs = sess.run(images).numpy()
        results = processor.eval(outputs, context=context)
        assert len(results) == 1, "the number of outputs must be 1"
        assert_results(results)
