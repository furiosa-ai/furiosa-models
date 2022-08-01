import numpy as np
import pytest

from furiosa.models.vision import ssd_resnet34 as detector


@pytest.mark.asyncio
async def test_mlcommons_ssd_resnet34_perf():
    test_image_path = "scripts/assets/cat.jpg"

    with await detector.create_session() as sess:
        true_bbox = np.array(
            [
                [264.24792, 259.05603, 699.12964, 474.65332],
                [221.0502, 123.12275, 549.879, 543.1015],
            ],
            dtype=np.float32,
        )
        true_score = np.array([0.37563688, 0.8747512], dtype=np.float32)

        result = detector.inference(sess, detector.load_image(test_image_path), confidence_threshold=0.3)
        print(f"result={result}")
        assert len(result) == 2, "ssd_resnet34 output shape must be 2"
        assert result[0].predicted_class == "cat", f"wrong predicted_class: {result[0].predicted_class}, expected cat"
        assert result[1].predicted_class == "cat", f"wrong predicted_class: {result[1].predicted_class}, expected cat"
        assert np.sum(np.abs(result[0].boundingbox.numpy() - true_bbox[0])) < 1e-3, f"bbox is different from expected value"
        assert np.sum(np.abs(result[1].boundingbox.numpy() - true_bbox[1])) < 1e-3, f"bbox is different from expected value"
        assert (
            np.abs(result[0].score - true_score[0]) < 1e-3
        ), "score is different from expected value"
        assert (
            np.abs(result[1].score - true_score[1]) < 1e-3
        ), "score is different from expected value"
