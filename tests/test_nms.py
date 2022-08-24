# FIXME: This is a temporary test script to check rust binding has imported
import numpy as np

from furiosa.models import nms_internal_ops_fast_rust


def test_nms():
    print(
        nms_internal_ops_fast_rust(
            np.array([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=np.float32),
            np.array([0.3, 0.2], dtype=np.float32),
            0.45,
            1e-5,
        )
    )


test_nms()
