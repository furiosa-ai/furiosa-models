import random
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np

from furiosa.registry import Model
from furiosa.runtime import session


class LazyPipeLine:
    def __init__(self, value: object):
        if isinstance(value, Callable):
            self.compute = value
        else:

            def return_val():
                return value

            self.compute = return_val

    def bind(self, f: Callable, *args, kwargs={}) -> "LazyPipeLine":
        def f_compute():
            computed_result = self.compute()
            if type(computed_result) == tuple:
                return f(*computed_result, *args, **kwargs)
            return f(computed_result, *args, **kwargs)

        return LazyPipeLine(f_compute)


class InferenceTestSessionWrapper(object):
    sess: Optional[Any] = None
    model: Optional[Model] = None

    def __init__(self, model: Model):
        self.model = model

    def open_session(self):
        self.sess = session.create(self.model.model)

    def close_session(self):
        if not (self.sess is None):
            self.sess.close()

    def pre_session(self, *args: Any):
        return self.model.preprocess(*args)

    def run_session(self, *args: Any):
        return self.sess.run(args[0]), args[1:]

    def post_session(self, sess_output, extra_param: List[Any], **kwargs) -> Any:
        return self.model.postprocess(sess_output, *extra_param, **kwargs)

    def inference(self, *args, pre_config={}, post_config={}) -> Any:
        return (
            LazyPipeLine(*args)
            .bind(self.pre_session, kwargs=pre_config)
            .bind(self.run_session)
            .bind(self.post_session, kwargs=post_config)
            .compute()
        )

    def __enter__(self):
        self.open_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_session()

    def __del__(self):
        self.close_session()


COLORS_MAP = [
    (178, 34, 34),
    (221, 160, 221),
    (0, 255, 0),
    (0, 128, 0),
    (210, 105, 30),
    (220, 20, 60),
    (144, 238, 144),
    (192, 192, 192),
    (255, 228, 196),
    (50, 205, 50),
    (139, 0, 139),
    (100, 149, 237),
    (138, 43, 226),
    (238, 130, 238),
    (255, 0, 255),
    (0, 100, 0),
    (127, 255, 0),
    (255, 0, 255),
    (0, 0, 205),
    (255, 140, 0),
    (255, 239, 213),
    (199, 21, 133),
    (124, 252, 0),
    (147, 112, 219),
    (106, 90, 205),
    (176, 196, 222),
    (65, 105, 225),
    (173, 255, 47),
    (255, 20, 147),
    (219, 112, 147),
    (186, 85, 211),
    (199, 21, 133),
    (148, 0, 211),
    (255, 99, 71),
    (144, 238, 144),
    (255, 255, 0),
    (230, 230, 250),
    (0, 0, 255),
    (128, 128, 0),
    (189, 183, 107),
    (255, 255, 224),
    (128, 128, 128),
    (105, 105, 105),
    (64, 224, 208),
    (205, 133, 63),
    (0, 128, 128),
    (72, 209, 204),
    (139, 69, 19),
    (255, 245, 238),
    (250, 240, 230),
    (152, 251, 152),
    (0, 255, 255),
    (135, 206, 235),
    (0, 191, 255),
    (176, 224, 230),
    (0, 250, 154),
    (245, 255, 250),
    (240, 230, 140),
    (245, 222, 179),
    (0, 139, 139),
    (143, 188, 143),
    (255, 0, 0),
    (240, 128, 128),
    (102, 205, 170),
    (60, 179, 113),
    (46, 139, 87),
    (165, 42, 42),
    (178, 34, 34),
    (175, 238, 238),
    (255, 248, 220),
    (218, 165, 32),
    (255, 250, 240),
    (253, 245, 230),
    (244, 164, 96),
    (210, 105, 30),
]


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img

    tl = line_thickness or round(0.0004 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return img


def draw_bboxes(
    ori_img,
    bbox: List[Tuple[Tuple[float], float, str, float]],
    identities=None,
    offset=(0, 0),
    cvt_color=False,
):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        (x1, y1, x2, y2), score, identities, class_id = box
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # box text and bar
        color = COLORS_MAP[class_id % len(COLORS_MAP)]
        label = f"{identities} ({score:.2f})"
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1, y1, x2, y2], img, color, label, line_thickness=None)
    return img
