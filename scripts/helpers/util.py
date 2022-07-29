from typing import Any, Callable, ForwardRef, List, Optional, Tuple

from PIL import Image, ImageOps
import numpy as np

from furiosa.registry import Model
from furiosa.runtime import session


def load_image(image_path: str, seq_channel: str="RGB") -> np.array:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert(seq_channel)
    image = np.asarray(image)
    return image

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
