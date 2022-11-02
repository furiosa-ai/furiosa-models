"""
ResNet50 v1.5 backbone model trained on ImageNet (224x224).
This model has been used since MLCommons v0.5.

## Usage

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

### Using Furiosa SDK
```python
--8<-- "docs/examples/resnet50.py"
```

!!! example "Example with the Python handler"
    === "aaaa"
        ```md
        # aaa
        ```

> NOTE: **Resources on YAML.**
> YAML can sometimes be a bit tricky, particularly on indentation.
> Here are some resources that other users found useful to better
> understand YAML's peculiarities.
>
> - [YAML idiosyncrasies](https://docs.saltproject.io/en/3000/topics/troubleshooting/yaml_idiosyncrasies.html)
> - [YAML multiline](https://yaml-multiline.info/)

=== "Markdown"
    ```md
    With a custom title:
    [`Object 1`][full.path.object1]

    With the identifier as title:
    [full.path.object2][]
    ```

=== "HTML Result"
    ```html
    <p>With a custom title:
    <a href="https://example.com/page1#full.path.object1"><code>Object 1</code></a><p>
    <p>With the identifier as title:
    <a href="https://example.com/page2#full.path.object2">full.path.object2</a></p>
    ```

## Model inputs
The input is a 3-channel image of 224x224 (height, width).

* Data Type: `numpy.float32`
* Tensor Shape: `[1, 3, 224, 224]`
* Memory Layout: NCHW
* Optimal Batch Size: <= 8

## Outputs
The output is a `numpy.float32` tensor with the shape (`[1,]`), including
a class id. `postprocess()` can transform the class id to a single label.

## Model Source
This model is originated from ResNet50 v1.5 in ONNX available at
[MLCommons - Supported Models](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#supported-models).

# API Reference
"""

"""int: Module level variable documented inline.

The docstring may span multiple lines. The type may optionally be specified
on the first line, separated by a colon.
"""

from typing import Any, Dict, List, Sequence, Union

import cv2
import numpy
import numpy as np

from furiosa.registry import Format, Metadata, Publication

from .. import native
from ...errors import ArtifactNotFound
from ...model import ClassificationModel
from ...utils import EXT_DFG, EXT_ENF, EXT_ONNX
from ..common.datasets import imagenet1k
from ..postprocess import PostProcessor
from ..preprocess import center_crop, resize_with_aspect_ratio

CLASSES: List[str] = imagenet1k.ImageNet1k_CLASSES


class ResNet50(ClassificationModel):
    """MLCommons ResNet50 model2

    ```python
    --8<-- "docs/examples/resnet50.py"
    ```

    """

    @classmethod
    def get_artifact_name(cls):
        return "mlcommons_resnet50_v1.5_int8"

    @classmethod
    def load_aux(cls, artifacts: Dict[str, bytes], *args, **kwargs):
        return cls(
            name="ResNet50",
            source=artifacts[EXT_ONNX],
            dfg=artifacts[EXT_DFG],
            enf=artifacts[EXT_ENF],
            format=Format.ONNX,
            family="ResNet",
            version="v1.5",
            metadata=Metadata(
                description="ResNet50 v1.5 int8 ImageNet-1K",
                publication=Publication(url="https://arxiv.org/abs/1512.03385.pdf"),
            ),
            *args,
            **kwargs,
        )


def preprocess(image: Union[str, np.ndarray]) -> np.array:
    """Read and preprocess an image located at image_path.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Examples:
        a = b
        c = d
    """
    # https://github.com/mlcommons/inference/blob/af7f5a0b856402b9f461002cfcad116736a8f8af/vision/classification_and_detection/python/main.py#L37-L39
    # https://github.com/mlcommons/inference/blob/af7f5a0b856402b9f461002cfcad116736a8f8af/vision/classification_and_detection/python/dataset.py#L168-L184
    if type(image) == str:
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_with_aspect_ratio(image, 224, 224, percent=87.5, interpolation=cv2.INTER_AREA)
    image = center_crop(image, 224, 224)
    image = np.asarray(image, dtype=np.float32)
    # https://github.com/mlcommons/inference/blob/af7f5a0b856402b9f461002cfcad116736a8f8af/vision/classification_and_detection/python/dataset.py#L178
    image -= np.array([123.68, 116.78, 103.94], dtype=np.float32)
    image = image.transpose([2, 0, 1])
    return image[np.newaxis, ...]


def postprocess(outputs: Sequence[numpy.ndarray]) -> str:
    """Read and preprocess an image located at image_path."""
    return CLASSES[int(outputs[0]) - 1]


class Resnet50PostProcessor(PostProcessor):
    """Read and preprocess an image located at image_path."""
    def eval(self, inputs: Sequence[numpy.ndarray], *args: Any, **kwargs: Any) -> str:
        return CLASSES[self._native.eval(inputs) - 1]


class NativePostProcessor(Resnet50PostProcessor):
    """Read and preprocess an image located at image_path.

    Examples:
        Some explanation of what is possible.

        >>> print("hello!")
        hello!

        Blank lines delimit prose vs. console blocks.

        >>> a = 0
        >>> a += 1
        >>> a
        1

    """
    def __init__(self, model: ResNet50):
        if not model.dfg:
            raise ArtifactNotFound(model.name, "dfg")

        self._native = native.resnet50.PostProcessor(model.dfg)

        super().__init__()
