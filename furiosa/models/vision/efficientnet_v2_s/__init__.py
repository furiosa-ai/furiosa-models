from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from furiosa.registry.model import Format, Metadata, Publication

from ...types import ImageClassificationModel, PostProcessor, PreProcessor
from ...utils import EXT_DFG, EXT_ENF, EXT_ONNX
from ..common.datasets import imagenet1k

CLASSES: List[str] = imagenet1k.ImageNet1k_CLASSES


class EfficientNetV2sPreProcessor(PreProcessor):
    @staticmethod
    def __call__(image: Path) -> Tuple[np.ndarray, None]:
        """Read and preprocess an image located at image_path."""
        pass


class EfficientNetV2sPostProcessor(PostProcessor):
    def __call__(self, model_outputs: Sequence[npt.ArrayLike], contexts: Any = None) -> str:
        pass


class EfficientNetV2s(ImageClassificationModel):
    """EfficientNetV2-s model"""

    @staticmethod
    def get_artifact_name():
        return "efficientnet_v2_s"

    @classmethod
    def load_aux(cls, artifacts: Dict[str, bytes], use_native: bool = False, *args, **kwargs):
        return cls(
            name="EfficientNetV2s",
            source=artifacts[EXT_ONNX],
            dfg=artifacts[EXT_DFG],
            enf=artifacts[EXT_ENF],
            format=Format.ONNX,
            family="EfficientNetV2",
            version="s",
            metadata=Metadata(
                description="EfficientNetV2s ImageNet-1K",
                publication=Publication(url="https://arxiv.org/abs/2104.00298"),
            ),
            preprocessor=EfficientNetV2sPreProcessor(),
            postprocessor=EfficientNetV2sPostProcessor(),
        )
