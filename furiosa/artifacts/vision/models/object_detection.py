from typing import Any

from furiosa.registry import Model

from .mlcommons.common.datasets import coco, dataset


class MLCommonsSSDSmallModel(Model):
    """MLCommons MobileNet v1 model"""

    def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        return dataset.pre_process_coco_pt_mobilenet(*args, **kwargs)

    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        return coco.PostProcessCocoSSDMobileNetORTlegacy(False, 0.3)(*args, **kwargs)


class MLCommonsSSDLargeModel(Model):
    """MLCommons ResNet34 model"""

    def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        return dataset.pre_process_coco_resnet34(*args, **kwargs)

    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        return coco.PostProcessCocoONNXNPlegacy()(*args, **kwargs)
