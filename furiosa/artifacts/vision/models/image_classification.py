from typing import Any, Callable, Optional

import timm
from furiosa.registry import Model

from .common.base import ImageNetRwightman
from .mlcommons.common.datasets import dataset


def call_model_generator(parent_class) -> Callable:
    class ModelGenerator(parent_class):
        configer = parent_class.configer

        def __init__(self, pretrained: Optional[bool] = False):
            self.check_validity(pretrained)
            if not pretrained:
                self.training_data = "none"
            super(ModelGenerator, self).__init__(
                model_func=self.model_func,
                input_shape=self.input_shape,
                pretrained=pretrained,
            )

        def check_validity(self, pretrained):
            if pretrained:
                if not self.has_pretrained:
                    raise Exception(
                        "%s has no trained weights." % self.__class__.__name__
                    )

    return ModelGenerator


TimmModelGenerator = call_model_generator(ImageNetRwightman)


class EfficientNetModelGenerator(TimmModelGenerator):
    module_name = "efficientnet"

    @staticmethod
    def configer(model_name, config_key=None):
        if not config_key:
            config_key = model_name
        return TimmModelGenerator.configer(
            module=getattr(timm.models, EfficientNetModelGenerator.module_name),
            model_name=model_name,
            config_key=config_key,
        )


class EfficientNetV2_S(EfficientNetModelGenerator):
    """EfficientNet V2 model

    https://github.com/google/automl/tree/master/efficientnetv2
    """

    model_family = "EffcientNetV2"
    model_name = "efficientnetv2_s"
    config_key = "tf_efficientnetv2_s"
    (
        model_config,
        model_func,
        input_shape,
        has_pretrained,
    ) = EfficientNetModelGenerator.configer(config_key)


class EfficientNetV2_M(EfficientNetV2_S):
    """EfficientNet V2 model

    https://github.com/google/automl/tree/master/efficientnetv2
    """

    model_name = "efficientnetv2_m"
    config_key = "tf_efficientnetv2_m"
    (
        model_config,
        model_func,
        input_shape,
        has_pretrained,
    ) = EfficientNetModelGenerator.configer(config_key)


class MLCommonsResNet50Model(Model):
    """MLCommons ResNet50 model"""

    def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        return dataset.pre_process_vgg(*args, **kwargs)

    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        return dataset.PostProcessArgMax(offset=0)(*args, **kwargs)
