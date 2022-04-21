import io
from typing import Callable, Optional, Tuple, TypeVar, Union

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from timm.data.transforms_factory import transforms_imagenet_eval

T = TypeVar("T")

__OPSET_VERSION__ = 13


class ModelBase:
    def __init__(self, model: T, input_shape: Tuple[int, int, int]):
        self.model = model
        self.input_shape = input_shape

    def export(
        self,
        onnx_path: Union[str, io.BytesIO],
        opset_version=__OPSET_VERSION__,
        batch_size: Optional[int] = 1,
        half_precision=False,
    ) -> None:
        dummy_input = torch.randn(batch_size, *self.input_shape, requires_grad=True)
        if half_precision:
            if not torch.cuda.is_available():
                raise Exception("Gpu(s) must be available.")
            self.model = self.model.eval().cuda().half()
            dummy_input = dummy_input.cuda().half()

        self.model.eval()

        # TODO How to keep the opset version up to date?
        torch.onnx.export(
            self.model,
            (dummy_input,),
            f=onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
        )

        # post optimization: convert batch norm param(s) with all zeros to ones
        self.post_optimization(onnx_path)

        return onnx_path

    @property
    def pytorch_model(self):
        return self.model

    @property
    def onnx_model(self) -> Union[onnx.ModelProto, bytes]:
        f = io.BytesIO()
        serialized_proto = io.BytesIO(self.export(f).getvalue())
        model = onnx.load_model(serialized_proto, onnx.ModelProto)
        # TODO Do we have a better solutioin for dynamic batching?
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch_size"
        return model

    @staticmethod
    def post_optimization(path):
        model = onnx.load_model(path)
        initializer = {init.name: init for init in model.graph.initializer}
        for node in model.graph.node:
            if node.op_type != "BatchNormalization":
                continue

            for idx, node_input in enumerate(node.input):
                if node_input not in initializer.keys():
                    continue

                init = initializer[node_input]
                arr = numpy_helper.to_array(init)
                if not all([v == 0.0 for v in arr]):
                    continue
                new_arr = np.ones_like(arr)

                model.graph.initializer.remove(init)
                model.graph.initializer.append(
                    onnx.helper.make_tensor(
                        name=init.name,
                        data_type=init.data_type,
                        dims=init.dims,
                        vals=new_arr.flatten(),
                    )
                )
        onnx.save_model(model, path)


class ImageNetRwightman(ModelBase):
    source = "rwightman"
    training_data = "imagenet"

    @staticmethod
    def configer(module, model_name, config_key=None):
        if not config_key:
            config_key = model_name

        default_configs = getattr(module, "default_cfgs")
        model_config = default_configs[config_key]
        model_func = staticmethod(getattr(module, config_key))
        input_shape = model_config["input_size"]
        has_pretrained = True if model_config["url"] else False

        crop_pct = model_config["crop_pct"]
        interpolation = model_config["interpolation"]
        mean = model_config["mean"]
        std = model_config["std"]
        img_size = model_config["input_size"][1:]

        transform = transforms_imagenet_eval(
            img_size, crop_pct, interpolation, False, mean, std
        )
        model_config["transform"] = transform
        return model_config, model_func, input_shape, has_pretrained

    def __init__(
        self,
        model_func: Callable,
        input_shape: Tuple[int, int, int],
        pretrained: Optional[bool] = False,
        **kwargs
    ):
        self.pretrained = pretrained
        super(ImageNetRwightman, self).__init__(
            model=model_func(pretrained=pretrained, **kwargs), input_shape=input_shape
        )
