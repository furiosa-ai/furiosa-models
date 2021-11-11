from furiosa.registry import Model
from furiosa.registry.client.transport import FileTransport

from .models.mlcommons.common.datasets import coco, dataset

loader = FileTransport()


class ResNet50(Model):

    preprocess = dataset.pre_process_vgg
    postprocess = dataset.PostProcessArgMax(offset=0)


class SSDSmall(Model):

    preprocess = dataset.pre_process_coco_pt_mobilenet
    postprocess = coco.PostProcessCocoSSDMobileNetORTlegacy(False, 0.3)


class SSDLarge(Model):

    preprocess = dataset.pre_process_coco_resnet34
    postprocess = coco.PostProcessCocoONNXNPlegacy()


async def mlcommons_resnet50():
    return ResNet50(
        name="mlcommons_resnet50",
        model=await loader.read(
            "furiosa/artifacts/vision/mlcommons_resnet50_v1.5_int8.onnx"
        ),
    )


async def mlcommons_ssd_mobilenet():
    return SSDSmall(
        name="mlcommons_ssd_mobilenet",
        model=await loader.read(
            "furiosa/artifacts/vision/mlcommons_ssd_mobilenet_v1_int8.onnx"
        ),
    )


async def mlcommons_ssd_resnet34():
    return SSDLarge(
        name="mlcommons_ssd_resnet34",
        model=await loader.read(
            "furiosa/artifacts/vision/mlcommons_ssd_resnet34_int8.onnx"
        ),
    )
