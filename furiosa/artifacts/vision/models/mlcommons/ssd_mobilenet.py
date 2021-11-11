from collections import OrderedDict
from typing import List, Tuple

import torch
from torch import nn

from .anchor_generator import create_ssd_anchors
from .utils import BatchNorm2d, BiasAdd, Conv2d_tf, decode_boxes, nms


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        OrderedDict(
            [
                ("0", Conv2d_tf(inp, oup, 3, stride, padding="SAME", bias=False)),
                ("0/BatchNorm", BiasAdd(oup)),
                ("0/ReLU", nn.ReLU6(inplace=True)),
            ]
        )
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "depthwise",
                    Conv2d_tf(
                        inp, inp, 3, stride, padding="SAME", groups=inp, bias=False
                    ),
                ),
                ("depthwise/BatchNorm", BatchNorm2d(inp)),
                ("depthwise/ReLU", nn.ReLU6(inplace=True)),
                ("pointwise", nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
                ("pointwise/BatchNorm", BiasAdd(oup)),
                ("pointwise/ReLU", nn.ReLU6(inplace=True)),
            ]
        )
    )


class MobileNetV1Base(nn.ModuleList):
    def __init__(self, return_layers=[11, 13]):
        super(MobileNetV1Base, self).__init__(
            [
                conv_bn(3, 32, 2),
                conv_dw(32, 64, 1),
                conv_dw(64, 128, 2),
                conv_dw(128, 128, 1),
                conv_dw(128, 256, 2),
                conv_dw(256, 256, 1),
                conv_dw(256, 512, 2),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 1024, 2),
                conv_dw(1024, 1024, 1),
            ]
        )
        self.return_layers = return_layers

    def forward(self, x):
        out = []
        for idx, module in enumerate(self):
            x = module(x)
            if idx in self.return_layers:
                out.append(x)
        return out


class PredictionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super(PredictionHead, self).__init__()
        self.classification = nn.Conv2d(
            in_channels, num_classes * num_anchors, kernel_size=1
        )
        self.regression = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):
        class_logits = self.classification(x)
        box_regression = self.regression(x)

        return class_logits, box_regression


class Block(nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Block, self).__init__(
            nn.Conv2d(in_channels, out_channels=mid_channels, kernel_size=1),
            nn.ReLU6(),
            Conv2d_tf(
                mid_channels, out_channels, kernel_size=3, stride=2, padding="SAME"
            ),
            nn.ReLU6(),
        )


class SSD(nn.Module):
    def __init__(self, backbone, predictors, extras):
        super(SSD, self).__init__()

        self.backbone = backbone
        self.extras = extras
        self.predictors = predictors

        # preprocess
        self.image_size = 300
        self.image_mean = 127.5
        self.image_std = 127.5

        self.coder_weights = torch.tensor((10, 10, 5, 5), dtype=torch.float32)
        self._feature_map_shapes = None

        # postprocess
        self.nms_threshold = 0.6

        # set it to 0.01 for better results but slower runtime
        self.score_threshold = 0.3

    def ssd_model(self, x):
        feature_maps = self.backbone(x)

        out = feature_maps[-1]
        for module in self.extras:
            out = module(out)
            feature_maps.append(out)

        results = []
        for feature, module in zip(feature_maps, self.predictors):
            results.append(module(feature))

        class_logits, box_regression = list(zip(*results))
        class_logits = torch.cat(class_logits, 1)
        box_regression = torch.cat(box_regression, 1)

        scores = torch.sigmoid(class_logits)
        box_regression = box_regression.squeeze(0)

        shapes = [o.shape[-2:] for o in feature_maps]
        if shapes != self._feature_map_shapes:
            # generate anchors for the sizes of the feature map
            priors = create_ssd_anchors()._generate(shapes)
            priors = torch.cat(priors, dim=0)
            self.priors = priors.to(scores)
            self._feature_map_shapes = shapes

        self.coder_weights = self.coder_weights.to(scores)
        if box_regression.dim() == 2:
            box_regression = box_regression[None]
        boxes = decode_boxes(box_regression, self.priors, self.coder_weights)
        # add a batch dimension
        return scores, boxes

    def forward(self, images):
        """
        Arguments:
            images (torch.Tensor[N,C,H,W]):
        """

        scores, boxes = self.ssd_model(images)
        list_boxes = []
        list_labels = []
        list_scores = []
        for b in range(len(scores)):
            bboxes, blabels, bscores = self.filter_results(scores[b], boxes[b])
            list_boxes.append(bboxes)
            list_labels.append(blabels.long())
            list_scores.append(bscores)
        # boxes = self.rescale_boxes(boxes, height, width)
        return [list_boxes, list_labels, list_scores]

    def filter_results(self, scores, boxes):
        # in order to avoid custom C++ extensions
        # we use an NMS implementation written purely
        # on python. This implementation is faster on the
        # CPU, which is why we run this part on the CPU
        cpu_device = torch.device("cpu")
        # boxes = boxes[0]
        # scores = scores[0]
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        selected_box_probs = []
        labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > self.score_threshold
            probs = probs[mask]
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = nms(box_probs, self.nms_threshold)
            selected_box_probs.append(box_probs)
            labels.append(
                torch.full((box_probs.size(0),), class_index, dtype=torch.int64)
            )
        selected_box_probs = torch.cat(selected_box_probs)
        labels = torch.cat(labels)
        return selected_box_probs[:, :4], labels, selected_box_probs[:, 4]

    def rescale_boxes(self, boxes, height, width):
        boxes[:, 0] *= width
        boxes[:, 1] *= height
        boxes[:, 2] *= width
        boxes[:, 3] *= height
        return boxes


def create_mobilenetv1_ssd(num_classes):
    backbone = MobileNetV1Base()

    extras = nn.ModuleList(
        [
            Block(1024, 256, 512),
            Block(512, 128, 256),
            Block(256, 128, 256),
            Block(256, 64, 128),
        ]
    )

    predictors = nn.ModuleList(
        [
            PredictionHead(in_channels, num_classes, num_anchors)
            for in_channels, num_anchors in zip(
                (512, 1024, 512, 256, 256, 128), (3, 6, 6, 6, 6, 6)
            )
        ]
    )

    return SSD(backbone, predictors, extras)


def get_tf_pretrained_mobilenet_ssd(weights_file):
    from models.convert_tf_weights import get_state_dict, read_tf_weights

    model = create_mobilenetv1_ssd(91)
    weights = read_tf_weights(weights_file)
    state_dict = get_state_dict(weights)
    model.load_state_dict(state_dict)
    return model


class MLCommons_SSDMobileNetV1(SSD):
    num_classes = 91
    input_size = [300, 300]
    # feature map shapes for fixed input_size [300, 300]
    # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L155
    feature_map_shapes = [(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)]

    def __init__(self):
        backbone = MobileNetV1Base()
        extras = nn.ModuleList(
            [
                Block(1024, 256, 512),
                Block(512, 128, 256),
                Block(256, 128, 256),
                Block(256, 64, 128),
            ]
        )

        predictors = nn.ModuleList(
            [
                PredictionHead(in_channels, self.num_classes, num_anchors)
                for in_channels, num_anchors in zip(
                    (512, 1024, 512, 256, 256, 128), (3, 6, 6, 6, 6, 6)
                )
            ]
        )
        super(MLCommons_SSDMobileNetV1, self).__init__(backbone, predictors, extras)
        self._feature_map_shapes = self.feature_map_shapes
        self.priors = self.anchor_generation(self.feature_map_shapes)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores, boxes = self.model(images)

        return scores, boxes

    def model(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L137-L153
        feature_maps = self.backbone(x)

        out = feature_maps[-1]
        for module in self.extras:
            out = module(out)
            feature_maps.append(out)

        results = []
        for feature, module in zip(feature_maps, self.predictors):
            results.append(module(feature))

        class_logits, box_regression = list(zip(*results))

        bs = class_logits[0].shape[0]
        class_logits = [
            logit.permute(0, 2, 3, 1).reshape(bs, -1, self.num_classes)
            for logit in class_logits
        ]
        box_regression = [
            regress.permute(0, 2, 3, 1).reshape(bs, -1, 4) for regress in box_regression
        ]

        class_logits = torch.cat(class_logits, 1)
        box_regression = torch.cat(box_regression, 1)

        return class_logits, box_regression

    @staticmethod
    def anchor_generation(feature_map_shapes: List[Tuple[int, int]]) -> torch.Tensor:
        # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L158-L159
        priors = create_ssd_anchors()._generate(feature_map_shapes)
        priors = torch.cat(priors, dim=0)

        return priors

    def preprocess(self):
        raise NotImplementedError

    def postprocess(self, class_logits, box_regression):
        scores = torch.sigmoid(class_logits)
        box_regression = box_regression.squeeze(0)

        # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L164-L166
        if box_regression.dim() == 2:
            box_regression = box_regression[None]
        # decode box_regression into bounding box
        # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L155-L167
        boxes = decode_boxes(box_regression, self.priors, self.coder_weights)
        # add a batch dimension

        # sort and nms
        # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L178-L185
        list_boxes = []
        list_labels = []
        list_scores = []
        for b in range(len(scores)):
            bboxes, blabels, bscores = self.filter_results(scores[b], boxes[b])
            list_boxes.append(bboxes)
            list_labels.append(blabels.long())
            list_scores.append(bscores)
        # boxes = self.rescale_boxes(boxes, height, width)
        return [list_boxes, list_labels, list_scores]


class MLCommons_SSDMobileNetV1_legacy(MLCommons_SSDMobileNetV1):
    def model(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L137-L153
        feature_maps = self.backbone(x)

        out = feature_maps[-1]
        for module in self.extras:
            out = module(out)
            feature_maps.append(out)

        results = []
        for feature, module in zip(feature_maps, self.predictors):
            results.append(module(feature))

        class_logits, box_regression = list(zip(*results))

        return class_logits, box_regression

    def postprocess(self, class_logits, box_regression):
        bs = class_logits[0].shape[0]
        class_logits = [
            logit.permute(0, 2, 3, 1).reshape(bs, -1, self.num_classes)
            for logit in class_logits
        ]
        box_regression = [
            regress.permute(0, 2, 3, 1).reshape(bs, -1, 4) for regress in box_regression
        ]

        class_logits = torch.cat(class_logits, 1)
        box_regression = torch.cat(box_regression, 1)
        scores = torch.sigmoid(class_logits)
        box_regression = box_regression.squeeze(0)

        # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L164-L166
        if box_regression.dim() == 2:
            box_regression = box_regression[None]
        # decode box_regression into bounding box
        # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L155-L167
        boxes = decode_boxes(box_regression, self.priors, self.coder_weights)
        # add a batch dimension

        # sort and nms
        # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L178-L185
        list_boxes = []
        list_labels = []
        list_scores = []
        for b in range(len(scores)):
            bboxes, blabels, bscores = self.filter_results(scores[b], boxes[b])
            list_boxes.append(bboxes)
            list_labels.append(blabels.long())
            list_scores.append(bscores)
        # boxes = self.rescale_boxes(boxes, height, width)
        return [list_boxes, list_labels, list_scores]
