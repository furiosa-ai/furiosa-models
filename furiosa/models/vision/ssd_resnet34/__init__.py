import itertools
import logging
from math import sqrt
from typing import Any, Dict, List, Sequence, Tuple, Type, Union

import cv2
import numpy
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from furiosa.registry.model import Format, Metadata, Publication

from .. import native
from ...errors import ArtifactNotFound
from ...types import ObjectDetectionModel, Platform, PostProcessor, PreProcessor
from ...utils import EXT_DFG, EXT_ENF, EXT_ONNX, get_field_default
from ..common.datasets import coco
from ..postprocess import LtrbBoundingBox, ObjectDetectionResult, calibration_ltrbbox

NUM_OUTPUTS: int = 12
CLASSES = coco.MobileNetSSD_Large_CLASSES

logger = logging.getLogger(__name__)


##Inspired by https://github.com/kuangliu/pytorch-ssd
class Encoder(object):
    """
    Transform between (bboxes, labels) <-> SSD output

    dboxes: default boxes in size 8732 x 4,
        encoder: input ltrb format, output xywh format
        decoder: input xywh format, output ltrb format

    decode:
        input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
        output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
        criteria : IoU threshold of bboexes
        max_output : maximum number of output bboxes
    """

    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)
        # print("# Bounding boxes: {}".format(self.nboxes))
        self.scale_xy = torch.tensor(dboxes.scale_xy)
        self.scale_wh = torch.tensor(dboxes.scale_wh)

    def decode_batch(self, bboxes_in, scores_in, criteria=0.50, max_output=200):
        self.dboxes = self.dboxes.to(bboxes_in)
        self.dboxes_xywh = self.dboxes_xywh.to(bboxes_in)
        bboxes, probs = scale_back_batch(
            bboxes_in, scores_in, self.scale_xy, self.scale_wh, self.dboxes_xywh
        )
        boxes = []
        labels = []
        scores = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            dbox, dlabel, dscore = self.decode_single(bbox, prob, criteria, max_output)
            boxes.append(dbox)
            labels.append(dlabel)
            scores.append(dscore)

        return [boxes, labels, scores]

    # perform non-maximum suppression
    def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
        # Reference to https://github.com/amdegroot/ssd.pytorch

        bboxes_out = []
        scores_out = []
        labels_out = []

        for i, score in enumerate(scores_in.split(1, 1)):
            # skip background
            if i == 0:
                continue

            score = score.squeeze(1)
            mask = score > 0.05

            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0:
                continue

            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                iou_sorted = calc_iou_tensor(bboxes_sorted, bboxes_idx).squeeze()
                # we only need iou < criteria
                score_idx_sorted = score_idx_sorted[iou_sorted < criteria]
                candidates.append(idx)

            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i] * len(candidates))

        bboxes_out, labels_out, scores_out = (
            torch.cat(bboxes_out, dim=0),
            torch.tensor(labels_out, dtype=torch.long),
            torch.cat(scores_out, dim=0),
        )

        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]


@torch.jit.script
def calc_iou_tensor(box1, box2):
    """Calculation of IoU based on two boxes tensor,
    Reference to https://github.com/kuangliu/pytorch-ssd
    input:
        box1 (N, 4)
        box2 (M, 4)
    output:
        IoU (N, M)
    """
    N = box1.size(0)
    M = box2.size(0)

    be1 = box1.unsqueeze(1).expand(-1, M, -1)
    be2 = box2.unsqueeze(0).expand(N, -1, -1)

    # Left Top & Right Bottom
    lt = torch.max(be1[:, :, :2], be2[:, :, :2])
    rb = torch.min(be1[:, :, 2:], be2[:, :, 2:])
    delta = rb - lt
    delta.clone().masked_fill_(delta < 0, 0)
    intersect = delta[:, :, 0] * delta[:, :, 1]
    delta1 = be1[:, :, 2:] - be1[:, :, :2]
    area1 = delta1[:, :, 0] * delta1[:, :, 1]
    delta2 = be2[:, :, 2:] - be2[:, :, :2]
    area2 = delta2[:, :, 0] * delta2[:, :, 1]

    iou = intersect / (area1 + area2 - intersect)
    return iou


@torch.jit.script
def scale_back_batch(bboxes_in, scores_in, scale_xy, scale_wh, dboxes_xywh):
    """
    Do scale and transform from xywh to ltrb
    suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
    """
    bboxes_in = bboxes_in.permute(0, 2, 1)
    scores_in = scores_in.permute(0, 2, 1)

    bboxes_in[:, :, :2] = scale_xy * bboxes_in[:, :, :2]
    bboxes_in[:, :, 2:] = scale_wh * bboxes_in[:, :, 2:]
    bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * dboxes_xywh[:, :, 2:] + dboxes_xywh[:, :, :2]
    bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * dboxes_xywh[:, :, 2:]
    # Transform format to ltrb
    l, t, r, b = (
        bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2],
        bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3],
        bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2],
        bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3],
    )
    bboxes_in[:, :, 0] = l
    bboxes_in[:, :, 1] = t
    bboxes_in[:, :, 2] = r
    bboxes_in[:, :, 3] = b
    return bboxes_in, F.softmax(scores_in, dim=-1)


def dboxes_R34_coco(figsize, strides):
    feat_size = [[50, 50], [25, 25], [13, 13], [7, 7], [3, 3], [3, 3]]
    steps = [(int(figsize[0] / fs[0]), int(figsize[1] / fs[1])) for fs in feat_size]
    scales = [
        (int(s * figsize[0] / 300), int(s * figsize[1] / 300))
        for s in [21, 45, 99, 153, 207, 261, 315]
    ]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


class DefaultBoxes(object):
    def __init__(
        self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2
    ):

        self.feat_size = feat_size
        self.fig_size_w, self.fig_size_h = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps_w = [st[0] for st in steps]
        self.steps_h = [st[1] for st in steps]
        self.scales = scales
        fkw = self.fig_size_w // np.array(self.steps_w)
        fkh = self.fig_size_h // np.array(self.steps_h)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):
            sfeat_w, sfeat_h = sfeat
            sk1 = scales[idx][0] / self.fig_size_w
            sk2 = scales[idx + 1][1] / self.fig_size_h
            sk3 = sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]
            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat_w), range(sfeat_h)):
                    cx, cy = (j + 0.5) / fkh[idx], (i + 0.5) / fkw[idx]
                    self.default_boxes.append((cx, cy, w, h))
        self.dboxes = torch.tensor(self.default_boxes)
        self.dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb":
            return self.dboxes_ltrb
        if order == "xywh":
            return self.dboxes


def _pick_best(detections, confidence_threshold):
    bboxes, classes, confidences = detections
    best = np.argwhere(confidences > confidence_threshold).squeeze(axis=1)
    return [pred[best].squeeze(axis=0) for pred in detections]


class SSDResNet34PreProcessor(PreProcessor):
    @staticmethod
    def __call__(
        images: Sequence[Union[str, np.ndarray]]
    ) -> Tuple[npt.ArrayLike, List[Dict[str, Any]]]:
        """Preprocess input images to a batch of input tensors

        Args:
            images: A list of paths of image files (e.g., JPEG, PNG)
                or a stacked image loaded as a numpy array in BGR order or gray order.

        Returns:
            The first element is a list of 3-channel images of 1200x1200
                in NCHW format, and the second element is a list of context about the original image metadata.
                Please learn more about the outputs of preprocess (i.e., model inputs),
                please refer to [Inputs](ssd_resnet34.md#inputs).

        """
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/main.py#L141
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/main.py#L61-L63
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/dataset.py#L252-L263
        batch_image = []
        batch_preproc_param = []
        if isinstance(images, str):
            images = [images]
        for image in images:
            if type(image) == str:
                image = cv2.imread(image)
                if image is None:
                    raise FileNotFoundError(image)
            image = np.array(image, dtype=np.float32)
            if len(image.shape) < 3 or image.shape[2] != 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            width = image.shape[1]
            height = image.shape[0]
            image = cv2.resize(image, (1200, 1200), interpolation=cv2.INTER_LINEAR)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image = image / 255.0 - mean
            image = image / std
            # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/main.py#L143
            # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/coco.py#L40
            # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/coco.py#L91
            image = image.transpose([2, 0, 1])
            batch_image.append(image)
            batch_preproc_param.append({"width": width, "height": height})
        return np.stack(batch_image, axis=0), batch_preproc_param


class SSDResNet34PythonPostProcessor(PostProcessor):
    @staticmethod
    def __call__(
        model_outputs: Sequence[np.ndarray],
        contexts: Sequence[Dict[str, Any]],
        confidence_threshold=0.05,
    ) -> List[List[ObjectDetectionResult]]:
        """Convert the outputs of this model to a list of bounding boxes, scores and labels

        Args:
            model_outputs: the outputs of the model. To learn more about the output of model,
                please refer to [Outputs](ssd_resnet34.md#outputs).
            contexts: context coming from `preprocess()`

        Returns:
            Detected Bounding Box and its score and label represented as
                `ObjectDetectionResult`.
                To learn more about `ObjectDetectionResult`, 'Definition of ObjectDetectionResult' can be found below.

        Definition of ObjectDetectionResult and LtrbBoundingBox:
            ::: furiosa.models.vision.postprocess.LtrbBoundingBox
                options:
                    show_source: true
            ::: furiosa.models.vision.postprocess.ObjectDetectionResult
                options:
                    show_source: true
        """
        if len(model_outputs) != NUM_OUTPUTS:
            raise Exception(f"output size must be {NUM_OUTPUTS}, but {len(model_outputs)}")
        classes, locations = model_outputs[:6], model_outputs[6:]

        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_r34.py#L317-L329
        classes = [np.reshape(cls, (cls.shape[0], 81, -1)) for cls in classes]
        locations = [np.reshape(loc, (loc.shape[0], 4, -1)) for loc in locations]
        classes = np.concatenate(classes, axis=2)
        locations = np.concatenate(locations, axis=2)

        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_r34.py#L251-L253
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_r34.py#L350
        # len(det_boxes: List[np.array]) = N batch
        det_boxes, det_labels, det_scores = Encoder(
            dboxes_R34_coco((1200, 1200), (3, 3, 2, 2, 2, 2))
        ).decode_batch(torch.from_numpy(locations), torch.from_numpy(classes), 0.50, 200)

        # Pick the best boxes
        # https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/
        # sometimes there are many boxes with localizaition and class probability distrituion.
        batch_results = []
        for boxes, labels, scores, preproc_params in zip(
            det_boxes, det_labels, det_scores, contexts
        ):
            width, height = preproc_params['width'], preproc_params['height']
            boxes, labels, scores = _pick_best(
                detections=(boxes, labels, scores),
                confidence_threshold=confidence_threshold,
            )
            cal_boxes = calibration_ltrbbox(boxes, width, height)
            predicted_result = []
            for b, l, s in zip(cal_boxes, labels, scores):
                bb_list = b.tolist()
                predicted_result.append(
                    ObjectDetectionResult(
                        index=int(l),
                        label=CLASSES[l],
                        score=s,
                        boundingbox=LtrbBoundingBox(
                            left=bb_list[0], top=bb_list[1], right=bb_list[2], bottom=bb_list[3]
                        ),
                    )
                )
            batch_results.append(predicted_result)
        return batch_results


class SSDResNet34NativePostProcessor(PostProcessor):
    def __init__(self, dfg: bytes):
        self._native = native.ssd_resnet34.RustPostProcessor(dfg)

    def __call__(self, model_outputs: Sequence[numpy.ndarray], contexts: Sequence[Dict[str, Any]]):
        raw_results = self._native.eval(model_outputs)

        results = []
        width = contexts['width']
        height = contexts['height']
        for value in raw_results:
            left = value.left * width
            right = value.right * width
            top = value.top * height
            bottom = value.bottom * height
            results.append(
                ObjectDetectionResult(
                    index=value.class_id,
                    label=CLASSES[value.class_id],
                    score=value.score,
                    boundingbox=LtrbBoundingBox(left=left, top=top, right=right, bottom=bottom),
                )
            )

        return results


class SSDResNet34(ObjectDetectionModel):
    """MLCommons SSD ResNet34 model"""

    postprocessor_map: Dict[Platform, Type[PostProcessor]] = {
        Platform.PYTHON: SSDResNet34PythonPostProcessor,
        Platform.RUST: SSDResNet34NativePostProcessor,
    }

    @staticmethod
    def get_artifact_name():
        return "mlcommons_ssd_resnet34_int8"

    @classmethod
    def load_aux(cls, artifacts: Dict[str, bytes], use_native: bool = True):
        dfg = artifacts[EXT_DFG]
        if use_native and dfg is None:
            raise ArtifactNotFound(cls.get_artifact_name(), EXT_DFG)
        postproc_type = Platform.RUST if use_native else Platform.PYTHON
        logger.debug(f"Using {postproc_type.name} postprocessor")
        postprocessor = get_field_default(cls, "postprocessor_map")[postproc_type](dfg)
        return cls(
            name="SSDResNet34",
            source=artifacts[EXT_ONNX],
            dfg=dfg,
            enf=artifacts[EXT_ENF],
            format=Format.ONNX,
            family="ResNet",
            version="v1.1",
            metadata=Metadata(
                description="SSD ResNet34 model for MLCommons v1.1",
                publication=Publication(
                    url="https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection"
                    # noqa: E501
                ),
            ),
            preprocessor=SSDResNet34PreProcessor(),
            postprocessor=postprocessor,
        )
