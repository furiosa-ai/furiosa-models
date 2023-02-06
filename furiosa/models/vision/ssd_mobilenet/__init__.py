import logging
from typing import Any, Dict, List, Sequence, Tuple, Type, Union

import cv2
import numpy
import numpy as np
import numpy.typing as npt

from furiosa.registry.model import Format, Metadata, Publication

from . import anchor_generator  # type: ignore[import]
from .. import native
from ...errors import ArtifactNotFound
from ...types import ObjectDetectionModel, Platform, PostProcessor, PreProcessor
from ...utils import EXT_DFG, EXT_ENF, EXT_ONNX, get_field_default
from ..common.datasets import coco
from ..postprocess import LtrbBoundingBox, ObjectDetectionResult, calibration_ltrbbox, sigmoid

NUM_OUTPUTS: int = 12
CLASSES = coco.MobileNetSSD_CLASSES
NUM_CLASSES = len(CLASSES) - 1  # remove background

logger = logging.getLogger(__name__)

# https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L155-L158
PRIORS = np.concatenate(
    [
        tensor.numpy()
        # pylint: disable=protected-access
        for tensor in anchor_generator.create_ssd_anchors()._generate(
            [(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)]
        )
    ],
    axis=0,
)
PRIORS_Y1 = np.expand_dims(np.expand_dims(PRIORS[:, 0], axis=1), axis=0)
PRIORS_X1 = np.expand_dims(np.expand_dims(PRIORS[:, 1], axis=1), axis=0)
PRIORS_Y2 = np.expand_dims(np.expand_dims(PRIORS[:, 2], axis=1), axis=0)
PRIORS_X2 = np.expand_dims(np.expand_dims(PRIORS[:, 3], axis=1), axis=0)
PRIORS_WIDTHS = PRIORS_X2 - PRIORS_X1
PRIORS_HEIGHTS = PRIORS_Y2 - PRIORS_Y1
PRIORS_CENTER_X = PRIORS_X1 + 0.5 * PRIORS_WIDTHS
PRIORS_CENTER_Y = PRIORS_Y1 + 0.5 * PRIORS_HEIGHTS

del PRIORS_Y1, PRIORS_X1, PRIORS_Y2, PRIORS_X2, PRIORS


def _decode_boxes(rel_codes: np.ndarray) -> np.ndarray:
    # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/utils.py#L149-L198

    # pylint: disable=invalid-name

    dy = np.expand_dims(rel_codes[:, :, 0], axis=2)
    dx = np.expand_dims(rel_codes[:, :, 1], axis=2)
    dh = np.expand_dims(rel_codes[:, :, 2], axis=2)
    dw = np.expand_dims(rel_codes[:, :, 3], axis=2)

    # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L127
    # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L166
    # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/utils.py#L177-L180
    dx /= 10.0
    dy /= 10.0
    dw /= 5.0
    dh /= 5.0

    prediction_center_x = dx * SSDSmallConstant.PRIORS_WIDTHS + SSDSmallConstant.PRIORS_CENTER_X
    prediction_center_y = dy * SSDSmallConstant.PRIORS_HEIGHTS + SSDSmallConstant.PRIORS_CENTER_Y
    prediction_w = np.exp(dw) * SSDSmallConstant.PRIORS_WIDTHS
    prediction_h = np.exp(dh) * SSDSmallConstant.PRIORS_HEIGHTS

    prediction_boxes = np.concatenate(
        (
            prediction_center_x - 0.5 * prediction_w,
            prediction_center_y - 0.5 * prediction_h,
            prediction_center_x + 0.5 * prediction_w,
            prediction_center_y + 0.5 * prediction_h,
        ),
        axis=2,
    )
    return prediction_boxes


def _filter_results(
    scores: np.ndarray,
    boxes: np.ndarray,
    confidence_threshold: float,
    iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L197-L212
    selected_box_probs = []
    labels = []
    for class_index in range(1, scores.shape[1]):
        probs = scores[:, class_index]
        mask = probs > confidence_threshold
        probs = probs[mask]
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate((subset_boxes, probs.reshape(-1, 1)), axis=1)
        box_probs = _nms(box_probs, iou_threshold)
        selected_box_probs.append(box_probs)
        labels.append(np.full((box_probs.shape[0],), class_index, dtype=np.int64))
    selected_box_probs = np.concatenate(selected_box_probs)  # type: ignore[assignment]
    labels = np.concatenate(labels)  # type: ignore[assignment]
    return selected_box_probs[:, :4], labels, selected_box_probs[:, 4]  # type: ignore[call-overload, return-value]


def _nms(box_scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/utils.py#L122-L146
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)[::-1]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = _box_iou(rest_boxes, np.expand_dims(current_box, axis=0))
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]


def _box_iou(boxes1: np.ndarray, boxes2: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Return intersection-over-union (Jaccard index) of boxes."""
    # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/utils.py#L103-L119
    overlap_left_top = np.maximum(boxes1[..., :2], boxes2[..., :2])
    overlap_right_bottom = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    overlap_area = _box_area(overlap_left_top, overlap_right_bottom)
    area1 = _box_area(boxes1[..., :2], boxes1[..., 2:])
    area2 = _box_area(boxes2[..., :2], boxes2[..., 2:])
    return overlap_area / (area1 + area2 - overlap_area + eps)


def _box_area(left_top: np.ndarray, right_bottom: np.ndarray):
    """Compute the areas of rectangles given two corners."""
    # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/utils.py#L89-L100
    width_height = np.clip(right_bottom - left_top, a_min=0.0, a_max=None)
    return width_height[..., 0] * width_height[..., 1]


class SSDSmallConstant(object):
    PRIORS_WIDTHS = PRIORS_WIDTHS
    PRIORS_HEIGHTS = PRIORS_HEIGHTS
    PRIORS_CENTER_X = PRIORS_CENTER_X
    PRIORS_CENTER_Y = PRIORS_CENTER_Y


class SSDMobileNetPreProcessor(PreProcessor):
    @staticmethod
    def __call__(
        images: Sequence[Union[str, np.ndarray]]
    ) -> Tuple[npt.ArrayLike, List[Dict[str, Any]]]:
        """Preprocess input images to a batch of input tensors.

        Args:
            images: A list of paths of image files (e.g., JPEG, PNG)
                or a stacked image loaded as a numpy array in BGR order or gray order.

        Returns:
            The first element is 3-channel images of 300x300 in NCHW format,
                and the second element is a list of context about the original image metadata.
                Please learn more about the outputs of preprocess (i.e., model inputs),
                please refer to [Inputs](ssd_mobilenet.md#inputs).

        """
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/main.py#L49-L51
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/dataset.py#L242-L249
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
            image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_LINEAR)
            image -= 127.5
            image /= 127.5
            image = image.transpose([2, 0, 1])
            batch_image.append(image)
            batch_preproc_param.append({"width": width, "height": height})
        return np.stack(batch_image, axis=0), batch_preproc_param


class SSDMobileNetPythonPostProcessor(PostProcessor):
    @staticmethod
    def __call__(
        model_outputs: Sequence[numpy.ndarray],
        contexts: Sequence[Dict[str, Any]],
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.6,
    ) -> List[List[ObjectDetectionResult]]:
        """Convert the outputs of this model to a list of bounding boxes, scores and labels

        Arguments:
            model_outputs: the outputs of the model. To learn more about the output of model,
                please refer to [Outputs](ssd_mobilenet.md#outputs).
            contexts: context coming from `preprocess()`

        Returns:
            Detected Bounding Box and its score and label represented as `ObjectDetectionResult`.
                To learn more about `ObjectDetectionResult`, 'Definition of ObjectDetectionResult' can be found below.

        Definitions of ObjectDetectionResult and LtrbBoundingBox:
            ::: furiosa.models.vision.postprocess.LtrbBoundingBox
                options:
                    show_source: true
            ::: furiosa.models.vision.postprocess.ObjectDetectionResult
                options:
                    show_source: true
        """
        assert (
            len(model_outputs) == NUM_OUTPUTS
        ), f"the number of model outputs must be {NUM_OUTPUTS}, but {len(model_outputs)}"
        batch_size = model_outputs[0].shape[0]
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L94-L97
        class_logits = [
            output.transpose((0, 2, 3, 1)).reshape((batch_size, -1, NUM_CLASSES))
            for output in model_outputs[0:6]
        ]
        box_regression = [
            output.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 4))
            for output in model_outputs[6:12]
        ]
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L144-L166
        class_logits = np.concatenate(class_logits, axis=1)  # type: ignore[assignment]
        box_regression = np.concatenate(box_regression, axis=1)  # type: ignore[assignment]
        batch_scores = sigmoid(class_logits)  # type: ignore[arg-type]
        batch_boxes = _decode_boxes(box_regression)  # type: ignore[arg-type]

        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L178-L185
        batch_results = []
        for scores, boxes, preproc_params in zip(
            batch_scores, batch_boxes, contexts
        ):  # loop mini-batch
            width, height = preproc_params["width"], preproc_params["height"]
            boxes, labels, scores = _filter_results(
                scores,
                boxes,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
            )
            cal_boxes = calibration_ltrbbox(boxes, width, height)
            predicted_result = []
            for b, l, s in zip(cal_boxes, labels, scores):
                bb_list = b.tolist()
                predicted_result.append(
                    ObjectDetectionResult(
                        index=l,
                        label=CLASSES[l],
                        score=s,
                        boundingbox=LtrbBoundingBox(
                            left=bb_list[0], top=bb_list[1], right=bb_list[2], bottom=bb_list[3]
                        ),
                    )
                )
            batch_results.append(predicted_result)
        return batch_results


class SSDMobileNetNativePostProcessor(PostProcessor):
    def __init__(self, dfg: bytes):
        self._native = native.ssd_mobilenet.RustPostProcessor(dfg)

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


class SSDMobileNet(ObjectDetectionModel):
    """MLCommons MobileNet v1 model"""

    postprocessor_map: Dict[Platform, Type[PostProcessor]] = {
        Platform.PYTHON: SSDMobileNetPythonPostProcessor,
        Platform.RUST: SSDMobileNetNativePostProcessor,
    }

    @staticmethod
    def get_artifact_name():
        return "mlcommons_ssd_mobilenet_v1_int8"

    @classmethod
    def load_aux(cls, artifacts: Dict[str, bytes], use_native: bool = True):
        dfg = artifacts[EXT_DFG]
        if use_native and dfg is None:
            raise ArtifactNotFound(cls.get_artifact_name(), EXT_DFG)
        postproc_type = Platform.RUST if use_native else Platform.PYTHON
        logger.debug(f"Using {postproc_type.name} postprocessor")
        postprocessor = get_field_default(cls, "postprocessor_map")[postproc_type](dfg)
        return cls(
            name="MLCommonsSSDMobileNet",
            source=artifacts[EXT_ONNX],
            dfg=dfg,
            enf=artifacts[EXT_ENF],
            format=Format.ONNX,
            family="MobileNetV1",
            version="v1.1",
            metadata=Metadata(
                description="SSD MobileNet model for MLCommons v1.1",
                publication=Publication(url="https://arxiv.org/abs/1704.04861.pdf"),
            ),
            preprocessor=SSDMobileNetPreProcessor(),
            postprocessor=postprocessor,
        )
