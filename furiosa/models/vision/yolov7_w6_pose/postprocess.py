import pathlib
from typing import List

import numpy as np
from pydantic import BaseModel
import yaml

from ...types import PythonPostProcessor
from .pose_decoder import PythonPoseDecoder

# from furiosa.models.vision.yolov7_w6_pose import PoseEstimationResult


INPUT_SIZE = (640, 384)
OUTPUT_SHAPES = [
    (1, 18, 48, 80),
    (1, 153, 48, 80),
    (1, 18, 24, 40),
    (1, 153, 24, 40),
    (1, 18, 12, 20),
    (1, 153, 12, 20),
    (1, 18, 6, 10),
    (1, 153, 6, 10),
]
with open(pathlib.Path(__file__).parent / "cfg.yaml", "r") as config_file:
    ANCHORS = yaml.safe_load(config_file)["anchors"]


class Keypoint(BaseModel):
    x: float
    y: float
    confidence: float


class PoseEstimationResult(BaseModel):
    bounding_box: List[float]
    confidence: float

    nose: Keypoint
    left_eye: Keypoint
    right_eye: Keypoint
    left_ear: Keypoint
    right_ear: Keypoint
    left_shoulder: Keypoint
    right_shoulder: Keypoint
    left_elbow: Keypoint
    right_elbow: Keypoint
    left_wrist: Keypoint
    right_wrist: Keypoint
    left_hip: Keypoint
    right_hip: Keypoint
    left_knee: Keypoint
    right_knee: Keypoint
    left_ankle: Keypoint
    right_ankle: Keypoint


def _nms(box_scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    output = []
    for box_scores in box_scores:
        scores = box_scores[:, 4]
        boxes = box_scores[:, :4]
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
        output.append(box_scores[picked, :])
    return output


def _box_area(left_top: np.ndarray, right_bottom: np.ndarray):
    """Compute the areas of rectangles given two corners."""
    # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/utils.py#L89-L100
    width_height = np.clip(right_bottom - left_top, a_min=0.0, a_max=None)
    return width_height[..., 0] * width_height[..., 1]


def _box_iou(boxes1: np.ndarray, boxes2: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Return intersection-over-union (Jaccard index) of boxes."""
    # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/utils.py#L103-L119
    overlap_left_top = np.maximum(boxes1[..., :2], boxes2[..., :2])
    overlap_right_bottom = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    overlap_area = _box_area(overlap_left_top, overlap_right_bottom)
    area1 = _box_area(boxes1[..., :2], boxes1[..., 2:])
    area2 = _box_area(boxes2[..., :2], boxes2[..., 2:])
    return overlap_area / (area1 + area2 - overlap_area + eps)


class YOLOv7w6PosePostProcessor(PythonPostProcessor):
    def __init__(
        self,
        input_size=INPUT_SIZE,
        output_shapes=OUTPUT_SHAPES,
        anchors=ANCHORS,
        conf_thres=0.1,
        iou_thres=0.45,
    ) -> None:
        self.height = input_size[1]  # height only
        self.output_shapes = output_shapes
        self.anchors = np.float32(anchors)
        self.nkpt = 17
        self.no_kpt = 3 * self.nkpt  ## number of outputs per anchor for keypoints
        self.class_names = ["person"]

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.stride = self._compute_stride()
        self.pose_decoder = PythonPoseDecoder(
            nc=len(self.class_names),
            anchors=self.anchors,
            stride=self.stride,
            conf_thres=self.conf_thres,
        )

    def _get_class_count(self):
        return len(self.class_names)

    def _get_output_feat_count(self):
        return self.anchors.shape[0]

    def _get_anchor_per_layer_count(self):
        return self.anchors.shape[1]

    def _reshape_output(self, i, y_det, y_kpt):
        shape = (
            y_det.shape[0],
            self._get_anchor_per_layer_count(),
            self._get_class_count() + 5 + self.no_kpt,
            y_det.shape[2],
            y_det.shape[3],
        )
        return np.concatenate((y_det, y_kpt), axis=1).reshape(shape).transpose(0, 1, 3, 4, 2)

    def __call__(self, feats_batched, preproc_params):
        assert len(feats_batched), "feats_batched must not be empty"

        boxes_batched = []
        batch_size = feats_batched[0].shape[0]
        assert batch_size == len(preproc_params)

        for i in range(batch_size):
            scale, (padw, padh) = preproc_params[i]["scale"], preproc_params[i]["pad"]
            if isinstance(scale, (tuple, list)):
                assert len(scale) == 2 and scale[0] == scale[1]
                scale = scale[0]
            feats = [f[i : i + 1] for f in feats_batched]

            y_dets, y_kpts = feats[0::2], feats[1::2]
            y_dets_kpts = [
                self._reshape_output(j, y_det, y_kpt)
                for j, (y_det, y_kpt) in enumerate(zip(y_dets, y_kpts))
            ]
            boxes_kpts_dec = self.pose_decoder(y_dets_kpts)
            boxes_kpts_dec = _nms(boxes_kpts_dec, self.iou_thres)[0]

            # rescale boxes
            boxes_kpts_dec[:, [0, 2]] = (1 / scale) * (boxes_kpts_dec[:, [0, 2]] - padw)
            boxes_kpts_dec[:, [1, 3]] = (1 / scale) * (boxes_kpts_dec[:, [1, 3]] - padh)

            boxes_kpts_dec[:, 6 + 0 : 6 + 3 * self.nkpt : 3] = (1 / scale) * (
                boxes_kpts_dec[:, 6 + 0 : 6 + 3 * self.nkpt : 3] - padw
            )
            boxes_kpts_dec[:, 6 + 1 : 6 + 3 * self.nkpt : 3] = (1 / scale) * (
                boxes_kpts_dec[:, 6 + 1 : 6 + 3 * self.nkpt : 3] - padh
            )

            # rescale keypoints

            boxes_batched.append(boxes_kpts_dec)

        return self._format_output(boxes_batched)

    def _format_output(self, boxes_batched):
        out_batched = []
        for dets in boxes_batched:
            box = dets[:, 0:4]
            conf = dets[:, 4]
            keypoints = dets[:, 6 : 6 + 3 * self.nkpt]

            out_batched.append(
                {
                    "box": box,
                    "conf": conf,
                    "pose": keypoints,
                }
            )
        final_output = []

        keypoints = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        for item in out_batched:
            parsed = []
            for i in range(len(item["box"])):
                result = {
                    "bounding_box": item["box"][i].tolist(),
                    "confidence": item["conf"][i].tolist(),
                }
                for key, (x, y, conf) in zip(keypoints, zip(*[iter(item["pose"][i])] * 3)):
                    result[key] = {"x": x, "y": y, "confidence": conf}

                parsed.append(PoseEstimationResult(**result))
            final_output.append(parsed)
        return final_output

    def _compute_stride(self):
        img_h = self.height
        output_shapes = self.output_shapes[::2]
        feat_h = np.float32([shape[2] for shape in output_shapes])
        strides = img_h / feat_h
        return strides
