from typing import Any, ClassVar, Dict, List, Sequence, Tuple, Type, Union

import cv2
import numpy as np

from ..._utils import validate_postprocessor_type
from ...types import (
    Format,
    Metadata,
    Platform,
    PoseEstimationModel,
    PostProcessor,
    PreProcessor,
    Publication,
)
from ..preprocess import read_image_opencv_if_needed
from .postprocess import PoseEstimationResult, YOLOv7w6PosePostProcessor

_INPUT_SIZE = (640, 384)


def _letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int] = _INPUT_SIZE,
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """Letterboxing is a method for fitting an image within
       a constrained space dictated by the model requirements.
       If some axis is smaller than new_shape, it will be filled with color.
    Args:
        im (np.ndarray): a numpy image. Its shape must be Channel x Height x Width.
        new_shape (Tuple[int, int], optional): Targeted Image size. Defaults to default input size
            (640, 640).
        color (Tuple[int, int, int], optional): Padding Color Value. Defaults to (114, 114, 114).
        auto (bool, optional): If True, calculate padding width and height along to stride. Default
            to True.
        scaleFill (bool, optional): If True and auto is False, stretch an give image to target shape
            without any padding. Default to False.
        scaleup (bool, optional): If True, only scale down. Default to True.
        stride (int, optional): an output size of an image must be divied by a stride it is
            dependent on a model. This is only valid for auto is True. Default to 32.

    Returns:
        Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
            The first element is an padded-resized image. The second element is a resized image.
            The last element is padded sizes, respectively width and height.
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # divide padding into 2 sides
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def _resize(img, model_input_size):
    w, h = model_input_size
    return _letterbox(img, (h, w), auto=False)


class YOLOv7w6PosePreProcessor(PreProcessor):
    @staticmethod
    def __call__(
        images: Sequence[Union[str, np.ndarray]], with_scaling: bool = False
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Preprocess input images to a batch of input tensors

        Args:
            images: Color images have (NHWC: Batch, Height, Width, Channel) dimensions.
            with_scaling: Whether to apply model-specific techniques that involve scaling the
                model's input and converting its data type to float32. Refer to the code to gain a
                precise understanding of the techniques used. Defaults to False.

        Returns:
            a pre-processed image, scales and padded sizes(width,height) per images.
                The first element is a stacked numpy array containing a batch of images.
                To learn more about the outputs of preprocess (i.e., model inputs),
                please refer to [YOLOv7w6Pose Inputs](yolov7_w6_pose.md#inputs).

                The second element is a list of dict objects about the original images.
                Each dict object has the following keys. 'scale' key of the returned dict has a
                rescaled ratio per width(=target/width) and height(=target/height), and the 'pad'
                key has padded width and height pixels. Specially, the last dictionary element of
                returning tuple will be passed to postprocessing as a parameter to calculate
                predicted coordinates on normalized coordinates back to an input image coordinator.
        """
        # image format must be chw
        batched_image = []
        batched_proc_params = []
        if isinstance(images, str):
            images = [images]
        for image in images:
            image = read_image_opencv_if_needed(image)
            assert image.dtype == np.uint8
            if with_scaling:
                image = image.astype(np.float32)
            image, (sx, sy), (padw, padh) = _resize(image, _INPUT_SIZE)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if with_scaling:
                image /= 255.0
            image = image.transpose([2, 0, 1])  # NHWC -> NCHW
            scale = sx
            batched_image.append(image)
            batched_proc_params.append({"scale": scale, "pad": (padw, padh)})

        return np.stack(batched_image, axis=0), batched_proc_params


class YOLOv7w6Pose(PoseEstimationModel):
    """YOLOv7 w6 Pose Estimation model"""

    postprocessor_map: ClassVar[Dict[Platform, Type[PostProcessor]]] = {
        Platform.PYTHON: YOLOv7w6PosePostProcessor,
    }

    @staticmethod
    def visualize(image: np.ndarray, results: List[PoseEstimationResult]):
        """This visualize function is an example of how to visualize the output of the model.
        It draws a skeleton of the human body on the input image in in-place manner.

        Args:
            image: an input image
            results: a list of PoseEstimationResult objects
        """

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
        palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            np.int32,
        )

        skeletons = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

        pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

        def to_int_position(keypoint):
            return tuple(map(int, [keypoint.x, keypoint.y]))

        def is_valid_keypoint(keypoint):
            return (
                keypoint.x % 640 != 0
                and keypoint.y % 640 != 0
                and keypoint.x >= 0
                and keypoint.y >= 0
            )

        for result in results:
            for color, keypoint_name in zip(pose_kpt_color, keypoints):
                point = getattr(result, keypoint_name)
                if is_valid_keypoint(point):
                    cv2.circle(
                        image,
                        to_int_position(point),
                        radius=3,
                        color=color.tolist(),
                        thickness=-1,
                    )

            for color, skeleton in zip(pose_limb_color, skeletons):
                pos1 = getattr(result, keypoints[skeleton[0] - 1])
                pos2 = getattr(result, keypoints[skeleton[1] - 1])
                if is_valid_keypoint(pos1) and is_valid_keypoint(pos2):
                    cv2.line(
                        image,
                        to_int_position(pos1),
                        to_int_position(pos2),
                        color.tolist(),
                        thickness=2,
                    )

    def __init__(self, *, postprocessor_type: Union[str, Platform] = Platform.PYTHON):
        postprocessor_type = Platform(postprocessor_type)
        validate_postprocessor_type(postprocessor_type, self.postprocessor_map.keys())
        super().__init__(
            name="YOLOv7w6Pose",
            format=Format.ONNX,
            family="YOLO",
            version="v7",
            metadata=Metadata(
                description="YOLOv7 w6 Pose Estimation model",
                publication=Publication(url="https://github.com/WongKinYiu/yolov7#pose-estimation"),
            ),
            preprocessor=YOLOv7w6PosePreProcessor(),
            postprocessor=self.postprocessor_map[postprocessor_type](),
        )
        self._artifact_name = "yolov7_w6_pose"
