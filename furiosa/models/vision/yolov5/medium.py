import pathlib

import cv2
import numpy as np
import yaml
from furiosa.registry import Model

from .box_decode.box_decoder import BoxDecoderC
from .utils.transforms import letterbox
from .core import preprocess, postprocess

INPUT_SIZE = (640, 640)
IOU_THRES = 0.45
OUTPUT_SHAPES = [(1, 45, 80, 80), (1, 45, 40, 40), (1, 45, 20, 20)]

with open(pathlib.Path(__file__).parent / "datasets/yolov5m/cfg.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    ANCHORS = np.float32(cfg["anchors"])
    CLASS_NAMES = cfg["class_names"]


def _compute_stride():
    img_h = INPUT_SIZE[1]
    feat_h = np.float32([shape[2] for shape in OUTPUT_SHAPES])
    strides = img_h / feat_h
    return strides


BOX_DECODER = BoxDecoderC(
    nc=len(CLASS_NAMES),
    anchors=ANCHORS,
    stride=_compute_stride(),
    conf_thres=0.25,
)


class YoloV5MediumModel(Model):
    def compile_config(self, input_format="hwc"):
        return {
            "without_quantize": {
                    "parameters": [
                        {
                            "input_min": 0.0, 
                            "input_max": 1.0, 
                            "permute": [0, 2, 3, 1] if input_format == "hwc" else [0, 1, 2, 3] # bchw
                        }
                    ]
                }
            }

    def get_class_names(self):
        return CLASS_NAMES

    def get_class_count(self):
        return len(CLASS_NAMES)

    def get_output_feat_count(self):
        return ANCHORS.shape[0]

    def get_anchor_per_layer_count(self):
        return ANCHORS.shape[1]

    def preprocess(self, img: np.array, input_color_format: str="bgr"):
        return preprocess(img, INPUT_SIZE, input_color_format)

    def postprocess(self, feats, preproc_param):
        return postprocess(feats, BOX_DECODER, self.get_anchor_per_layer_count(), self.get_class_names(), preproc_param, IOU_THRES)