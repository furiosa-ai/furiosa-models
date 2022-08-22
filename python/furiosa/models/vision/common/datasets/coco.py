"""
implementation of imagenet dataset
"""

import logging
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco")

MobileNetSSD_CLASSES = [
    label.strip()
    for label in open(os.path.join(os.path.dirname(__file__), "coco-label-paper.txt")).readlines()
]

MobileNetSSD_Large_CLASSES = [
    label.strip()
    for label in open(
        os.path.join(os.path.dirname(__file__), "coco-label-ssdmobilelarge.txt")
    ).readlines()
]
