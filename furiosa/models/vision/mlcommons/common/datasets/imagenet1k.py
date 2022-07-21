import os

ImageNet1k_CLASSES = [
    label.strip()
    for label in open(os.path.join(os.path.dirname(__file__), "imagenet-1k-label.txt")).readlines()
]
