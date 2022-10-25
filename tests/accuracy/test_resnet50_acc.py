import os
from pathlib import Path
from typing import Any, List

import tqdm

from furiosa.models.vision import ResNet50
from furiosa.models.vision.common.datasets import imagenet1k
from furiosa.models.vision.resnet50 import NativePostProcessor, postprocess, preprocess
from furiosa.runtime import session

# NOTE: e2e-testing = 76.126 %, mlperf submission = 76.106 %
EXPECTED_ACCURACY = 75.982
EXPECTED_ACCURACY_NATIVE = 76.002
CLASSES: List[str] = imagenet1k.ImageNet1k_CLASSES


def test_mlcommons_resnet50_accuracy():
    imagenet_val_images = os.environ.get('IMAGENET_VAL_IMAGES')
    imagenet_val_labels = os.environ.get('IMAGENET_VAL_LABELS')

    if imagenet_val_images is None or imagenet_val_labels is None:
        raise Exception("Environment variables not set")
    imagenet_val_images = Path(imagenet_val_images)
    imagenet_val_labels = Path(imagenet_val_labels)

    resnet50 = ResNet50.load()

    correct_predictions, incorrect_predictions = 0, 0
    image_paths = list(imagenet_val_images.glob("*.[Jj][Pp][Ee][Gg]"))
    with open(imagenet_val_labels, encoding="ascii") as file:
        image_filename_to_label = {
            image_filename: int(label) for image_filename, label in (line.split() for line in file)
        }

    with session.create(resnet50.source) as sess:
        for image_path in tqdm.tqdm(image_paths):
            image = preprocess(str(image_path))
            output = postprocess(sess.run(image).numpy())

            if output == CLASSES[image_filename_to_label[image_path.name]]:
                correct_predictions += 1
            else:
                incorrect_predictions += 1

    total_predictions = correct_predictions + incorrect_predictions
    accuracy = 100.0 * correct_predictions / total_predictions
    print("accuracy :", accuracy, "% (", correct_predictions, "/", total_predictions, ")")
    assert accuracy == EXPECTED_ACCURACY, "Accuracy check failed"


def test_mlcommons_resnet50_with_native_pp_accuracy():
    imagenet_val_images = os.environ.get('IMAGENET_VAL_IMAGES')
    imagenet_val_labels = os.environ.get('IMAGENET_VAL_LABELS')

    if imagenet_val_images is None or imagenet_val_labels is None:
        raise Exception("Environment variables not set")
    imagenet_val_images = Path(imagenet_val_images)
    imagenet_val_labels = Path(imagenet_val_labels)

    model = ResNet50.load(use_native_post=True)
    postprocessor = NativePostProcessor(model)

    correct_predictions, incorrect_predictions = 0, 0
    image_paths = list(imagenet_val_images.glob("*.[Jj][Pp][Ee][Gg]"))
    with open(imagenet_val_labels, encoding="ascii") as file:
        image_filename_to_label = {
            image_filename: int(label) for image_filename, label in (line.split() for line in file)
        }

    with session.create(model.enf) as sess:
        for image_path in tqdm.tqdm(image_paths):
            image = preprocess(str(image_path))
            output = postprocessor.eval(sess.run(image).numpy())

            if output - 1 == image_filename_to_label[image_path.name]:
                correct_predictions += 1
            else:
                incorrect_predictions += 1

    total_predictions = correct_predictions + incorrect_predictions
    accuracy = 100.0 * correct_predictions / total_predictions
    print("accuracy :", accuracy, "% (", correct_predictions, "/", total_predictions, ")")
    assert accuracy == EXPECTED_ACCURACY_NATIVE, "Accuracy check failed"
