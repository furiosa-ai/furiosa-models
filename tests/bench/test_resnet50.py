import os
from pathlib import Path
from typing import List

import cv2
import tqdm

from furiosa.models.vision import ResNet50
from furiosa.models.vision.common.datasets import imagenet1k
from furiosa.runtime import session

# NOTE: e2e-testing = 76.126 %, mlperf submission = 76.106 %
EXPECTED_ACCURACY = 75.982
EXPECTED_ACCURACY_NATIVE = 76.002
CLASSES: List[str] = imagenet1k.ImageNet1k_CLASSES


def test_mlcommons_resnet50_accuracy(benchmark):
    imagenet_val_images = Path(os.environ.get('IMAGENET_VAL_IMAGES', 'tests/data/imagenet/val'))
    imagenet_val_labels = Path(
        os.environ.get('IMAGENET_VAL_LABELS', 'tests/data/imagenet/aux/val.txt')
    )

    model = ResNet50.load(use_native=False)

    image_paths = list(imagenet_val_images.glob("*.[Jj][Pp][Ee][Gg]"))
    with open(imagenet_val_labels, encoding="ascii") as file:
        image_filename_to_label = {
            image_filename: int(label) for image_filename, label in (line.split() for line in file)
        }

    image_src_iter = iter(tqdm.tqdm(image_paths))
    num_images = len(image_paths)
    global correct_predictions, incorrect_predictions
    correct_predictions, incorrect_predictions = 0, 0

    def read_image():
        image_src = next(image_src_iter)
        image = cv2.imread(str(image_src))
        return (image, CLASSES[image_filename_to_label[image_src.name]]), {}

    def workload(image, answer):
        global correct_predictions, incorrect_predictions
        image, _ = model.preprocess(image)
        output = model.postprocess(sess.run(image).numpy())

        if output == answer:
            correct_predictions += 1
        else:
            incorrect_predictions += 1

    sess = session.create(model)
    benchmark.pedantic(workload, setup=read_image, rounds=num_images)
    sess.close()

    total_predictions = correct_predictions + incorrect_predictions
    accuracy = 100.0 * correct_predictions / total_predictions
    print("accuracy :", accuracy, "% (", correct_predictions, "/", total_predictions, ")")
    assert accuracy == EXPECTED_ACCURACY, "Accuracy check failed"


def test_mlcommons_resnet50_with_native_pp_accuracy(benchmark):
    imagenet_val_images = Path(os.environ.get('IMAGENET_VAL_IMAGES', 'tests/data/imagenet/val'))
    imagenet_val_labels = Path(
        os.environ.get('IMAGENET_VAL_LABELS', 'tests/data/imagenet/aux/val.txt')
    )

    model = ResNet50.load(use_native=True)

    image_paths = list(imagenet_val_images.glob("*.[Jj][Pp][Ee][Gg]"))
    with open(imagenet_val_labels, encoding="ascii") as file:
        image_filename_to_label = {
            image_filename: int(label) for image_filename, label in (line.split() for line in file)
        }

    image_src_iter = iter(tqdm.tqdm(image_paths))
    num_images = len(image_paths)
    global correct_predictions, incorrect_predictions
    correct_predictions, incorrect_predictions = 0, 0

    def read_image():
        image_src = next(image_src_iter)
        image = cv2.imread(str(image_src))
        return (image, CLASSES[image_filename_to_label[image_src.name]]), {}

    def workload(image, answer):
        global correct_predictions, incorrect_predictions
        image, _ = model.preprocess(image)
        output = model.postprocess(sess.run(image).numpy())

        if output == answer:
            correct_predictions += 1
        else:
            incorrect_predictions += 1

    sess = session.create(model)
    benchmark.pedantic(workload, setup=read_image, rounds=num_images)
    sess.close()

    total_predictions = correct_predictions + incorrect_predictions
    accuracy = 100.0 * correct_predictions / total_predictions
    print("accuracy :", accuracy, "% (", correct_predictions, "/", total_predictions, ")")
    assert accuracy == EXPECTED_ACCURACY_NATIVE, "Accuracy check failed"
