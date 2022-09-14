from pathlib import Path
from typing import Any, List

import tqdm

from furiosa.models.vision import ResNet50
from furiosa.models.vision.common.datasets import imagenet1k
from furiosa.models.vision.resnet50 import postprocess, preprocess
from furiosa.runtime import session

EXPECTED_ACCURACY = 76.126
CLASSES: List[str] = imagenet1k.ImageNet1k_CLASSES


def test_mlcommons_resnet50_accuracy():
    resnet50 = ResNet50()

    # TODO
    imagenet_val_images = Path("/root/src/e2e-testing/data/imagenet/val")
    imagenet_val_labels = Path("/root/src/e2e-testing/data/imagenet/aux/val.txt")

    correct_predictions, incorrect_predictions = 0, 0
    image_paths = list(imagenet_val_images.glob("*.[Jj][Pp][Ee][Gg]"))
    with open(imagenet_val_labels, encoding="ascii") as file:
        image_filename_to_label = {
            image_filename: int(label) for image_filename, label in (line.split() for line in file)
        }

    with session.create(resnet50.source) as sess:
        for image_path in tqdm.tqdm(image_paths):
            image = preprocess(str(image_path))
            output = postprocess(sess.run(image))

            if output == CLASSES[image_filename_to_label[image_path.name]]:
                correct_predictions += 1
            else:
                incorrect_predictions += 1

    total_predictions = correct_predictions + incorrect_predictions
    accuracy = 100.0 * correct_predictions / total_predictions
    assert accuracy == EXPECTED_ACCURACY
