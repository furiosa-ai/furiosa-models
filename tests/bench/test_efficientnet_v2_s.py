import os
from pathlib import Path
from typing import List

import tqdm

from furiosa.models.vision import EfficientNetV2s
from furiosa.models.vision.common.datasets import imagenet1k
from furiosa.runtime.sync import create_runner

EXPECTED_ACCURACY = 83.532
CLASSES: List[str] = imagenet1k.ImageNet1k_CLASSES


def test_efficientnetv2s_accuracy(benchmark):
    imagenet_val_images = Path(os.environ.get('IMAGENET_VAL_IMAGES', 'tests/data/imagenet/val'))
    imagenet_val_labels = Path(
        os.environ.get('IMAGENET_VAL_LABELS', 'tests/data/imagenet/aux/val.txt')
    )

    model = EfficientNetV2s()

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
        return (str(image_src), CLASSES[image_filename_to_label[image_src.name]]), {}

    def workload(image, answer):
        global correct_predictions, incorrect_predictions
        image, _ = model.preprocess(image)
        output = runner.run(image)
        output = model.postprocess(output)

        if output == answer:
            correct_predictions += 1
        else:
            incorrect_predictions += 1

    runner = create_runner(model.model_source())
    benchmark.pedantic(workload, setup=read_image, rounds=num_images)
    runner.close()

    total_predictions = correct_predictions + incorrect_predictions
    accuracy = 100.0 * correct_predictions / total_predictions
    print("accuracy :", accuracy, "% (", correct_predictions, "/", total_predictions, ")")
    assert accuracy == EXPECTED_ACCURACY, "Accuracy check failed"
