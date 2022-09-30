SHELL=/bin/bash -o pipefail

.PHONY: lint
lint:
	isort --diff .
	black --diff .
	cargo fmt --all --check

.PHONY: test
test:
	cargo test --release
	pytest ./tests -s

.PHONY: unit_tests
unit_tests:
	pytest ./tests/unit/ -s

.PHONY: regression_tests
regression_tests:
	COCO_VAL_IMAGES=$$(realpath tests/data/coco/val2017) \
	COCO_VAL_LABELS=$$(realpath tests/data/coco/annotations/instances_val2017.json) \
	IMAGENET_VAL_IMAGES=$$(realpath tests/data/imagenet/val/) \
	IMAGENET_VAL_LABELS=$$(realpath tests/data/imagenet/aux/val.txt) \
	pytest ./tests/accuracy/ -s

.PHONY: regression-test-ssd-resnet34
regression-test-ssd-resnet34:
	COCO_VAL_IMAGES=$$(realpath tests/data/coco/val2017) \
	COCO_VAL_LABELS=$$(realpath tests/data/coco/annotations/instances_val2017.json) \
	pytest ./tests/accuracy/test_ssd_resnet34_acc.py -s

.PHONY: regression-test-yolov5
regression-test-yolov5:
	YOLOV5_DATASET_PATH=$$(realpath tests/data/bdd100k_val/) \
	pytest -s ./tests/accuracy/test_yolov5l_acc.py	&&	\
	pytest -s ./tests/accuracy/test_yolov5m_acc.py
