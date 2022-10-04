SHELL=/bin/bash -o pipefail

.PHONY: examples doc lint test unit_tests regression_tests regression-test-ssd-resnet34 regression-test-yolov5

lint:
	isort --diff .
	black --diff .
	cargo fmt --all --check

test:
	cargo test --release
	pytest ./tests -s

unit_tests:
	pytest ./tests/unit/ -s

examples:
	for f in $$(ls docs/examples/*.py); do echo"";echo "[TEST] $$f ..."; python3 $$f; done

regression_tests:
	COCO_VAL_IMAGES=$$(realpath tests/data/coco/val2017) \
	COCO_VAL_LABELS=$$(realpath tests/data/coco/annotations/instances_val2017.json) \
	IMAGENET_VAL_IMAGES=$$(realpath tests/data/imagenet/val/) \
	IMAGENET_VAL_LABELS=$$(realpath tests/data/imagenet/aux/val.txt) \
	pytest ./tests/accuracy/ -s

regression-test-ssd-resnet34:
	COCO_VAL_IMAGES=$$(realpath tests/data/coco/val2017) \
	COCO_VAL_LABELS=$$(realpath tests/data/coco/annotations/instances_val2017.json) \
	pytest ./tests/accuracy/test_ssd_resnet34_acc.py -s

regression-test-yolov5:
	YOLOV5_DATASET_PATH=$$(realpath tests/data/bdd100k_val/) \
	pytest -s ./tests/accuracy/test_yolov5l_acc.py	&&	\
	pytest -s ./tests/accuracy/test_yolov5m_acc.py

doc:
	mkdocs build
