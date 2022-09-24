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
	dvc pull tests/data/coco/val2017.dvc -r e2e-testing-data && \
	dvc pull tests/data/coco/annotations/instances_val2017.json.dvc -r e2e-testing-data && \
	dvc pull tests/data/imagenet/aux.dvc -r e2e-testing-data && \
	dvc pull tests/data/imagenet/val.dvc -r e2e-testing-data && \
	COCO_VAL_IMAGES=$$(realpath tests/data/coco/val2017) \
	COCO_VAL_LABELS=$$(realpath tests/data/coco/annotations/instances_val2017.json) \
	IMAGENET_VAL_IMAGES=$$(realpath tests/data/imagenet/val/) \
	IMAGENET_VAL_LABELS=$$(realpath tests/data/imagenet/aux/val.txt) \
	pytest ./tests/accuracy/ -s
