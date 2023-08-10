SHELL=/bin/bash -o pipefail

ONNXRUNTIME_VERSION := 1.15.1-?
TOOLCHAIN_VERSION := 0.10.0-?
LIBHAL_VERSION := 0.11.0-?

.PHONY: check-docker-tag toolchain lint test unit_tests examples regression-test-all \
regression-test-resnet50 regression-test-ssd-mobilenet \
regression-test-ssd-resnet34 regression-test-yolov5 doc docker-build docker-push \
regression-test-efficientnet-b0 regression-test-efficientnet-v2-s

check-docker-tag:
ifndef DOCKER_TAG
	$(error "DOCKER_TAG is not set")
endif

toolchain:
	apt-get update
	apt-get install -y --allow-downgrades libonnxruntime=$(ONNXRUNTIME_VERSION)
	apt-get install -y --allow-downgrades furiosa-compiler=$(TOOLCHAIN_VERSION)
	apt-get install -y --allow-downgrades furiosa-libhal-warboy=$(LIBHAL_VERSION)

lint:
	isort --diff --check .
	black --diff --check .
	ruff check .

test:
	pytest ./tests -s

unit_tests:
	pytest ./tests/unit/ -s

examples:
	for f in $$(ls docs/examples/*.py | grep -v "ssd_resnet34"); do echo"";echo "[TEST] $$f ..."; python3 $$f || exit 1; done

regression-test-all:
	pytest ./tests/bench/

regression-test-resnet50:
	pytest ./tests/bench/test_resnet50.py

regression-test-ssd-mobilenet:
	pytest ./tests/bench/test_ssd_mobilenet.py

regression-test-ssd-resnet34:
	pytest ./tests/bench/test_ssd_resnet34.py

regression-test-yolov5:
	pytest -s ./tests/bench/test_yolov5m.py && \
	pytest -s ./tests/bench/test_yolov5l.py

regression-test-efficientnet-b0:
	pytest -s ./tests/bench/test_efficientnet_b0.py

regression-test-efficientnet-v2-s:
	pytest -s ./tests/bench/test_efficientnet_v2_s.py

doc:
	mkdocs build

docker-build: check-docker-tag
	DOCKER_BUILDKIT=1 docker build -t asia-northeast3-docker.pkg.dev/next-gen-infra/furiosa-ai/furiosa-models:${DOCKER_TAG}  --secret id=.netrc,src=/root/.netrc --secret id=furiosa.conf,src=/etc/apt/auth.conf.d/furiosa.conf -f docker/Dockerfile ./docker/

docker-push:
	docker push asia-northeast3-docker.pkg.dev/next-gen-infra/furiosa-ai/furiosa-models:${DOCKER_TAG}
