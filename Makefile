SHELL=/bin/bash -o pipefail

.PHONY: check-docker-tag toolchain lint test unit_tests examples regression-test-all \
regression-test-resnet50 regression-test-ssd-mobilenet \
regression-test-ssd-resnet34 regression-test-yolov5 doc docker-build docker-push

check-docker-tag:
ifndef DOCKER_TAG
	$(error "DOCKER_TAG is not set")
endif

toolchain:
	env TOOLCHAIN_VERSION=0.8.0-2+nightly-221023 sh -c 'apt-get install -y --allow-downgrades furiosa-libcompiler=$$TOOLCHAIN_VERSION furiosa-libnux-extrinsic=$$TOOLCHAIN_VERSION furiosa-libnux=$$TOOLCHAIN_VERSION'

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

regression-test-all:
	pytest ./tests/accuracy/

regression-test-resnet50:
	pytest ./tests/accuracy/test_resnet50_acc.py

regression-test-ssd-mobilenet:
	pytest ./tests/accuracy/test_ssd_mobilenet_acc.py

regression-test-ssd-resnet34:
	pytest ./tests/accuracy/test_ssd_resnet34_acc.py

regression-test-yolov5:
	pytest -s ./tests/accuracy/test_yolov5l_acc.py	&&	\
	pytest -s ./tests/accuracy/test_yolov5m_acc.py

doc:
	mkdocs build

docker-build: check-docker-tag
	DOCKER_BUILDKIT=1 docker build -t asia-northeast3-docker.pkg.dev/next-gen-infra/furiosa-ai/furiosa-models:${DOCKER_TAG} --secret id=furiosa.conf,src=/etc/apt/auth.conf.d/furiosa.conf -f docker/Dockerfile ./docker/

docker-push:
	docker push asia-northeast3-docker.pkg.dev/next-gen-infra/furiosa-ai/furiosa-models:${DOCKER_TAG}
