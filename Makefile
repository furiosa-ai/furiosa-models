SHELL=/bin/bash -o pipefail

.PHONY: lint
lint:
	isort --diff .
	black --diff .
	cargo fmt --all --check
	cargo -q clippy --all-targets -- -D rust_2018_idioms -D warnings

.PHONY: test
test:
	cargo test --release
	pytest ./tests -s

.PHONY: unit_tests
unit_tests:
	pytest ./tests/unit/ -s

.PHONY: regression_tests
regression_tests:
	pytest ./tests/accuracy/ -s
