PYTHON ?= python
MVP_DIR := ciphers/kirchenbauer_et_al/binary_classification_mvp

.PHONY: check mypy ruff

check: ruff mypy

mypy:
	$(PYTHON) -m mypy --disable-error-code=import-untyped --explicit-package-bases $(MVP_DIR)

ruff:
	$(PYTHON) -m ruff format --check $(MVP_DIR)
	$(PYTHON) -m ruff check $(MVP_DIR)
