CONDA_ENV ?= stegobench
CONDA_RUN := conda run --no-capture-output -n $(CONDA_ENV)
PYTHON := $(CONDA_RUN) python
RUFF := $(CONDA_RUN) ruff

MVP_DIR := ciphers/kirchenbauer_et_al/binary_classification_mvp
MVP_TEST_DIR := $(MVP_DIR)/tests
MVP_SMOKE_CONFIG := $(MVP_DIR)/configurations/cpu_smoke.yaml
ARTIFACTS_DIR ?= $(CURDIR)/.context/binary_classification_mvp_smoke

.PHONY: test lint format format-check compile check smoke

test:
	$(PYTHON) -m unittest discover -s $(MVP_TEST_DIR) -v

lint:
	$(RUFF) check $(MVP_DIR)

format:
	$(RUFF) format $(MVP_DIR)

format-check:
	$(RUFF) format --check $(MVP_DIR)

compile:
	$(PYTHON) -m compileall -q $(MVP_DIR)

check: test lint format-check compile

smoke: check
	ARTIFACTS_DIR="$(ARTIFACTS_DIR)" $(PYTHON) -m ciphers.kirchenbauer_et_al.binary_classification_mvp.run_experiment \
		--config $(MVP_SMOKE_CONFIG) \
		--wandb-mode disabled
