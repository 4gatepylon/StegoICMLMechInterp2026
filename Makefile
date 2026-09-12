.PHONY: format test

format:
	ruff format .
	ruff check --fix .

test:
	python -m pytest
