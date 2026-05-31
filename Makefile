.PHONY: install test lint format clean

install:
	pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

clean:
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf src/difflc/__pycache__
	rm -rf tests/__pycache__
	rm -rf *.egg-info
