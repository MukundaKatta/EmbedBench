.PHONY: install test lint format clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/embedbench/

format:
	ruff format src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache .mypy_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
