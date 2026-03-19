.PHONY: install test bench run lint clean

PYTHON := python3
SRC := src/bpe_tokenizer

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short \
		--cov=$(SRC) \
		--cov-report=term-missing \
		--cov-report=xml

bench:
	$(PYTHON) benchmarks/bench_tokenizer.py

run:
	$(PYTHON) examples/quickstart.py

lint:
	$(PYTHON) -m ruff check $(SRC) tests/ examples/ benchmarks/ 2>/dev/null || echo "ruff not installed, skipping"
	$(PYTHON) -m mypy $(SRC) --ignore-missing-imports 2>/dev/null || echo "mypy not installed, skipping"

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage coverage.xml htmlcov/ dist/ build/ .mypy_cache/ .ruff_cache/ .pytest_cache/
