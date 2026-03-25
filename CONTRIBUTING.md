# Contributing to EmbedBench

Thank you for your interest in contributing to EmbedBench! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a feature branch: `git checkout -b feature/my-feature`
4. Install development dependencies: `make install`

## Development Workflow

```bash
# Install in editable mode with dev dependencies
make install

# Run tests
make test

# Lint code
make lint

# Auto-format code
make format
```

## Code Standards

- Follow PEP 8 conventions (enforced by ruff)
- Add type annotations to all public functions
- Write docstrings for all public classes and functions
- Keep test coverage high — add tests for new functionality

## Pull Request Process

1. Ensure all tests pass: `make test`
2. Ensure linting passes: `make lint`
3. Update documentation if you change public APIs
4. Write a clear PR description explaining the change

## Adding a New Embedder

To add a new embedding strategy:

1. Create a class in `src/embedbench/core.py` that inherits from `BaseEmbedder`
2. Implement `fit()` and `embed()` methods
3. Add a corresponding config model in `src/embedbench/config.py`
4. Register the embedder name in `EmbedBench._build_embedders()`
5. Export from `__init__.py`
6. Add tests in `tests/test_core.py`

## Reporting Issues

- Use GitHub Issues to report bugs
- Include Python version, OS, and a minimal reproduction script
- Include the full traceback if applicable

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
