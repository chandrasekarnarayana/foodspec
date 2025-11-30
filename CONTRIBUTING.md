# Contributing to foodspec

Thank you for your interest in contributing to foodspec!  
This project is developed and maintained by a scientific researcher, and contributions are welcome as long as they follow a few simple guidelines.

## Before You Start

- Please open an issue if you want to discuss new features, improvements, or ideas.
- For bug reports, include a clear description and a minimal reproducible example if possible.

## Development Setup

To set up a local development environment:

```bash
git clone https://github.com/chandrasekarnarayana/foodspec.git
cd foodspec
pip install -e ".[dev]"
```

## Code Style

- Follow PEP 8 conventions (Black formatting + Ruff linting).
- Ensure functions are clear, documented, and focused.
- Use type hints where possible.
- Format and lint:

```bash
black .
ruff check .
```

## Tests

- All new code should include tests.
- Run the full suite:

```bash
pytest
```

- Coverage is expected to remain above 80%.

## Documentation

- If you add new functionality, please update the relevant files under `docs/` and, if appropriate, add an example or CLI usage note.

## Pull Requests

A good pull request includes:

- A clear description of the change and motivation.
- Tests demonstrating correctness.
- Updated documentation (if needed).

## Communication

If you have questions, feedback, or want to collaborate:

📧 chandrasekarnarayana@gmail.com

Thank you for helping make foodspec a reliable, research-grade toolkit for Raman and FTIR spectroscopy.
