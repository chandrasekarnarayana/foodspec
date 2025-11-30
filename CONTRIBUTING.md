# Contributing to foodspec

Thanks for your interest in improving foodspec! Please follow these guidelines to help us review and merge changes efficiently.

## Getting started
- Install development dependencies: `pip install -e ".[dev]"`
- Run formatting and linting: `black . && ruff check .`
- Run tests with coverage: `pytest`

## Workflow
- Open an issue or discussion for significant changes.
- Use feature branches and small, focused commits.
- Include tests for new functionality and bug fixes.
- Update docs or examples when behavior changes.

## Code style
- Follow PEP 8, enforced by Black and Ruff.
- Keep functions small and focused; prefer type hints and docstrings.
- Add minimal, helpful comments when logic is non-obvious.

## Pull requests
- Ensure CI passes (format, lint, tests).
- Describe the change, motivation, and testing performed.
- Be responsive to review feedback; we iterate together.

## Community
- Treat others with respect and empathy.
- Abide by our Code of Conduct (see CODE_OF_CONDUCT.md).

