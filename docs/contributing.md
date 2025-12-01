# Contributing

FoodSpec is a research project led by Chandrasekar Subramani Narayan. External collaborators are welcome; please align with the style and testing expectations to keep the protocol robust.

## How to propose changes
- Open an issue to discuss new features or bugs before a major change.
- Fork and submit PRs for code/docs improvements; keep changes focused and documented.

## Development setup
```bash
git clone https://github.com/chandrasekarnarayana/foodspec.git
cd foodspec
pip install -e ".[dev]"
```
Run formatting/linting/tests before PRs:
```bash
black .
ruff check .
pytest
```

## Style and scope
- Follow existing type hints and sklearn-like patterns.
- Keep tests deterministic (no network) and add coverage for new public APIs.
- Update docs when adding user-facing features or CLI flags.

Contact: chandrasekarnarayana@gmail.com for coordination or questions.
