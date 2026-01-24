# Contributing to FoodSpec

Thank you for your interest in contributing to FoodSpec! This project is developed as a research-grade, protocol-driven spectroscopy framework. Contributions are welcome and should follow these engineering guardrails to maintain reliability, reproducibility, and scientific rigor.

---

## Before You Start

- **Open an issue first** for any new feature, improvement, or architectural change. This helps coordinate effort and aligns proposals with project direction.
- **For bug reports**, include a clear description, minimal reproducible example, and OS/Python version.
- **Read** [ENGINEERING_RULES.md](./docs/developer-guide/ENGINEERING_RULES.md) for the project's core principles.

---

## Development Setup

To set up a local development environment:

```bash
git clone https://github.com/chandrasekarnarayana/foodspec.git
cd foodspec
pip install -e ".[dev]"
```

Then run the guardrail checks:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
pytest --cov=src/foodspec tests/ -v
```

---

## Core Engineering Rules (Non-Negotiables)

All contributions **must** adhere to these principles:

### 1. **Deterministic Outputs**
- **Never rely on floating-point seeding or global random state.**
- Pass `random_state` or `seed` explicitly as parameters to all probabilistic functions.
- Example:
  ```python
  def synthetic_spectrum(n_peaks=10, seed=None):
      """Generate a synthetic spectrum.
      
      Parameters
      ----------
      n_peaks : int
          Number of peaks.
      seed : int, optional
          Random seed for reproducibility. If None, results are non-deterministic.
      
      Examples
      --------
      >>> spec = synthetic_spectrum(seed=42)  # Always same result
      """
      rng = np.random.default_rng(seed)
      # ... use rng for all randomness
  ```

### 2. **No Hidden Global State**
- Avoid module-level mutable state (lists, dicts, global config objects).
- Use `dataclasses` or `pydantic` models for configuration; pass config explicitly.
- Singletons are **prohibited** unless documented in ENGINEERING_RULES.md with justification.
- Example (❌ Bad):
  ```python
  _CONFIG = {}  # Global state, hidden coupling
  def process(data):
      return apply_config(_CONFIG, data)
  ```
- Example (✅ Good):
  ```python
  @dataclass
  class ProcessConfig:
      method: str = "baseline"
  
  def process(data, config: ProcessConfig = ProcessConfig()):
      return apply_method(data, config.method)
  ```

### 3. **Every Public Function/Class Must Have Docstring + Example**
- **Docstring format**: Follow NumPy/SciPy style.
- **Examples section required** in all public APIs.
- Type hints are **mandatory**.
- Example:
  ```python
  def compute_snr(spectrum: np.ndarray, signal_region: tuple[int, int], 
                  noise_region: tuple[int, int]) -> float:
      """Compute signal-to-noise ratio.
      
      Parameters
      ----------
      spectrum : np.ndarray
          1D spectral intensity array.
      signal_region : tuple[int, int]
          (start, end) indices for signal.
      noise_region : tuple[int, int]
          (start, end) indices for noise baseline.
      
      Returns
      -------
      float
          SNR in dB.
      
      Examples
      --------
      >>> s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
      >>> snr = compute_snr(s, signal_region=(1, 3), noise_region=(3, 5))
      >>> snr > 0
      True
      """
  ```

### 4. **Every New Feature Must Include Tests + Docs**
- **Test coverage minimum**: 80% for new code.
- **Tests file location**: `tests/test_<module>.py` mirrors `src/foodspec/<module>.py`.
- **Test structure**:
  ```python
  import pytest
  from foodspec import MyNewClass
  
  class TestMyNewClass:
      def test_basic_initialization(self):
          obj = MyNewClass(param=42)
          assert obj.param == 42
      
      def test_determinism_with_seed(self):
          result1 = obj.method(seed=42)
          result2 = obj.method(seed=42)
          np.testing.assert_array_equal(result1, result2)
      
      def test_invalid_input_raises_valueerror(self):
          with pytest.raises(ValueError, match="param must be positive"):
              MyNewClass(param=-1)
  ```
- **Documentation**: Add example to `docs/` or update [API reference](./docs/reference/).

### 5. **Metadata Schema Must Be Validated Early**
- Use `pydantic.BaseModel` or `dataclasses` with validation.
- Raise `ValueError` **immediately** if metadata is invalid (don't defer).
- Example:
  ```python
  from pydantic import BaseModel, field_validator
  
  class SpectrumMetadata(BaseModel):
      timestamp: str
      instrument_id: str
      wavelength_start: float
      wavelength_end: float
      
      @field_validator('timestamp')
      @classmethod
      def validate_iso8601(cls, v):
          try:
              datetime.fromisoformat(v)
          except ValueError:
              raise ValueError(f"timestamp must be ISO 8601, got {v}")
          return v
  ```

### 6. **Pipelines Must Be Serializable**
- All pipeline configs must serialize to JSON/YAML.
- Use `dataclasses` or `pydantic` (not plain dict).
- Example:
  ```python
  @dataclass
  class PreprocessingPipeline:
      steps: list[str]  # ["baseline_als", "normalize"]
      baseline_method: str = "als"
      normalize_to: str = "l2"
      
      def to_dict(self) -> dict:
          return asdict(self)
      
      @classmethod
      def from_dict(cls, d: dict):
          return cls(**d)
  ```

### 7. **Errors Must Be Actionable**
- Error messages must clearly state: **what failed**, **why**, and **how to fix**.
- Example (❌ Bad): `ValueError: Invalid metadata`
- Example (✅ Good): `ValueError: metadata['timestamp'] must be ISO 8601 format (got '2025-01-24 10:30:00'). Use datetime.isoformat() to format.`
- Avoid generic `Exception` or `RuntimeError`; use specific error types.

---

## Code Style & Tooling

### Formatting & Linting

Use these tools to enforce consistency:

```bash
# Format with Ruff (PEP 8 + Black style)
ruff format src/ tests/

# Lint with Ruff (500+ rules)
ruff check src/ tests/ --fix

# Type checking with mypy
mypy src/ --strict

# Run tests with pytest
pytest tests/ -v --cov=src/foodspec --cov-report=html
```

### Pre-commit Hook (Optional but Recommended)

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        args: [--strict]
```

Install: `pre-commit install`

### Configuration Files

- **ruff.toml** or **pyproject.toml** `[tool.ruff]` section
- **pyproject.toml** `[tool.mypy]` section for type checking
- **pyproject.toml** `[tool.pytest.ini_options]` for test discovery

---

## Pull Request Checklist

Before submitting a PR, ensure:

- [ ] **Issue linked**: PR references an open issue (e.g., "Fixes #123").
- [ ] **All tests pass**: `pytest tests/ -v` runs with no failures.
- [ ] **Coverage maintained**: `pytest --cov=src/foodspec` shows ≥80% coverage for new code.
- [ ] **Code formatted**: `ruff format .` applied.
- [ ] **Linting passes**: `ruff check . --fix` shows no errors.
- [ ] **Type checking passes**: `mypy src/` (strict mode) has no errors.
- [ ] **Determinism guaranteed**: All random operations use explicit `seed` parameter.
- [ ] **No global state**: No module-level mutable defaults or singletons.
- [ ] **Public APIs documented**: All public functions have docstring + example.
- [ ] **Backward compatibility**: No breaking changes without deprecation (see [COMPATIBILITY_PLAN.md](./docs/developer-guide/COMPATIBILITY_PLAN.md)).
- [ ] **Metadata validated**: Config classes use pydantic or dataclass validators.
- [ ] **Errors actionable**: Error messages guide users to resolution.
- [ ] **Serializable**: Pipelines/configs can be saved to JSON/YAML.

---

## Documentation

- If you add new functionality:
  - Update relevant files under `docs/` (e.g., `docs/reference/`, `docs/user-guide/`).
  - Add a worked example if the feature is user-facing.
  - Update `docs/api/` for API reference.

- For **breaking changes**:
  - Document in `RELEASE_NOTES.md`.
  - Provide migration guide.
  - Use deprecation warnings (`warnings.warn()`) for at least one minor version.

---

## Communication

If you have questions, feedback, or want to collaborate:

📧 chandrasekarnarayana@gmail.com

---

## References

- [ENGINEERING_RULES.md](./docs/developer-guide/ENGINEERING_RULES.md) — Detailed engineering principles
- [COMPATIBILITY_PLAN.md](./docs/developer-guide/COMPATIBILITY_PLAN.md) — Backward compatibility strategy
- [PyTest Documentation](https://docs.pytest.org/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [MyPy Documentation](https://mypy.readthedocs.io/)

---

Thank you for helping make FoodSpec a reliable, research-grade, protocol-driven toolkit for Raman and FTIR spectroscopy!
