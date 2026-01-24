# FoodSpec Engineering Rules — Quick Reference

**Print this or bookmark for daily development!**

---

## The 7 Non-Negotiables

### 1️⃣ Deterministic Outputs
✅ Pass `seed` explicitly to all functions with randomness  
✅ Use `np.random.default_rng(seed)` for numpy  
✅ Pass `random_state=seed` to sklearn/scipy  
❌ Never use `np.random.seed()` (modifies global state)

```python
def analyze(data, seed=None):
    rng = np.random.default_rng(seed)
    # use rng, not np.random
```

---

### 2️⃣ No Hidden Global State
✅ Pass config as parameter: `def func(config: MyConfig)`  
✅ Use `@dataclass` or `pydantic.BaseModel` for config  
✅ Instantiate objects explicitly  
❌ Never use module-level mutable defaults  
❌ Never use singletons (except with documented justification)

```python
@dataclass
class MyConfig:
    param: int = 5

def process(data, config=MyConfig()):
    return _impl(data, config.param)
```

---

### 3️⃣ Every Public Function/Class Has Docstring + Example
✅ NumPy-style docstring (Parameters, Returns, Raises, Examples)  
✅ Type hints on all parameters and return  
✅ Runnable example code  
❌ Missing docstring or example → don't merge

```python
def compute_metric(spectrum: np.ndarray, 
                   region: tuple[int, int]) -> float:
    """Compute quality metric.
    
    Parameters
    ----------
    spectrum : np.ndarray
        1D array.
    region : tuple[int, int]
        (start, end) indices.
    
    Returns
    -------
    float
        Metric value.
    
    Examples
    --------
    >>> spec = np.array([1.0, 2.0, 3.0])
    >>> compute_metric(spec, (0, 2))
    2.5
    """
```

---

### 4️⃣ Every New Feature Includes Tests + Docs
✅ ≥80% code coverage for new code  
✅ Test file: `tests/test_<module>.py`  
✅ Test class: `TestMyFeature` with descriptive test names  
✅ Update `docs/` or add to API reference  
❌ Feature without tests/docs → PR rejected

```python
class TestMyFeature:
    def test_basic_usage(self): ...
    def test_edge_case(self): ...
    def test_invalid_input_raises(self): 
        with pytest.raises(ValueError, match="..."):
            ...
```

---

### 5️⃣ Metadata Schema Validated Early
✅ Use `pydantic.BaseModel` with validators  
✅ Validate at entry point (before processing)  
✅ Raise `ValueError` immediately with clear message  
❌ Defer validation (errors become hard to debug)

```python
from pydantic import BaseModel, field_validator

class Meta(BaseModel):
    timestamp: str
    
    @field_validator('timestamp')
    @classmethod
    def check_iso(cls, v):
        datetime.fromisoformat(v)  # Raises if invalid
        return v
```

---

### 6️⃣ Pipelines Must Be Serializable
✅ Config as `@dataclass` or `pydantic` model  
✅ `.to_dict()` / `.from_dict()` methods  
✅ Saves to JSON/YAML  
✅ Roundtrip tests: `dict → obj → dict` are identical  
❌ Non-serializable state (functions, file handles)

```python
@dataclass
class Pipeline:
    steps: list[str]
    params: dict
    
    def to_dict(self): return asdict(self)
    
    @classmethod
    def from_dict(cls, d): return cls(**d)
```

---

### 7️⃣ Errors Must Be Actionable
✅ **What failed**: Parameter, operation, context  
✅ **Why**: What was wrong with input/state  
✅ **How to fix**: Specific suggestion or example  
❌ Vague (`"Error"`, `"Something went wrong"`)

```python
raise ValueError(
    f"wavelength_end ({v}) must be > wavelength_start ({start}).\n"
    f"Fix: Ensure end > start, or swap if needed."
)
```

---

## Daily Workflow

### Before Coding
```bash
# Read the rules (5 min)
cat docs/developer-guide/ENGINEERING_RULES.md

# Or the quick reference
cat docs/developer-guide/QUICK_REFERENCE.md
```

### While Coding
```bash
# Check for issues as you go
ruff check src/ --fix
ruff format src/
```

### Before Committing
```bash
# Run all checks locally
ruff format src/ tests/
ruff check src/ tests/
mypy src/ --strict
pytest tests/ --cov=src/foodspec --cov-fail-under=80
```

### Before PR
Use the **PR Checklist** from [CONTRIBUTING.md](../../CONTRIBUTING.md)

---

## Tool Commands Cheat Sheet

| Task | Command |
|------|---------|
| Format code | `ruff format src/ tests/` |
| Lint | `ruff check src/ tests/ --fix` |
| Type check | `mypy src/ --strict` |
| Run tests | `pytest tests/ -v` |
| Coverage | `pytest --cov=src/foodspec` |
| Coverage HTML | `pytest --cov=src/foodspec --cov-report=html` |
| Single test | `pytest tests/test_foo.py::TestClass::test_method -v` |
| Tests + strict coverage | `pytest --cov=src/foodspec --cov-fail-under=80` |
| Check warnings | `pytest -W error::DeprecationWarning` |
| Pre-commit | `pre-commit run --all-files` |

---

## Example: Adding a New Function

Follow this template:

```python
# ✅ In src/foodspec/my_module.py

def my_new_function(data: np.ndarray, 
                    threshold: float = 0.5,
                    seed: int | None = None) -> dict:
    """One-line summary.
    
    Longer description explaining what it does and why.
    
    Parameters
    ----------
    data : np.ndarray
        Input array.
    threshold : float
        Threshold for filtering.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Result with keys 'output', 'quality'.
    
    Raises
    ------
    ValueError
        If threshold is invalid.
    TypeError
        If data is not array-like.
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> result = my_new_function(data, threshold=1.5, seed=42)
    >>> 'output' in result
    True
    """
    
    # Validate inputs early
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be np.ndarray, got {type(data)}")
    if threshold < 0 or threshold > 1:
        raise ValueError(
            f"threshold must be in [0, 1], got {threshold}.\n"
            f"Fix: Use 0 ≤ threshold ≤ 1."
        )
    
    # Use seed for reproducibility
    rng = np.random.default_rng(seed)
    
    # Implementation...
    output = data[data > threshold]
    quality = rng.uniform(0, 1)
    
    return {'output': output, 'quality': quality}
```

```python
# ✅ In tests/test_my_module.py

import pytest
import numpy as np
from foodspec.my_module import my_new_function

class TestMyNewFunction:
    @pytest.fixture
    def sample_data(self):
        return np.array([0.1, 0.5, 0.9, 1.5, 2.0])
    
    def test_basic_functionality(self, sample_data):
        result = my_new_function(sample_data, threshold=0.5)
        assert 'output' in result
        assert 'quality' in result
    
    def test_deterministic_with_seed(self, sample_data):
        r1 = my_new_function(sample_data, seed=42)
        r2 = my_new_function(sample_data, seed=42)
        assert r1['quality'] == r2['quality']
    
    def test_invalid_threshold_raises(self, sample_data):
        with pytest.raises(ValueError, match="threshold.*\\[0, 1\\]"):
            my_new_function(sample_data, threshold=1.5)
    
    def test_invalid_data_raises(self):
        with pytest.raises(TypeError, match="data must be"):
            my_new_function([1, 2, 3], threshold=0.5)
```

---

## When to Ask for Help

- 🤔 **Unsure about a rule?** Read [ENGINEERING_RULES.md](./ENGINEERING_RULES.md) or ask in issue
- 🔄 **Breaking an existing API?** Discuss in issue first; see [COMPATIBILITY_PLAN.md](./COMPATIBILITY_PLAN.md)
- 🐛 **Edge case for error messages?** See [Rule 7 examples](./ENGINEERING_RULES.md#rule-7-errors-must-be-actionable)
- 📚 **Need backward compat example?** See [BACKWARD_COMPAT_EXAMPLES.md](./BACKWARD_COMPAT_EXAMPLES.md)

---

## Links

- 📖 [Full Engineering Rules](./ENGINEERING_RULES.md)
- 🤝 [Contributing Guide](../../CONTRIBUTING.md)
- 🔄 [Compatibility Plan](./COMPATIBILITY_PLAN.md)
- 💡 [Backward Compat Examples](./BACKWARD_COMPAT_EXAMPLES.md)

---

**Updated**: 2026-01-24  
**Print & Share!** 🚀
