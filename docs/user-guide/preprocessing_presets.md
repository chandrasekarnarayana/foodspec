# Preprocessing Presets

FoodSpec ships with a small set of preprocessing presets to reduce setup time.
Presets are YAML recipes loaded with `load_preset_yaml()` or used via `load_recipe()`.

Available presets:

- default
- raman
- ftir
- oil_auth
- chips_matrix
- dairy
- meat
- fruit
- grain

Example

```python
from foodspec.preprocess import load_preset_yaml, load_recipe

preset = load_preset_yaml("dairy")
pipeline = load_recipe(preset="dairy")
```

Notes

- Presets are conservative defaults. Tune parameters for your instrument and matrix.
- Use `load_recipe()` to merge protocol overrides with presets.
