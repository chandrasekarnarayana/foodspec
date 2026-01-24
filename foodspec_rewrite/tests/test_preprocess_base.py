"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.
"""

import numpy as np

from foodspec.preprocess.base import IdentityTransformer


def test_identity_transformer_noop() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = IdentityTransformer()

    out = t.fit_transform(X)
    assert np.array_equal(out, X)
    assert t.get_params() == {}

    # set_params should be chainable and no-op
    t2 = t.set_params(foo=1)
    assert t2 is t
