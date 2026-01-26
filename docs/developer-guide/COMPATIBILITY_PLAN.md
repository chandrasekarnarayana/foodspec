# Compatibility Plan

This document defines how FoodSpec maintains backward compatibility as the mindmap
architecture becomes the primary source of truth.

## Principles
- Public APIs remain stable across minor releases.
- Deprecated imports remain available for one full version cycle.
- Deprecations emit `DeprecationWarning` with clear alternatives.

## Deprecation policy
1. Introduce new module path.
2. Keep old path as a shim that re-exports the new implementation.
3. Warn at import or call time.
4. Remove after one major version boundary.

## Communication
- Add a CHANGELOG entry.
- Update docs and examples to the new path.

