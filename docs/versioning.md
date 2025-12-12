## Versioning Policy

- FoodSpec uses semantic versioning: MAJOR.MINOR.PATCH.
  - MAJOR: breaking changes.
  - MINOR: new features, backwards compatible.
  - PATCH: bugfixes.
- Protocol versions:
  - Each protocol has its own `version` and may declare `min_foodspec_version`.
  - Ensure your FoodSpec version meets or exceeds the protocol’s `min_foodspec_version`.
- Recommendation:
  - Pin protocol + FoodSpec versions for reproducibility in manuscripts and registry entries.
