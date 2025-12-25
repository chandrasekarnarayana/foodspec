# Domains overview

Questions this page answers
- How do domain templates relate to core workflows?
- Which template should I use for oils, meat, or microbial data?
- How do I run them via CLI/Python?

## What are domain templates?
Domain templates are thin wrappers around the core workflows (oil-auth, heating, QC) pre-configured for specific food types (oils, meat, microbial, etc.). They reuse the same preprocessing, feature extraction, and chemometrics to keep results consistent and reproducible.

## When to use templates vs raw workflows
- Use a **template** when your task matches a provided domain (e.g., meat authentication with a label column) and you want sensible defaults with minimal setup.
- Use **raw workflows** (oil-auth/heating/qc) when you need full control over preprocessing, labels, or model choices beyond the templates.

## Domain quick reference
| Domain   | Typical task                  | Recommended workflow         | Example CLI command |
| ---      | ---                           | ---                          | ---                 |
| Oils     | Multi-class authentication    | Oil-auth workflow            | `foodspec oil-auth libraries/oils.h5 --label-column oil_type` |
| Meat     | Multi-class meat type QC/auth | Domains (meat template)      | `foodspec domains libraries/meat.h5 --type meat --label-column meat_type` |
| Microbial| Species/strain identification | Domains (microbial template) | `foodspec domains libraries/microbial.h5 --type microbial --label-column species` |

See also
- `meat_tutorial.md`
- `microbial_tutorial.md`
- `oil_auth_tutorial.md`
- `metrics_interpretation.md`
