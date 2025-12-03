# Workflow: Domain Templates (Meat, Microbial, and Beyond)

This chapter explains how FoodSpec’s domain templates reuse core workflows for specific food types (e.g., meat, microbial ID) with sensible defaults. It connects domain pages to the underlying oil-style pipeline.

## What this chapter covers
- How domain templates map to the oil-auth style pipeline (preprocessing + classifier).
- Typical metadata/label expectations per domain (meat_type, species/strain, etc.).
- When to use a domain template vs configure your own workflow.
- Links to meat/microbial tutorial pages for runnable examples.

## Outline
- **Template concept:** Thin wrappers around preprocessing + classification; default features/models.
- **Meat:** Raman/FTIR use cases; label expectations; adapting oil defaults.
- **Microbial:** Spectral IDs; class imbalance considerations; QC steps.
- **Dairy/adulteration (future):** Apply the same preprocessing/ratios/PCA + classifier pattern; record instrument (FTIR/NIR), matrix (milk powders/liquids), target labels (adulterant level/type); reuse reproducibility fields for plots/reports.
- **Spices/grains (future):** Heterogeneous matrices; emphasize preprocessing choices (baseline, normalization), feature selection (key bands), and QC/statistics similar to oil workflows.
- **Extensibility:** Adding new domain templates; using CLI `domains` command (if applicable).
- **Pointers:** See `../meat_tutorial.md` and `../microbial_tutorial.md` for code/CLI recipes.

## Next steps
- Use a template for rapid prototyping; switch to custom pipelines for specialized datasets.
- Explore **Protocols & reproducibility** to document template use in studies.
