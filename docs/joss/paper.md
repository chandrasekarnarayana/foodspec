---
title: "FoodSpec: Protocol-driven spectroscopy workflows for food matrices"
tags:
  - spectroscopy
  - chemometrics
  - food science
  - raman
  - ftir
authors:
  - name: FoodSpec Contributors
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: FoodSpec Project
    index: 1
date: 2024-01-01
bibliography: references.bib
---

# Summary

FoodSpec is a Python toolkit for protocol-driven Raman and FTIR pipelines focused on food matrices.
It provides reproducible preprocessing, quality control, feature extraction, and modeling workflows,
with explicit trust and uncertainty artifacts (calibration outputs, conformal coverage, and cards).

# Statement of Need

Food spectroscopy workflows often mix preprocessing, chemometrics, and modeling in ad hoc scripts.
FoodSpec provides a structured, protocol-based system that enforces data validation, QC, and
run artifacts so that results are traceable and comparable across laboratories and instruments.

# Design

FoodSpec centers on explicit data objects, modular preprocessing steps, a quality system, and
standardized run artifacts. The CLI surfaces mindmap-aligned commands (io, preprocess, qc,
features, model, trust, report) and produces manifests, summaries, and logs for every run.

# Acknowledgements

We acknowledge the FoodSpec community and open-source scientific Python ecosystem.
