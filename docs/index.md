# foodspec

foodspec is a headless, research-grade Python toolkit for Raman and FTIR
spectroscopy in food science. It provides a unified data model for 1D spectra
and hyperspectral cubes, reproducible preprocessing pipelines, feature
extraction, chemometrics, and domain-specific workflows such as oil
authentication and heating degradation analysis.

The library is designed for:

- food scientists and analytical chemists who work with Raman/FTIR data,
- data scientists who want a clean, sklearn-style API for spectral analysis,
- researchers who care about reproducible, FAIR-compliant workflows.

foodspec implements the computational protocol that underpins the planned
MethodsX article on FAIR-compliant Raman/FTIR analysis in food science. All
major analyses can be reproduced using public datasets, configuration files,
and standardized CLI commands.

## Where to start

- [Getting started](getting_started.md) – installation, basic examples, how to load data.  
- [Libraries](libraries.md) – building and loading spectral libraries, public dataset loaders.  
- [Validation & chemometrics](validation_chemometrics_oils.md) – PCA and oil-authentication workflows.  
- [MethodsX protocol](methodsx_protocol.md) – mapping between foodspec commands and the MethodsX article.  
- [Citing foodspec](citing.md) – how to cite the software and the protocol paper.
