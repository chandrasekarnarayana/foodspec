# foodspec

foodspec is a headless, research-grade Python toolkit for Raman and FTIR spectroscopy in food science. It provides unified data models for 1D spectra and hyperspectral cubes, reproducible preprocessing pipelines, feature extraction, chemometrics, and turnkey workflows such as oil authentication, heating degradation, and mixture modeling.

## Who is it for?
- Food scientists and analytical chemists working with Raman/FTIR/NIR data.
- Data scientists who want a clean, sklearn-style API for spectral analysis.
- Researchers who care about reproducible, FAIR-compliant workflows aligned with a MethodsX-style protocol.

## What it solves
- Consistent handling of spectra, metadata, and modalities via `FoodSpectrumSet` and `HyperSpectralCube`.
- Pipeline-ready preprocessing (baseline, smoothing, scatter correction, normalization, FTIR/Raman helpers).
- Chemometrics and ML (PCA/PLS, classifier factory, mixture analysis, QC/novelty detection).
- Turnkey workflows (oil authentication, heating degradation, domain templates) with CLI + Python entry points.
- Reporting, logging, configs, and spectral libraries for reproducible runs.

## Quick start
```bash
pip install foodspec
foodspec about
```

## Acknowledgements

The development of **foodspec** draws upon the collective strength of interdisciplinary research at the intersection of food science, spectroscopy, physics, and machine learning. Its evolution has been enriched by the generous guidance, scientific exchange, and collaborative spirit of several individuals.

### Collaborators :

- **Dr. Jhinuk Gupta**  
  *Department of Food and Nutritional Sciences, Sri Sathya Sai Institute of Higher Learning (SSSIHL), Andhra Pradesh, India*  
  LinkedIn: https://www.linkedin.com/in/dr-jhinuk-gupta-a7070141/  
  

- **Dr. Sai Muthukumar V**  
  *Department of Physics, SSSIHL, Andhra Pradesh, India*  
  LinkedIn: https://www.linkedin.com/in/sai-muthukumar-v-ab78941b/  
  

- **Ms. Amrita Shaw**  
  *Department of Food and Nutritional Sciences, SSSIHL, Andhra Pradesh, India*  
  LinkedIn: https://www.linkedin.com/in/amrita-shaw-246491213/  
  

- **Deepak L. N. Kallepalli**  
  *Cognievolve AI Inc., Canada & HCL Technologies Ltd., Bangalore, India*  
  LinkedIn: https://www.linkedin.com/in/deepak-kallepalli/  
  

### Author

- **Chandrasekar SUBRAMANI NARAYANA**  
  *Aix-Marseille University, Marseille, France*  
  LinkedIn: https://www.linkedin.com/in/snchandrasekar/  
  

## Where to start
- [Getting started](getting_started.md) – installation, basic examples, how to load data.  
- [Libraries](libraries.md) – building and loading spectral libraries, public dataset loaders.  
- [Validation & chemometrics](validation_chemometrics_oils.md) – PCA and oil-authentication workflows.  
- [MethodsX protocol](methodsx_protocol.md) – mapping between foodspec commands and the MethodsX article.  
- [Citing foodspec](citing.md) – how to cite the software and the protocol paper.
 - [Instrument & file formats](user_guide/instrument_file_formats.md) – how to load CSV/JCAMP/SPC/OPUS exports into FoodSpec.

## Project Intention & Philosophy
- **Core principles**: open, clear, reproducible workflows; strong scientific grounding (physics + chemistry + statistics + ML); real-world usability in labs and QA/QC; interpretability and transparency; accessible to beginners and experts.
- **Overall vision**: a reference toolkit, teaching resource, shared computational language across interdisciplinary teams, and a long-term foundation for reproducible food-science analytics.
