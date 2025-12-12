# Advanced Topic – Web API & Services

FoodSpec can be served as a web microservice for prediction and diagnostics.

## FastAPI backend
- Endpoints:
  - `/models`: list available frozen models.
  - `/predict`: score uploaded spectra/metadata using a selected model (applies preprocessing + feature extraction).
  - `/upload`: store datasets for later use.
  - `/diagnostics`: return harmonization/alignment diagnostics.
- Deployment: containerize with Docker; host on cloud (Heroku/AWS/GCP) or on-prem behind LIMS/instrument PCs. Add auth/SSL as required.

## SPA frontend
- React/Vue scaffold (see `webapp/frontend`) with drag/drop upload, model selection, and result/plot display suitable for lab technicians.

## Integration with bundles/models
- Service loads `FrozenModel` packages from bundle paths; honors preprocessing/harmonization configs embedded in the model.
- Diagnostics endpoints can visualize pre/post alignment when harmonization is enabled.

Refer to `webapp/README.md` for setup and to the CLI guide for model packaging workflows.
