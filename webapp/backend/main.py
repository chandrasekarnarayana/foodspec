"""
FoodSpec Prediction Service (FastAPI)
-------------------------------------
- /models: list available frozen models (prefix paths)
- /predict: apply a frozen model to uploaded CSV spectra/peaks
- /diagnostics: return simple harmonization diagnostics (placeholder)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from foodspec.model_lifecycle import FrozenModel

MODELS_DIR = Path(os.environ.get("FOODSPEC_MODELS_DIR", "models"))
MODELS_DIR.mkdir(exist_ok=True, parents=True)

app = FastAPI(title="FoodSpec Prediction Service", version="0.1.0")


class ModelInfo(BaseModel):
    name: str
    path: str


class PredictionResult(BaseModel):
    model: str
    n_samples: int
    predictions: List[str]


@app.get("/models", response_model=List[ModelInfo])
def list_models():
    out: List[ModelInfo] = []
    for json_file in MODELS_DIR.glob("*.json"):
        out.append(ModelInfo(name=json_file.stem, path=str(json_file)))
    return out


@app.post("/predict", response_model=PredictionResult)
async def predict(model_name: str = Form(...), file: UploadFile = File(...)):
    model_prefix = MODELS_DIR / model_name
    json_path = model_prefix.with_suffix(".json")
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    contents = await file.read()
    try:
        from io import BytesIO

        df = pd.read_csv(BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}")
    mdl = FrozenModel.load(model_prefix)
    preds_df = mdl.predict(df)
    return PredictionResult(model=model_name, n_samples=len(preds_df), predictions=preds_df["prediction"].astype(str).tolist())


@app.get("/diagnostics")
def diagnostics():
    """
    Return simple diagnostics for the demo service.
    In a full deployment, add harmonization metrics/plots here.
    """
    return {
        "status": "ok",
        "message": "Diagnostics available: none in demo mode. Add harmonization metrics/plots here.",
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Simply acknowledge upload in demo mode
    await file.read()
    return JSONResponse({"status": "received", "filename": file.filename})
