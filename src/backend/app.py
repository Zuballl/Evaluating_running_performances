from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.backend.fit_parser import parse_fit_metrics
from src.data.read_data import PROCESSED_DATA_PATH
from src.xai.pca_xai import PCAExplainer, explain_sample, fit_pca_explainer


class PredictRequest(BaseModel):
    total_distance: float = Field(..., ge=0)
    elevation_gain: float
    average_hr: float = Field(..., ge=0)
    aerobic_decoupling: float
    athlete_weight: float = Field(..., ge=0)
    final_cadence: float = Field(..., ge=0)
    pace_min_km: float = Field(..., ge=0)
    age: int


class ContributionItem(BaseModel):
    feature: str
    contribution: float
    contribution_pct_abs: float


class PredictResponse(BaseModel):
    score: float
    raw_score: float
    percentile: float
    top_contributions: list[ContributionItem]


app = FastAPI(title="Running Score API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_explainer(
    clean_data_path: str = str(PROCESSED_DATA_PATH),
    preprocessing: str = "standard_clip",
) -> PCAExplainer:
    path = Path(clean_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Clean data not found: {path}")
    df = pd.read_csv(path)
    return fit_pca_explainer(df, preprocessing=preprocessing)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict-json", response_model=PredictResponse)
def predict_json(
    payload: PredictRequest,
    top_k: int = 5,
    preprocessing: str = "standard_clip",
) -> PredictResponse:
    return _predict_from_features(payload.model_dump(), top_k, preprocessing)


@app.post("/predict-fit", response_model=PredictResponse)
async def predict_fit(
    fit_file: UploadFile = File(...),
    top_k: int = 5,
    preprocessing: str = "standard_clip",
    age: int | None = Form(None),
    athlete_weight: float | None = Form(None),
) -> PredictResponse:
    if not fit_file.filename:
        raise HTTPException(status_code=400, detail="fit_file must have a filename")

    content = await fit_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="fit_file is empty")

    parsed = parse_fit_metrics(content)
    features: dict[str, Any] = {
        "total_distance": parsed.total_distance,
        "elevation_gain": parsed.elevation_gain,
        "average_hr": parsed.average_hr,
        "aerobic_decoupling": parsed.aerobic_decoupling,
        "athlete_weight": parsed.athlete_weight if parsed.athlete_weight is not None else athlete_weight,
        "final_cadence": parsed.final_cadence,
        "pace_min_km": parsed.pace_min_km,
        "age": parsed.age if parsed.age is not None else age,
    }

    missing = [k for k, v in features.items() if v is None]
    if missing:
        _raise_missing_fields_error(missing)

    return _predict_from_features(features, top_k, preprocessing)


def _predict_from_features(features: dict[str, Any], top_k: int, preprocessing: str) -> PredictResponse:
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be > 0")

    try:
        payload = PredictRequest(**features)
        explainer = get_explainer(preprocessing=preprocessing)
        result = explain_sample(explainer, payload.model_dump())
        top_df = result.contributions[["feature", "contribution", "contribution_pct_abs"]].head(top_k)
        top_contributions = [ContributionItem(**row) for row in top_df.to_dict(orient="records")]
        return PredictResponse(
            score=result.score,
            raw_score=result.raw_score,
            percentile=result.percentile,
            top_contributions=top_contributions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _raise_missing_fields_error(missing_fields: list[str]) -> None:
    profile_fields = {"age", "athlete_weight"}
    missing_profile = [field for field in missing_fields if field in profile_fields]
    missing_metrics = [field for field in missing_fields if field not in profile_fields]

    if missing_metrics:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "missing_fit_metrics",
                "message": "FIT file is missing required activity metrics.",
                "missing_fields": missing_metrics,
            },
        )

    raise HTTPException(
        status_code=422,
        detail={
            "code": "missing_profile_fields",
            "message": "Provide missing profile fields before scoring.",
            "missing_fields": missing_profile,
            "prompt": "Please provide age and weight if missing.",
        },
    )
