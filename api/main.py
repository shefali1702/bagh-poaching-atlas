"""FastAPI service for the Poaching Risk Predictor.

  GET  /api/health
  GET  /api/reserves
  GET  /api/predict?lat=..&lon=..&season=dry|wet     snap to the nearest 1 km cell and explain it
  POST /api/predict/features                         what-if: score any feature vector you send
  /                                                  the dashboard (web/)

Run:  uvicorn api.main:app --reload --port 8000
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import CLASSES, PRETTY, X_of, add_flags, base_feature  # noqa: E402

app = FastAPI(title="Poaching Risk Predictor", version="1.0",
              description="Predicts poaching risk for 1 km cells in Indian tiger, elephant and rhino reserves.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@lru_cache
def assets():
    clf = joblib.load(ROOT / "models" / "xgb_clf.joblib")
    reg = joblib.load(ROOT / "models" / "xgb_reg.joblib")
    cells = add_flags(pd.read_parquet(ROOT / "data" / "processed" / "grid_cells.parquet"))
    lat0 = np.radians(22)
    xy = np.c_[cells.lon.values * np.cos(lat0) * 111.32, cells.lat.values * 110.57]
    return clf, reg, cells, cKDTree(xy), lat0


def explain(reg, X: pd.DataFrame, top=5):
    prep, booster = reg.named_steps["prep"], reg.named_steps["m"].get_booster()
    names = list(prep.get_feature_names_out())
    contrib = booster.predict(xgb.DMatrix(prep.transform(X), feature_names=names), pred_contribs=True)[0]
    s = pd.Series(contrib[:-1], index=[base_feature(n) for n in names]).groupby(level=0).sum()
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    return {"base_value": round(float(contrib[-1]), 2),
            "reasons": [{"feature": f, "label": PRETTY.get(f, f), "value": X.iloc[0][f] if f in X else None,
                         "impact": round(float(v), 2), "direction": "raises risk" if v > 0 else "lowers risk"}
                        for f, v in s.head(top).items()]}


def score(X: pd.DataFrame):
    clf, reg, *_ = assets()
    proba = clf.predict_proba(X)[0]
    out = {"risk_class": CLASSES[int(proba.argmax())],
           "probabilities": {c: round(float(p), 3) for c, p in zip(CLASSES, proba)},
           "threat_score": round(float(np.clip(reg.predict(X)[0], 0, 100)), 1)}
    ex = explain(reg, X)
    for r in ex["reasons"]:
        v = r["value"]
        r["value"] = v.item() if hasattr(v, "item") else v
    out["explanation"] = ex
    return out


@app.get("/api/health")
def health():
    _, _, cells, *_ = assets()
    return {"status": "ok", "cells": int(cells.cell_id.nunique()), "reserves": int(cells.reserve_id.nunique())}


@app.get("/api/reserves")
def reserves():
    _, _, cells, *_ = assets()
    g = cells[cells.season == "dry"].groupby(["reserve_id", "name", "state", "species"])
    return [{"id": int(k[0]), "name": k[1], "state": k[2], "species": k[3], "cells": int(len(v)),
             "lat": round(float(v.lat.mean()), 3), "lon": round(float(v.lon.mean()), 3)} for k, v in g]


@app.get("/api/predict")
def predict(lat: float = Query(..., ge=6, le=37), lon: float = Query(..., ge=68, le=98),
            season: Literal["dry", "wet"] = "dry"):
    _, _, cells, tree, lat0 = assets()
    d, i = tree.query([lon * np.cos(lat0) * 111.32, lat * 110.57], k=8)
    cand = cells.iloc[i]
    cand = cand[cand.season == season]
    if cand.empty or d.min() > 3:
        raise HTTPException(404, "That point is outside the 54 modelled reserves (more than 3 km from any cell).")
    row = cand.iloc[[0]]
    res = score(X_of(row))
    res.update({"cell_id": row.cell_id.iloc[0], "reserve": row.name.iloc[0], "state": row.state.iloc[0],
                "season": season, "snap_distance_km": round(float(d.min()), 2),
                "cell_lat": float(row.lat.iloc[0]), "cell_lon": float(row.lon.iloc[0])})
    return res


class Features(BaseModel):
    dist_to_boundary_km: float = Field(1.0, ge=0)
    dist_to_road_km: float = Field(5.0, ge=0)
    dist_to_town_km: float = Field(30.0, ge=0)
    dist_to_water_km: float = Field(2.0, ge=0)
    dist_to_border_km: float = Field(150.0, ge=0)
    dist_to_ranger_km: float = Field(8.0, ge=0)
    species_records_10km: float = Field(20, ge=0)
    target_species_records_10km: float = Field(15, ge=0)
    elevation_m: float = 500
    slope_deg: float = Field(6, ge=0, le=60)
    ndvi: float = Field(0.5, ge=-1, le=1)
    forest_cover: float = Field(0.6, ge=0, le=1)
    fragmentation_index: float = Field(0.4, ge=0, le=1)
    reserve_pressure: float = Field(0.4, ge=0, le=1)
    land_cover: Literal["dense_forest", "open_forest", "grassland", "scrub", "wetland", "degraded"] = "open_forest"
    season: Literal["dry", "wet"] = "dry"
    species: str = Field("TE", description="Letters for species present: T tiger, E elephant, R rhino")


@app.post("/api/predict/features")
def predict_features(f: Features):
    return score(X_of(pd.DataFrame([f.model_dump()])))


WEB = ROOT / "web"
if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
