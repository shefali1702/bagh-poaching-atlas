"""Step 4 - export everything the dashboard needs as small JSON files in web/data/.

  web/data/app.json          reserves, metrics, SHAP, map outlines, provenance
  web/data/grid/<id>.json    1 km cells of one reserve (scores, classes, reasons, key features)

Run:  python -m src.export_web
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping

from .config import PROC, RAW, REPORTS, MODELS, WEB_DATA, ROOT
from .features import PRETTY, REAL_FEATURES, NUMERIC

LON0, LAT1, K = 68.0, 37.5, 35.95
COSL = np.cos(np.radians(22))
VIEW_W, VIEW_H = 1000, 1135


def proj(lon, lat):
    return (lon - LON0) * K * COSL, (LAT1 - lat) * K


def path_of(geom, tol):
    g = geom.simplify(tol)
    polys = getattr(g, "geoms", [g])
    out = []
    for p in polys:
        if p.geom_type != "Polygon" or p.area < tol * tol * 4:
            continue
        for ring in [p.exterior] + list(p.interiors):
            pts = [proj(x, y) for x, y in ring.coords]
            out.append("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + "Z")
    return "".join(out)


def r1(v):
    return [round(float(x), 1) for x in v]


def main():
    (WEB_DATA / "grid").mkdir(parents=True, exist_ok=True)
    cells = pd.read_parquet(PROC / "grid_cells.parquet")
    pred = pd.read_parquet(PROC / "predictions.parquet")
    metrics = json.load(open(REPORTS / "metrics.json"))
    feat_order = json.load(open(MODELS / "meta.json"))["feature_order"]
    fidx = {f: i for i, f in enumerate(feat_order)}
    df = cells.merge(pred.drop(columns=["reserve_id", "x", "y", "lat", "lon"]), on=["cell_id", "season"])
    test = set(metrics["split"]["test_reserves"])

    # ---- map outlines
    countries = gpd.read_file(RAW / "ne_10m_admin_0_countries.geojson")
    india = countries[countries.ADM0_A3 == "IND"].geometry.buffer(0).union_all()
    states = gpd.read_file(RAW / "ne_10m_admin_1_states_provinces.geojson")
    states = states[states.adm0_a3 == "IND"]
    from shapely.geometry import box
    frame = box(LON0, 5.5, 98.5, LAT1)
    neigh = countries[(countries.ADM0_A3 != "IND") & countries.intersects(frame)]
    fps = gpd.read_file(PROC / "reserve_footprints.geojson")
    mapd = {"w": VIEW_W, "h": VIEW_H, "india": path_of(india, 0.04),
            "states": [path_of(g.buffer(0), 0.05) for g in states.geometry],
            "neighbours": path_of(neigh.geometry.buffer(0).union_all().intersection(frame), 0.08),
            "footprints": {int(r.reserve_id): path_of(r.geometry, 0.004) for r in fps.itertuples()}}

    # ---- per reserve
    gb = pd.read_csv(RAW / "gbif_india_occurrences.csv")
    reserves = []
    for rid, g in df.groupby("reserve_id"):
        d, w = g[g.season == "dry"].sort_values("cell_id"), g[g.season == "wet"].sort_values("cell_id")
        minx, maxy = d.x.min(), d.y.max()
        col = ((d.x - minx) / 1000).round().astype(int).values
        row = ((maxy - d.y) / 1000).round().astype(int).values

        def codes(s):
            return [int((fidx[s[f"r{j}_feat"]] + 1) * (1 if s[f"r{j}_val"] >= 0 else -1)) for j in range(3)]
        cls = {"Low": 0, "Medium": 1, "High": 2}
        grid = {
            "id": int(rid), "w": int(col.max() + 1), "h": int(row.max() + 1),
            "lat0": round(float(d.lat.iloc[0]), 4), "lon0": round(float(d.lon.iloc[0]), 4),
            "col": col.tolist(), "row": row.tolist(),
            "lat": [round(float(v), 4) for v in d.lat], "lon": [round(float(v), 4) for v in d.lon],
            "score": {"dry": r1(d.pred_score), "wet": r1(w.pred_score)},
            "truth": {"dry": r1(d.threat_score), "wet": r1(w.threat_score)},
            "cls": {"dry": d.pred_class.map(cls).tolist(), "wet": w.pred_class.map(cls).tolist()},
            "why": {"dry": [codes(s) for _, s in d.iterrows()], "wet": [codes(s) for _, s in w.iterrows()]},
            "why_val": {"dry": [[round(float(s[f"r{j}_val"]), 1) for j in range(3)] for _, s in d.iterrows()],
                        "wet": [[round(float(s[f"r{j}_val"]), 1) for j in range(3)] for _, s in w.iterrows()]},
            "feat": {f: r1(d[f]) for f in ["dist_to_boundary_km", "dist_to_road_km", "dist_to_town_km",
                                           "dist_to_water_km", "dist_to_border_km", "dist_to_ranger_km",
                                           "fragmentation_index", "forest_cover", "target_species_records_10km",
                                           "slope_deg"]},
            "land_cover": d.land_cover.tolist(),
        }
        json.dump(grid, open(WEB_DATA / "grid" / f"{rid}.json", "w"), separators=(",", ":"))
        first = g.iloc[0]
        hi_d = d[d.pred_class == "High"]
        top = hi_d.r0_feat.value_counts(normalize=True).head(3) if len(hi_d) else pd.Series(dtype=float)
        cx, cy = proj(d.lon.mean(), d.lat.mean())
        reserves.append({
            "id": int(rid), "name": first["name"], "state": first.state, "species": first.species,
            "landscape": first.landscape, "cells": int(len(d)), "x": round(cx, 1), "y": round(cy, 1),
            "lat": round(float(d.lat.mean()), 3), "lon": round(float(d.lon.mean()), 3),
            "mean": {"dry": round(float(d.pred_score.mean()), 1), "wet": round(float(w.pred_score.mean()), 1)},
            "high_pct": {"dry": round(float((d.pred_class == "High").mean() * 100), 1),
                         "wet": round(float((w.pred_class == "High").mean() * 100), 1)},
            "top_drivers": [[k, round(float(v), 2)] for k, v in top.items()],
            "fragmentation": round(float(d.fragmentation_index.mean()), 3),
            "edge_share": round(float((d.dist_to_boundary_km < 2).mean()), 3),
            "test": first["name"] in test,
        })

    # ---- national figures
    dry = df[df.season == "dry"]; wet = df[df.season == "wet"]
    edge = dry.dist_to_boundary_km < 2
    nat = {
        "cells": int(len(dry)), "reserves": len(reserves), "area_km2": int(len(dry)),
        "high_pct_dry": round(float((dry.pred_class == "High").mean() * 100), 1),
        "high_pct_wet": round(float((wet.pred_class == "High").mean() * 100), 1),
        "high_share_near_edge": round(float((dry[dry.pred_class == "High"].dist_to_boundary_km < 2).mean() * 100), 1),
        "edge_area_share": round(float(edge.mean() * 100), 1),
        "gbif": gb.species.value_counts().to_dict(), "gbif_total": int(len(gb)),
    }
    m = metrics
    for blk in m["regression"]:
        blk.pop("scatter", None) if blk["model"] != "XGBoost Regressor" else None
    app = {"map": mapd, "reserves": reserves, "national": nat, "metrics": m, "features": feat_order,
           "pretty": PRETTY, "real": sorted(REAL_FEATURES),
           "thresholds": json.load(open(PROC / "label_thresholds.json"))}
    json.dump(app, open(WEB_DATA / "app.json", "w"), separators=(",", ":"))
    print("reserves", len(reserves), "app.json KB", round((WEB_DATA / "app.json").stat().st_size / 1024))


if __name__ == "__main__":
    main()
