# Endangered Species Poaching Risk Predictor (India)

**Live site: https://bagh-poaching-atlas.vercel.app**

Contributors: [Shefali Sharma](https://github.com/shefali1702), [Bhavyam Misra](https://github.com/bhavyammisra)

Predicts poaching risk for every 1 km² cell of **54 Indian tiger, elephant and rhino reserves**, so forest
departments can send limited patrol teams where they are needed most. Built by Shefali Sharma and Bhavyam Misra.

- **Classification:** risk level (High / Medium / Low) per cell
- **Regression:** continuous threat score (0–100) per cell
- **Explainability:** TreeSHAP reasons for every prediction
- **Deployment:** FastAPI service + custom animated dashboard (plus a simple Streamlit view)

## Results (12 whole reserves held out, never seen in training)

| Classification | Macro F1 | High recall | PR-AUC (High) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.764 | 0.781 | 0.836 | 0.924 |
| Decision Tree | 0.687 | 0.699 | 0.704 | 0.849 |
| Random Forest (Optuna) | 0.751 | 0.745 | 0.803 | 0.919 |
| **XGBoost (Optuna)** | **0.776** | **0.773** | **0.844** | **0.934** |

| Regression | RMSE | MAE | R² |
|---|---|---|---|
| Multiple Linear Regression | 12.39 | 9.94 | 0.772 |
| Random Forest Regressor | 12.29 | 9.74 | 0.775 |
| **XGBoost Regressor (Optuna)** | **11.57** | **9.04** | **0.801** |

5-fold grouped cross-validation over all 54 reserves: macro F1 0.79, high-risk recall 0.81, R² 0.83.
Top SHAP drivers: distance to reserve edge, habitat fragmentation, distance to the international border,
the reserve's past pressure, distance to road, and target-species sightings.

## Quick start

```bash
pip install -r requirements.txt
uvicorn api.main:app --port 8000        # opens the dashboard at http://localhost:8000
# or
streamlit run streamlit_app.py
```

Everything is already built (data, models, dashboard data). To rebuild from scratch: `./run_all.sh`.

## Project layout

```
data/reserves_india.csv     54 reserves: state, centre, official area, species, landscape
data/raw/                   Natural Earth vectors + GBIF sightings (downloaded by fetch_data)
data/processed/             grid_cells.parquet (143k rows), predictions.parquet, footprints
src/fetch_data.py           1. download Natural Earth + GBIF
src/build_dataset.py        2. 1 km grid, features, proxy labels
src/train.py                3. LR / DT / RF / XGB, Optuna, spatial split, metrics, SHAP, out-of-fold maps
src/make_art.py             halftone art for the dashboard from the reference photos
src/export_web.py           4. JSON for the dashboard
src/ntca_scraper.py         optional: real NTCA tiger mortality → replace simulated reserve pressure
api/main.py                 FastAPI: /api/predict, /api/predict/features, /api/reserves
web/                        the dashboard (plain HTML/JS, no build step)
reports/metrics.json        every number above, plus curves
reports/label_rule.md       the documented proxy-label rule
```

## API

```
GET  /api/predict?lat=26.62&lon=93.35&season=dry
     → risk_class, probabilities, threat_score, top-5 SHAP reasons, snapped cell
POST /api/predict/features   {"dist_to_boundary_km": 0.3, "dist_to_road_km": 1.2, ...}
     → what-if scoring of any feature vector
```
Interactive docs at http://localhost:8000/docs.

## Data: what is real and what is not

| Layer | Source | Status |
|---|---|---|
| Roads, towns, rivers, lakes, borders | Natural Earth 1:10m | Real |
| Tiger, elephant, rhino, leopard sightings | GBIF API (first 3,000 per species, India) | Real |
| Reserve list, centres, official areas | NTCA / state forest departments | Real, rounded |
| Reserve shapes | Footprint generated at the official area | Approximate |
| Elevation, slope, NDVI, forest cover, fragmentation | Placeholders for SRTM / MODIS / ESA WorldCover / Hansen | Simulated |
| Ranger posts, past pressure | Placeholders for NTCA / MIKE | Simulated |
| Threat score, risk level | Documented proxy rule (reports/label_rule.md) | Proxy label |

**Say this clearly in the report and viva:** the labels are proxies, not verified poaching incidents,
because exact incident locations are confidential. Several features are simulated placeholders, each one named
above, with the real dataset it stands in for. The pipeline is built so each placeholder can be swapped for the
real raster without changing the model code.

## Rigour points

- **Spatial split:** whole reserves are held out (GroupShuffleSplit / GroupKFold by reserve), so there is no leakage from neighbouring cells.
- **Map predictions are out-of-fold:** every reserve on the dashboard is predicted by a model that never saw it.
- **Class imbalance:** balanced sample weights; evaluation focuses on recall and PR-AUC for High.
- **Honest baseline:** logistic regression (with log-scaled distances) comes close to XGBoost, which shows the proxy labels are fairly smooth. Expect a bigger gap with real incident data.

## Limitations

Proxy labels; approximate reserve shapes; Natural Earth only has major roads and rivers; several simulated
features; structural risk only, with no live intelligence. This is a decision-support tool for ranking where to
patrol first, not a replacement for forest officers' judgement.

## Website (BAGH)

The dashboard in `web/` is a scroll-story site: a tiger leaps out of a wine-red cover, walks through a "threat ballet" of objects that stand for the model's strongest risk drivers, and settles on a forest plate, followed by the atlas, patrol planner, model results and data notes. It is deployed on Vercel as a static site (`web/` folder); the Python API is optional and only powers live coordinate scoring.

Animal and object images were generated with Canva and Figma AI for this project.
