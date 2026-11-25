"""Lightweight Streamlit view of the same models (the main UI is the custom dashboard in web/).

Run:  streamlit run streamlit_app.py
"""
import json
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).parent
st.set_page_config(page_title="Poaching Risk Predictor", layout="wide")


@st.cache_data
def load():
    cells = pd.read_parquet(ROOT / "data/processed/grid_cells.parquet")
    pred = pd.read_parquet(ROOT / "data/processed/predictions.parquet")
    df = cells.merge(pred.drop(columns=["reserve_id", "x", "y", "lat", "lon"]), on=["cell_id", "season"])
    metrics = json.load(open(ROOT / "reports/metrics.json"))
    return df, metrics


df, metrics = load()
st.title("Poaching risk across Indian reserves")
c1, c2, c3 = st.columns(3)
season = c1.radio("Season", ["dry", "wet"], horizontal=True)
reserve = c2.selectbox("Reserve", sorted(df.name.unique()))
show = c3.radio("Show", ["Predicted risk", "High-risk cells only"], horizontal=True)

sub = df[(df.name == reserve) & (df.season == season)]
if show == "High-risk cells only":
    sub = sub[sub.pred_class == "High"]
k1, k2, k3 = st.columns(3)
k1.metric("Cells (1 km²)", f"{len(sub):,}")
k2.metric("Mean threat score", f"{sub.pred_score.mean():.1f}")
k3.metric("High-risk share", f"{(sub.pred_class == 'High').mean() * 100:.1f}%")

colors = {"Low": [157, 184, 166], "Medium": [242, 154, 69], "High": [194, 59, 18]}
sub = sub.assign(color=sub.pred_class.map(colors))
st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(latitude=float(sub.lat.mean()), longitude=float(sub.lon.mean()), zoom=9),
    layers=[pdk.Layer("ScatterplotLayer", data=sub[["lon", "lat", "color", "pred_score"]],
                      get_position="[lon, lat]", get_fill_color="color", get_radius=450, pickable=True)],
    tooltip={"text": "Threat score {pred_score}"}, map_style=None))

st.subheader("Model comparison (held-out reserves)")
st.dataframe(pd.DataFrame([{k: m[k] for k in ["model", "macro_f1", "recall_high", "pr_auc_high", "roc_auc_ovr"]}
                           for m in metrics["classification"]]), hide_index=True)
st.dataframe(pd.DataFrame([{k: m[k] for k in ["model", "rmse", "mae", "r2"]} for m in metrics["regression"]]),
             hide_index=True)
st.subheader("What drives risk (mean |SHAP|)")
st.bar_chart(pd.Series(metrics["shap"]["regressor_mean_abs"]).head(10))
