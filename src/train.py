"""Step 3 - train, tune (Optuna), evaluate and explain all models.

Spatially-aware evaluation: whole reserves are held out (GroupShuffleSplit / GroupKFold by
reserve), so the model is never tested on cells from a reserve it trained on.

Outputs
  models/*.joblib                  fitted pipelines
  reports/metrics.json             every number for the report / dashboard
  data/processed/predictions.parquet  out-of-fold predictions + SHAP top reasons for every cell

Run:  python -m src.train            (add --fast for a quick smoke run)
"""
import json, sys, time, warnings
import numpy as np
import pandas as pd
import joblib
import optuna
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (f1_score, recall_score, precision_score, roc_auc_score, average_precision_score,
                             confusion_matrix, precision_recall_curve, roc_curve, mean_squared_error,
                             mean_absolute_error, r2_score, accuracy_score)
from sklearn.utils.class_weight import compute_sample_weight

from .config import PROC, MODELS, REPORTS, SEED
from .features import NUMERIC, CATEGORICAL, FLAGS, CLASSES, X_of, add_flags, base_feature

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
FAST = "--fast" in sys.argv
N_TRIALS_XGB = 6 if FAST else 25
N_TRIALS_RF = 3 if FAST else 10


SKEWED = [c for c in NUMERIC if c.startswith("dist_") or c.endswith("_10km")]


def preprocessor(scale=True):
    """Tree models get raw values; linear models get log1p on skewed distances/counts + standard scaling."""
    if scale:
        num = Pipeline([("log", ColumnTransformer([("log", FunctionTransformer(np.log1p), SKEWED)],
                                                  remainder="passthrough", verbose_feature_names_out=False)),
                        ("sc", StandardScaler())])
    else:
        num = "passthrough"
    return ColumnTransformer([
        ("num", num, NUMERIC + FLAGS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def curve_points(x, y, n=60):
    idx = np.linspace(0, len(x) - 1, min(n, len(x))).astype(int)
    return [[round(float(x[i]), 4), round(float(y[i]), 4)] for i in idx]


def clf_report(name, y_true, proba, extra=None):
    pred = proba.argmax(1)
    hi = CLASSES.index("High")
    yb = (y_true == hi).astype(int)
    p, r, _ = precision_recall_curve(yb, proba[:, hi])
    fpr, tpr, _ = roc_curve(yb, proba[:, hi])
    out = {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "f1_per_class": dict(zip(CLASSES, f1_score(y_true, pred, average=None).round(4).tolist())),
        "recall_high": recall_score(yb, pred == hi),
        "precision_high": precision_score(yb, pred == hi, zero_division=0),
        "roc_auc_ovr": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
        "pr_auc_high": average_precision_score(yb, proba[:, hi]),
        "confusion": confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist(),
        "pr_curve_high": curve_points(r[::-1], p[::-1]),
        "roc_curve_high": curve_points(fpr, tpr),
    }
    if extra: out.update(extra)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def reg_report(name, y_true, pred, landscape):
    res = y_true - pred
    by_land = pd.DataFrame({"l": landscape, "abs": np.abs(res)}).groupby("l")["abs"].mean().round(2).to_dict()
    hist, edges = np.histogram(res, bins=30)
    rng = np.random.default_rng(SEED)
    s = rng.choice(len(pred), size=min(600, len(pred)), replace=False)
    return {"model": name, "rmse": round(float(np.sqrt(mean_squared_error(y_true, pred))), 3),
            "mae": round(float(mean_absolute_error(y_true, pred)), 3), "r2": round(float(r2_score(y_true, pred)), 4),
            "mae_by_landscape": by_land, "residual_hist": {"counts": hist.tolist(), "edges": edges.round(2).tolist()},
            "scatter": [[round(float(y_true[i]), 1), round(float(pred[i]), 1)] for i in s]}


def main():
    t0 = time.time()
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    df = add_flags(pd.read_parquet(PROC / "grid_cells.parquet"))
    X, groups = X_of(df), df.reserve_id.values
    y_cls = df.risk_label.map({c: i for i, c in enumerate(CLASSES)}).values
    y_reg = df.threat_score.values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=SEED)
    tr, te = next(gss.split(X, y_cls, groups))
    test_reserves = sorted(df.iloc[te].name.unique().tolist())
    print(f"train {len(tr)} rows / test {len(te)} rows; test reserves: {test_reserves}")
    Xtr, Xte, ytr, yte, rtr, rte = X.iloc[tr], X.iloc[te], y_cls[tr], y_cls[te], y_reg[tr], y_reg[te]
    gtr = groups[tr]
    sw = compute_sample_weight("balanced", ytr)

    # subsample for tuning speed
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(tr), size=min(40000, len(tr)), replace=False)
    Xs, ys, rs, gs = Xtr.iloc[sub], ytr[sub], rtr[sub], gtr[sub]
    gkf = GroupKFold(n_splits=3)

    metrics = {"split": {"train_rows": int(len(tr)), "test_rows": int(len(te)), "test_reserves": test_reserves,
                         "n_reserves": int(df.reserve_id.nunique()), "n_cells": int(df.cell_id.nunique())},
               "classification": [], "regression": [], "optuna": {}}
    fitted = {}

    # ---------------- classification
    lr = Pipeline([("prep", preprocessor()), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    lr.fit(Xtr, ytr); fitted["logreg"] = lr
    metrics["classification"].append(clf_report("Logistic Regression", yte, lr.predict_proba(Xte), {"role": "Baseline"}))

    dt = Pipeline([("prep", preprocessor(False)), ("m", DecisionTreeClassifier(max_depth=8, min_samples_leaf=50,
                                                                               class_weight="balanced", random_state=SEED))])
    dt.fit(Xtr, ytr); fitted["tree"] = dt
    metrics["classification"].append(clf_report("Decision Tree", yte, dt.predict_proba(Xte), {"role": "Simple non-linear"}))

    def rf_obj(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators", 80, 250), max_depth=trial.suggest_int("max_depth", 6, 20),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 40),
                 max_features=trial.suggest_float("max_features", 0.3, 0.9))
        sc = []
        for a, b in gkf.split(Xs, ys, gs):
            m = Pipeline([("prep", preprocessor(False)), ("m", RandomForestClassifier(**p, class_weight="balanced",
                                                                                      n_jobs=-1, random_state=SEED))])
            m.fit(Xs.iloc[a], ys[a]); sc.append(f1_score(ys[b], m.predict(Xs.iloc[b]), average="macro"))
        return np.mean(sc)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(rf_obj, n_trials=N_TRIALS_RF)
    metrics["optuna"]["random_forest"] = {"best_params": st.best_params, "best_cv_macro_f1": round(st.best_value, 4),
                                          "history": [round(t.value, 4) for t in st.trials]}
    rf = Pipeline([("prep", preprocessor(False)), ("m", RandomForestClassifier(**st.best_params, class_weight="balanced",
                                                                               n_jobs=-1, random_state=SEED))])
    rf.fit(Xtr, ytr); fitted["rf"] = rf
    metrics["classification"].append(clf_report("Random Forest", yte, rf.predict_proba(Xte), {"role": "Main model"}))
    print("rf done", round(time.time() - t0))

    def xgb_params(trial):
        return dict(n_estimators=trial.suggest_int("n_estimators", 150, 700), max_depth=trial.suggest_int("max_depth", 3, 9),
                    learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    min_child_weight=trial.suggest_float("min_child_weight", 1, 20, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True))

    def xgbc_obj(trial):
        p = xgb_params(trial); sc = []
        for a, b in gkf.split(Xs, ys, gs):
            m = Pipeline([("prep", preprocessor(False)), ("m", xgb.XGBClassifier(**p, tree_method="hist", n_jobs=-1,
                                                                                 random_state=SEED))])
            m.fit(Xs.iloc[a], ys[a], m__sample_weight=compute_sample_weight("balanced", ys[a]))
            sc.append(f1_score(ys[b], m.predict(Xs.iloc[b]), average="macro"))
        return np.mean(sc)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(xgbc_obj, n_trials=N_TRIALS_XGB)
    metrics["optuna"]["xgboost_classifier"] = {"best_params": st.best_params, "best_cv_macro_f1": round(st.best_value, 4),
                                               "history": [round(t.value, 4) for t in st.trials]}
    xgc_params = st.best_params
    xgc = Pipeline([("prep", preprocessor(False)), ("m", xgb.XGBClassifier(**xgc_params, tree_method="hist", n_jobs=-1,
                                                                           random_state=SEED))])
    xgc.fit(Xtr, ytr, m__sample_weight=sw); fitted["xgb_clf"] = xgc
    metrics["classification"].append(clf_report("XGBoost", yte, xgc.predict_proba(Xte), {"role": "Best performer"}))
    print("xgb clf done", round(time.time() - t0))

    # ---------------- regression
    land_te = df.iloc[te].landscape.values
    lin = Pipeline([("prep", preprocessor()), ("m", LinearRegression())]).fit(Xtr, rtr); fitted["linreg"] = lin
    metrics["regression"].append({**reg_report("Multiple Linear Regression", rte, lin.predict(Xte), land_te), "role": "Baseline"})
    rp = metrics["optuna"]["random_forest"]["best_params"]
    rfr = Pipeline([("prep", preprocessor(False)), ("m", RandomForestRegressor(**rp, n_jobs=-1, random_state=SEED))]).fit(Xtr, rtr)
    fitted["rf_reg"] = rfr
    metrics["regression"].append({**reg_report("Random Forest Regressor", rte, rfr.predict(Xte), land_te), "role": "Main model"})

    def xgbr_obj(trial):
        p = xgb_params(trial); sc = []
        for a, b in gkf.split(Xs, rs, gs):
            m = Pipeline([("prep", preprocessor(False)), ("m", xgb.XGBRegressor(**p, tree_method="hist", n_jobs=-1,
                                                                                random_state=SEED))])
            m.fit(Xs.iloc[a], rs[a]); sc.append(np.sqrt(mean_squared_error(rs[b], m.predict(Xs.iloc[b]))))
        return np.mean(sc)
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(xgbr_obj, n_trials=N_TRIALS_XGB)
    metrics["optuna"]["xgboost_regressor"] = {"best_params": st.best_params, "best_cv_rmse": round(st.best_value, 4),
                                              "history": [round(t.value, 4) for t in st.trials]}
    xgr_params = st.best_params
    xgr = Pipeline([("prep", preprocessor(False)), ("m", xgb.XGBRegressor(**xgr_params, tree_method="hist", n_jobs=-1,
                                                                          random_state=SEED))]).fit(Xtr, rtr)
    fitted["xgb_reg"] = xgr
    metrics["regression"].append({**reg_report("XGBoost Regressor", rte, xgr.predict(Xte), land_te), "role": "Best performer"})
    print("regression done", round(time.time() - t0))

    # ---------------- global explainability (TreeSHAP via XGBoost's native implementation)
    names = list(xgr.named_steps["prep"].get_feature_names_out())
    base = [base_feature(n) for n in names]
    dte = xgb.DMatrix(xgr.named_steps["prep"].transform(Xte), feature_names=names)
    contrib = xgr.named_steps["m"].get_booster().predict(dte, pred_contribs=True)[:, :-1]
    agg = pd.DataFrame(np.abs(contrib), columns=base).T.groupby(level=0).sum().T.mean().sort_values(ascending=False)
    # beeswarm-style sample: feature value (scaled rank) vs shap value
    Xte_raw = Xte.reset_index(drop=True)
    sw_rows = np.random.default_rng(SEED).choice(len(Xte_raw), size=min(250, len(Xte_raw)), replace=False)
    signed = pd.DataFrame(contrib, columns=base).T.groupby(level=0).sum().T
    swarm = {}
    for f in agg.index[:8]:
        if f in NUMERIC + FLAGS:
            v = Xte_raw[f].values.astype(float); rk = pd.Series(v).rank(pct=True).values
            swarm[f] = [[round(float(signed[f].values[i]), 2), round(float(rk[i]), 2)] for i in sw_rows]
    dc = xgb.DMatrix(xgc.named_steps["prep"].transform(Xte), feature_names=list(xgc.named_steps["prep"].get_feature_names_out()))
    cc = xgc.named_steps["m"].get_booster().predict(dc, pred_contribs=True)  # (n, classes, feats+1)
    hi = CLASSES.index("High")
    agg_c = pd.DataFrame(np.abs(cc[:, hi, :-1]), columns=[base_feature(n) for n in dc.feature_names]).T.groupby(level=0).sum().T.mean()
    metrics["shap"] = {"regressor_mean_abs": agg.round(3).to_dict(),
                       "classifier_high_mean_abs": agg_c.sort_values(ascending=False).round(4).to_dict(),
                       "swarm": swarm}

    # ---------------- out-of-fold predictions for the map (every reserve predicted by a model that never saw it)
    oof_score, oof_proba = np.zeros(len(df)), np.zeros((len(df), 3))
    shap_all = np.zeros((len(df), len(set(base))))
    feat_order = sorted(set(base), key=lambda f: list(agg.index).index(f))
    for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(X, y_cls, groups)):
        mr = Pipeline([("prep", preprocessor(False)), ("m", xgb.XGBRegressor(**xgr_params, tree_method="hist", n_jobs=-1,
                                                                             random_state=SEED))]).fit(X.iloc[a], y_reg[a])
        mc = Pipeline([("prep", preprocessor(False)), ("m", xgb.XGBClassifier(**xgc_params, tree_method="hist", n_jobs=-1,
                                                                              random_state=SEED))])
        mc.fit(X.iloc[a], y_cls[a], m__sample_weight=compute_sample_weight("balanced", y_cls[a]))
        oof_score[b] = mr.predict(X.iloc[b]); oof_proba[b] = mc.predict_proba(X.iloc[b])
        nm = list(mr.named_steps["prep"].get_feature_names_out())
        c = mr.named_steps["m"].get_booster().predict(xgb.DMatrix(mr.named_steps["prep"].transform(X.iloc[b]),
                                                                   feature_names=nm), pred_contribs=True)[:, :-1]
        s = pd.DataFrame(c, columns=[base_feature(n) for n in nm]).T.groupby(level=0).sum().T[feat_order]
        shap_all[b] = s.values
        print("oof fold", k, round(time.time() - t0))
    yhat = oof_proba.argmax(1)
    metrics["oof"] = {"macro_f1": round(f1_score(y_cls, yhat, average="macro"), 4),
                      "recall_high": round(recall_score(y_cls == 2, yhat == 2), 4),
                      "rmse": round(float(np.sqrt(mean_squared_error(y_reg, oof_score))), 3),
                      "r2": round(float(r2_score(y_reg, oof_score)), 4)}

    top = np.argsort(-np.abs(shap_all), axis=1)[:, :3]
    pred = df[["cell_id", "reserve_id", "season", "x", "y", "lat", "lon"]].copy()
    pred["pred_score"] = np.clip(oof_score, 0, 100).round(1)
    pred["pred_class"] = [CLASSES[i] for i in yhat]
    pred["p_high"] = oof_proba[:, 2].round(3)
    for j in range(3):
        pred[f"r{j}_feat"] = [feat_order[i] for i in top[:, j]]
        pred[f"r{j}_val"] = shap_all[np.arange(len(df)), top[:, j]].round(2)
    pred.to_parquet(PROC / "predictions.parquet", index=False)

    # ---------------- persist
    for k, m in fitted.items():
        if k.startswith("rf"):
            continue  # random forests are hundreds of MB; re-train with this script if needed
        joblib.dump(m, MODELS / f"{k}.joblib", compress=3)
    json.dump({"feature_order": feat_order}, open(MODELS / "meta.json", "w"))
    metrics["runtime_s"] = round(time.time() - t0, 1)
    json.dump(metrics, open(REPORTS / "metrics.json", "w"), indent=1)
    print(json.dumps({m["model"]: [m["macro_f1"], m["recall_high"], m["pr_auc_high"]] for m in metrics["classification"]}))
    print(json.dumps({m["model"]: [m["rmse"], m["r2"]] for m in metrics["regression"]}))
    print("oof", metrics["oof"], "time", metrics["runtime_s"])


if __name__ == "__main__":
    main()
