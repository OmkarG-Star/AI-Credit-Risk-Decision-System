"""Train, calibrate and validate the credit risk model.

Run from the repository root:  python -m credit_risk.train
Writes models/credit_scorer.joblib, reports/metrics.json and reports/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.data import (
    CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, build_features, load_modelling_table,
)
from credit_risk.evaluate import (
    best_threshold, bootstrap_ci, brier, calibration_table, policy_outcome, psi, score_metrics,
)
from credit_risk.model import CreditScorer, to_model_frame

REPORTS = Path("reports")
FIGURES = REPORTS / "figures"
MODELS = Path("models")
SEED = 42

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
    cat_smooth=50, verbose=-1, seed=SEED,
)


def logistic_baseline(X_train, y_train):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
         NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200), CATEGORICAL_FEATURES),
    ])
    model = make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))
    return model.fit(X_train, y_train)


def fit_lgb(X_tr, y_tr, X_es, y_es):
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=CATEGORICAL_FEATURES)
    des = lgb.Dataset(X_es, y_es, reference=dtr)
    return lgb.train(LGB_PARAMS, dtr, num_boost_round=3000, valid_sets=[des],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    table = load_modelling_table()
    X_all = build_features(table)
    parts = {s: table.index[table["split"] == s] for s in ["train", "early_stop", "calibrate", "test"]}
    y = table["default"].to_numpy()
    counts = {s: {"loans": int(len(i)), "default_rate": float(y[i].mean()),
                  "issued": f"{table.loc[i, 'issue_d'].min():%b %Y} to {table.loc[i, 'issue_d'].max():%b %Y}"}
              for s, i in parts.items()}

    categories = {c: sorted(X_all.loc[parts["train"], c].unique().tolist()) for c in CATEGORICAL_FEATURES}
    Xm = {s: to_model_frame(X_all.loc[i], categories) for s, i in parts.items()}
    Xr = {s: X_all.loc[i] for s, i in parts.items()}
    ys = {s: y[i] for s, i in parts.items()}
    tr, es, cal, te = "train", "early_stop", "calibrate", "test"

    # 1. Logistic regression on application data
    lr = logistic_baseline(Xr[tr], ys[tr])
    # 2. LightGBM on application data (the model we ship)
    booster = fit_lgb(Xm[tr], ys[tr], Xm[es], ys[es])
    # 3. LightGBM with LC's sub-grade added (what an investor buying LC loans could use)
    Xg = {s: Xm[s].assign(sub_grade_rank=table.loc[parts[s], "sub_grade_rank"].to_numpy()) for s in parts}
    booster_g = fit_lgb(Xg[tr], ys[tr], Xg[es], ys[es])

    scores_test = {
        "Logistic regression (application data)": lr.predict_proba(Xr[te])[:, 1],
        "LightGBM (application data)": booster.predict(Xm[te]),
        "LC sub-grade alone (benchmark)": table.loc[parts[te], "sub_grade_rank"].to_numpy(),
        "LightGBM + LC sub-grade": booster_g.predict(Xg[te]),
    }
    comparison = []
    for name, s in scores_test.items():
        m = score_metrics(ys[te], s)
        m["roc_auc_ci"] = bootstrap_ci(ys[te], s, lambda a, b: score_metrics(a, b)["roc_auc"])
        comparison.append({"model": name, **m})

    # Calibration on Jul-Dec 2013, never seen by training or early stopping
    raw_cal = booster.predict(Xm[cal])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_cal, ys[cal])
    pd_cal = iso.predict(raw_cal)
    pd_test = iso.predict(booster.predict(Xm[te]))
    raw_test = booster.predict(Xm[te])

    cash = table["net_cash"].to_numpy()
    funded = table["funded_amnt"].astype(float).to_numpy()
    grade = table["grade"].astype(str).to_numpy()

    # Decision 1: a plain decline threshold, chosen on 2013 H2 and judged on 2014.
    thr = best_threshold(pd_cal, cash[parts[cal]])
    approve_all = policy_outcome(pd_test, ys[te], cash[parts[te]], funded[parts[te]], 1.01)
    decline_policy = policy_outcome(pd_test, ys[te], cash[parts[te]], funded[parts[te]], thr)

    # Decision 2: within each LC grade, flag loans riskier than that grade's
    # 75th-percentile PD. Cutoffs come from 2013 H2 only.
    cal_df = pd.DataFrame({"grade": grade[parts[cal]], "pd": pd_cal})
    grade_cutoffs = cal_df.groupby("grade")["pd"].quantile(0.75).to_dict()
    grade_avg_pd = cal_df.groupby("grade")["pd"].mean().to_dict()
    te_df = pd.DataFrame({"grade": grade[parts[te]], "pd": pd_test, "raw": booster.predict(Xm[te]), "y": ys[te],
                          "cash": cash[parts[te]], "funded": funded[parts[te]]})
    te_df["flag"] = te_df["pd"] >= te_df["grade"].map(grade_cutoffs)

    def summary(d):
        return {"loans": int(len(d)), "default_rate": float(d["y"].mean()),
                "return_on_funded": float(d["cash"].sum() / d["funded"].sum())}

    flag_policy = {"buy_all": summary(te_df), "skip_flagged": summary(te_df[~te_df["flag"]]),
                   "flagged_only": summary(te_df[te_df["flag"]]),
                   "share_flagged": float(te_df["flag"].mean())}
    by_grade = (te_df.groupby(["grade", "flag"])
                .apply(lambda d: pd.Series(summary(d)), include_groups=False)
                .reset_index())
    flag_policy["by_grade"] = by_grade.round(4).to_dict(orient="records")
    deciles = te_df.assign(decile=pd.qcut(te_df["raw"].rank(method="first"), 10, labels=False) + 1)
    decile_returns = (deciles.groupby("decile")
                      .apply(lambda d: pd.Series(summary(d)), include_groups=False).reset_index())

    calib = calibration_table(ys[te], pd_test, rank_score=raw_test)
    stability = {
        "score_psi_train_vs_test": psi(booster.predict(Xm[tr]), raw_test),
        "feature_psi_train_vs_test": {
            c: psi(Xr[tr][c], Xr[te][c]) for c in NUMERIC_FEATURES
        },
    }

    # SHAP on a sample of 2014 loans
    sample = Xm[te].sample(3000, random_state=SEED)
    sv = shap.TreeExplainer(booster).shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    importance = (pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False))

    # Figures
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 0.3], [0, 0.3], "--", color="grey", lw=1)
    ax.plot(calib["predicted"], calib["actual"], "o-", color="#0B6E4F")
    ax.set(xlabel="Predicted PD (decile mean)", ylabel="Actual default rate",
           title="Calibration on 2014 loans (test)", xlim=(0, 0.3), ylim=(0, 0.3))
    fig.tight_layout(); fig.savefig(FIGURES / "calibration.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(decile_returns["decile"], decile_returns["return_on_funded"] * 100, color="#0B6E4F")
    ax.set(xlabel="Model PD decile (1 = safest)", ylabel="3-year net cash return, %",
           title="2014 loans: every decile profitable, riskiest earns least")
    ax.set_xticks(range(1, 11))
    fig.tight_layout(); fig.savefig(FIGURES / "returns_by_decile.png", dpi=150); plt.close(fig)

    order = sorted(te_df["grade"].unique())
    piv = by_grade.pivot(index="grade", columns="flag", values="return_on_funded").reindex(order) * 100
    fig, ax = plt.subplots(figsize=(6.5, 4))
    w = 0.38
    xs = np.arange(len(order))
    ax.bar(xs - w / 2, piv[False], w, label="Not flagged", color="#0B6E4F")
    ax.bar(xs + w / 2, piv[True], w, label="Flagged: riskier than grade", color="#C0392B")
    counts_g = te_df.groupby("grade").size()
    ax.set_xticks(xs, [f"{g}\n{counts_g[g]:,} loans" for g in order], fontsize=8)
    ax.set(xlabel="LC grade", ylabel="3-year net cash return, %",
           title="2014 loans: flagged loans earn less in grades A-F")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGURES / "flag_by_grade.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    importance.iloc[::-1].plot.barh(ax=ax, color="#0B6E4F")
    ax.set(xlabel="Mean |SHAP value| (log-odds)", title="What drives the score (2014 sample)")
    fig.tight_layout(); fig.savefig(FIGURES / "shap_importance.png", dpi=150); plt.close(fig)

    scorer = CreditScorer(booster=booster, calibrator=iso, categories=categories,
                          grade_cutoffs=grade_cutoffs, grade_avg_pd=grade_avg_pd,
                          metadata={"trained_on": counts[tr]["issued"], "features": FEATURES})
    joblib.dump(scorer, MODELS / "credit_scorer.joblib", compress=3)

    metrics = {
        "splits": counts,
        "comparison_test": comparison,
        "best_iteration": booster.best_iteration,
        "calibration_test": {"brier": brier(ys[te], pd_test),
                             "mean_predicted": float(pd_test.mean()),
                             "actual_rate": float(ys[te].mean()),
                             "deciles": calib.round(4).to_dict(orient="records")},
        "policy_test": {"approve_all": approve_all, "decline_threshold": decline_policy,
                        "flag_within_grade": flag_policy,
                        "grade_cutoffs": grade_cutoffs,
                        "returns_by_decile": decile_returns.round(4).to_dict(orient="records")},
        "stability": stability,
        "shap_mean_abs": importance.round(4).to_dict(),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps({k: metrics[k] for k in ["splits", "comparison_test", "policy_test"]}, indent=2, default=float))
    print("brier", metrics["calibration_test"]["brier"], "score PSI", stability["score_psi_train_vs_test"])


if __name__ == "__main__":
    main()
