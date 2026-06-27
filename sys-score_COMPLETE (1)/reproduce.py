#!/usr/bin/env python3
"""Reproduce EndoSysScore held-out internal-validation results from the deposited model.

Loads the trained model and the held-out internal validation cohort, regenerates the
EndoSysScore predictions, and prints the headline metrics. Compare the output
against results/G_BEST_full_metrics.json.

Usage:  python reproduce.py
Requires:  scikit-learn, xgboost, lightgbm, betacal, numpy, pandas, scipy
           (pinned versions in requirements.txt; Python >= 3.11)
"""
import sys, json, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent

# ===================================================================
#  Model class definitions (required to unpickle the deposited model)
# ===================================================================
import json
import pickle
import os
import numpy as np
import pandas as pd

# ── Reproduce ModelSGMeta and add_pattern_features from training script ────────

def add_pattern_features(df, lvef_med):
    df = df.copy()
    lv = df["lvef"].fillna(lvef_med)
    df["in_pattern_A"] = (
        (df["multivalve_final"] == 1) & (df["saureus_b"] == 1) &
        ((df["IRC_stage"] >= 2) | (df["DIALISI"] == 1) | (df["PAPSgt50"] == 1))
    ).astype(int)
    df["in_pattern_B"] = (
        (df["ETA"] >= 80) & (df["NVE"] == 1) &
        ((df["BPCO"] == 1) | (lv < 50))
    ).astype(int)
    df["in_pattern_C"] = (
        ((df["ASCESSO"] == 1) | (df["periannular"] == 1)) &
        (df["isolated_IM"] == 1) & (df["ETA"] >= 65)
    ).astype(int)
    df["in_pattern_D"] = (
        (df["PVE"] == 1) & (df["saureus_b"] == 1) &
        (df["IM"] == 1) & (df["AO_final"] == 1)
    ).astype(int)
    df["in_pattern_E"] = (
        (df["NVE"] == 1) & (df["ETA"] >= 80) & (lv < 50)
    ).astype(int)
    df["in_pattern_F"] = (
        (df["ASCESSO"] == 1) & (df["isolated_IM"] == 1) &
        (df["ETA"] >= 65)
    ).astype(int)
    df["n_patterns_satisfied"] = sum(
        df[f"in_pattern_{p}"] for p in "ABCDEF"
    )
    return df


class ModelSGMeta:
    """Stacking model: base + XGB/LGB meta + primary iso + beta cal."""
    def __init__(self, base_models, base_names, meta_model, iso_primary,
                 beta_cal, feat_cols, pat_feat_idx, clinical_idx,
                 study_medians, lvef_med):
        self.base_models   = base_models
        self.base_names    = base_names
        self.meta_model    = meta_model
        self.iso_primary   = iso_primary
        self.beta_cal      = beta_cal
        self.feat_cols     = feat_cols
        self.pat_feat_idx  = pat_feat_idx
        self.clinical_idx  = clinical_idx
        self.study_medians = study_medians
        self.lvef_med      = lvef_med

    def _to_X(self, df):
        df2 = add_pattern_features(df, self.lvef_med)
        for c in self.feat_cols:
            if c not in df2.columns:
                df2[c] = 0
        df2[self.feat_cols] = df2[self.feat_cols].fillna(self.study_medians)
        return df2[self.feat_cols].values.astype(np.float32)

    def predict_from_df(self, df):
        X = self._to_X(df)
        bp = np.column_stack([
            self.base_models[n].predict_proba(X)[:, 1]
            for n in self.base_names
        ])
        mfeat = np.hstack([bp, X[:, self.pat_feat_idx], X[:, self.clinical_idx]])
        meta_pred = self.meta_model.predict_proba(mfeat)[:, 1]
        iso_pred  = self.iso_primary.transform(meta_pred)
        return np.clip(
            self.beta_cal.predict(iso_pred.reshape(-1, 1)).ravel(),
            1e-6, 1 - 1e-6
        )




# unpickle needs these in __main__
sys.modules["__main__"].ModelSGMeta = ModelSGMeta
sys.modules["__main__"].add_pattern_features = add_pattern_features

def main():
    model = pickle.load(open(ROOT / "models" / "Model_S_v3_WINNER.pkl", "rb"))
    ctrl  = pd.read_csv(ROOT / "data" / "ZENODO_CONTROL_external_930.csv", low_memory=False)
    y = ctrl["DECESSO_EARLY"].values.astype(float)
    p = model.predict_from_df(ctrl)

    auc   = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    oe    = y.sum() / p.sum()
    print("=" * 60)
    print("  EndoSysScore -- held-out internal validation reproduction")
    print("=" * 60)
    print(f"  n = {len(y)}   events = {int(y.sum())} ({100*y.mean():.1f}%)")
    print(f"  AUC   = {auc:.4f}")
    print(f"  Brier = {brier:.4f}")
    print(f"  O/E   = {oe:.4f}")

    X  = model._to_X(ctrl); fc = model.feat_cols
    print("\n  Rare high-lethality phenotypes (observed vs predicted):")
    print("  pattern |  n | observed | predicted |  gap")
    for k in "ABCDEF":
        m = X[:, fc.index("in_pattern_" + k)] == 1
        if m.sum() == 0:
            continue
        obs = 100 * y[m].mean(); prd = 100 * p[m].mean()
        print(f"     {k}    | {int(m.sum()):2d} |  {obs:5.1f}%  |  {prd:5.1f}%   | {prd-obs:+5.1f}")

    ref = json.load(open(ROOT / "results" / "G_BEST_full_metrics.json"))
    print(f"\n  Deposited reference (results/G_BEST_full_metrics.json):")
    print(f"     AUC {ref['auc']}  Brier {ref['brier']}  O/E {ref['oe']}  cal-slope {ref['cal_slope']}  HL p {ref['hl_p']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
