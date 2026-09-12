"""
train_model.py - Quantitative NFL Predictive Modeling & Calibration Pipeline.
Builds, validates, and serializes nfl_model.json with zero temporal leakage.
"""
import json
import math
import os
import nflreadpy as nfl
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
import xgboost as xgb

FEATURES = [
    "net_pass_edge",
    "net_rush_edge",
    "net_late_down_edge",
    "diff_success",
    "diff_explosive",
    "rest_diff",
    "is_divisional",
    "market_home_prob"
]

TARGET = "home_win"
MODEL_FILE = "nfl_model.json"
METRICS_FILE = "model_metadata.json"

SEASONS = [2021, 2022, 2023, 2024, 2025]
TEAM_MAP = {"LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX"}

def clean_abbr(val):
    if not isinstance(val, str):
        return val
    c = val.strip().upper()
    return TEAM_MAP.get(c, c)

def load_and_preprocess_pbp():
    print(f"Ingesting play-by-play data for seasons: {SEASONS}...")
    pbp = nfl.load_pbp(seasons=SEASONS).to_pandas()
    
    # 1. Filter to neutral game-script leverage (10% to 90% Win Prob)
    clean_pbp = pbp[
        (pbp["play_type"].isin(["pass", "run"])) &
        (pbp["home_wp"].between(0.10, 0.90)) &
        (pbp["qtr"] <= 4)
    ].copy()

    for col in ["posteam", "defteam"]:
        clean_pbp[col] = clean_pbp[col].apply(clean_abbr)

    clean_pbp["is_early_down"] = clean_pbp["down"].isin([1, 2]).astype(int)
    clean_pbp["is_late_down"] = clean_pbp["down"].isin([3, 4]).astype(int)
    clean_pbp["is_explosive"] = (
        ((clean_pbp["play_type"] == "pass") & (clean_pbp["yards_gained"] >= 15)) |
        ((clean_pbp["play_type"] == "run") & (clean_pbp["yards_gained"] >= 10))
    ).astype(int)

    off = clean_pbp.groupby(["season", "week", "posteam"]).agg(
        off_dropback_epa=("epa", lambda x: x[clean_pbp.loc[x.index, "play_type"] == "pass"].mean()),
        off_rush_epa=("epa", lambda x: x[clean_pbp.loc[x.index, "play_type"] == "run"].mean()),
        off_early_success=("success", lambda x: x[clean_pbp.loc[x.index, "is_early_down"] == 1].mean()),
        off_late_epa=("epa", lambda x: x[clean_pbp.loc[x.index, "is_late_down"] == 1].mean()),
        off_explosive=("is_explosive", "mean")
    ).reset_index().rename(columns={"posteam": "team"})

    defs = clean_pbp.groupby(["season", "week", "defteam"]).agg(
        def_dropback_epa=("epa", lambda x: x[clean_pbp.loc[x.index, "play_type"] == "pass"].mean()),
        def_rush_epa=("epa", lambda x: x[clean_pbp.loc[x.index, "play_type"] == "run"].mean()),
        def_early_success=("success", lambda x: x[clean_pbp.loc[x.index, "is_early_down"] == 1].mean()),
        def_late_epa=("epa", lambda x: x[clean_pbp.loc[x.index, "is_late_down"] == 1].mean())
    ).reset_index().rename(columns={"defteam": "team"})

    team_perf = pd.merge(off, defs, on=["season", "week", "team"], how="outer").fillna(0.0)
    team_perf.sort_values(["team", "season", "week"], inplace=True)

    # 2. Strict Lagging: Roll exponentially weighted metrics using strictly prior games (shift 1)
    stat_cols = [
        "off_dropback_epa", "off_rush_epa", "off_early_success", "off_late_epa", "off_explosive",
        "def_dropback_epa", "def_rush_epa", "def_early_success", "def_late_epa"
    ]
    for col in stat_cols:
        team_perf[f"roll_{col}"] = (
            team_perf.groupby("team")[col]
            .transform(lambda s: s.shift(1).ewm(span=6, min_periods=1).mean())
        )

    return team_perf

def build_training_dataset(team_perf):
    print("Ingesting schedules and merging causal features...")
    sched = nfl.load_schedules(seasons=SEASONS).to_pandas()
    sched = sched[sched["game_type"] == "REG"].copy()
    sched = sched[sched["result"].notna()].copy()

    for col in ["home_team", "away_team"]:
        sched[col] = sched[col].apply(clean_abbr)

    sched["home_win"] = (sched["result"] > 0).astype(int)

    # Implied Market Probability
    def calc_mkt_prob(row):
        h_ml = row.get("home_moneyline")
        a_ml = row.get("away_moneyline")
        if pd.notna(h_ml) and pd.notna(a_ml) and h_ml != 0 and a_ml != 0:
            p_h = 100.0 / (h_ml + 100.0) if h_ml > 0 else abs(h_ml) / (abs(h_ml) + 100.0)
            p_a = 100.0 / (a_ml + 100.0) if a_ml > 0 else abs(a_ml) / (abs(a_ml) + 100.0)
            return float(p_h / (p_h + p_a))
        sp = row.get("spread_line", 0.0) or 0.0
        return float(norm.cdf(sp / 13.45))

    sched["market_home_prob"] = sched.apply(calc_mkt_prob, axis=1)

    rows = []
    for _, g in sched.iterrows():
        s = g["season"]
        w = g["week"]
        h = g["home_team"]
        a = g["away_team"]

        h_stat = team_perf[(team_perf["team"] == h) & (team_perf["season"] == s) & (team_perf["week"] == w)]
        a_stat = team_perf[(team_perf["team"] == a) & (team_perf["season"] == s) & (team_perf["week"] == w)]

        if h_stat.empty or a_stat.empty:
            continue

        h_row = h_stat.iloc[0]
        a_row = a_stat.iloc[0]

        net_pass = (h_row["roll_off_dropback_epa"] - a_row["roll_def_dropback_epa"]) - \
                   (a_row["roll_off_dropback_epa"] - h_row["roll_def_dropback_epa"])
        net_rush = (h_row["roll_off_rush_epa"] - a_row["roll_def_rush_epa"]) - \
                   (a_row["roll_off_rush_epa"] - h_row["roll_def_rush_epa"])
        net_late = (h_row["roll_off_late_epa"] - a_row["roll_def_late_epa"]) - \
                   (a_row["roll_off_late_epa"] - h_row["roll_def_late_epa"])
        diff_succ = h_row["roll_off_early_success"] - a_row["roll_off_early_success"]
        diff_expl = h_row["roll_off_explosive"] - a_row["roll_off_explosive"]
        rest_diff = float(g.get("home_rest", 7.0) or 7.0) - float(g.get("away_rest", 7.0) or 7.0)
        is_div = int(g.get("div_game", 0) or 0)

        rows.append({
            "season": s,
            "week": w,
            "home_win": g["home_win"],
            "net_pass_edge": net_pass,
            "net_rush_edge": net_rush,
            "net_late_down_edge": net_late,
            "diff_success": diff_succ,
            "diff_explosive": diff_expl,
            "rest_diff": rest_diff,
            "is_divisional": is_div,
            "market_home_prob": g["market_home_prob"]
        })

    return pd.DataFrame(rows)

def train_and_export():
    team_perf = load_and_preprocess_pbp()
    df_train = build_training_dataset(team_perf)

    print(f"Compiled {len(df_train)} historical fixtures with zero leakage.")
    
    # Time-series Split: Train on earlier seasons, test out-of-sample on the latest completed season
    split_season = max(SEASONS)
    train_data = df_train[df_train["season"] < split_season].copy()
    test_data = df_train[df_train["season"] == split_season].copy()

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]
    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    # Hyperparameters tuned to prevent collinear overfitting to market pricing
    clf = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.50,
        reg_lambda=1.50,
        random_state=42,
        eval_metric="logloss"
    )

    clf.fit(X_train, y_train)

    # Probability Calibration via Isotonic Regression
    calibrated = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv="prefit")
    calibrated.fit(X_train, y_train)

    preds_prob = calibrated.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, preds_prob)
    loss = log_loss(y_test, preds_prob)

    print(f"Test Set ({split_season}) Out-of-Sample Results:")
    print(f" -> Brier Score: {brier:.4f}")
    print(f" -> Log Loss:    {loss:.4f}")

    # Retrain on full dataset and serialize
    clf.fit(df_train[FEATURES], df_train[TARGET])
    clf.save_model(MODEL_FILE)
    print(f"Serialized production model to '{MODEL_FILE}'.")

    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "test_season": split_season,
        "brier_score": round(brier, 4),
        "log_loss": round(loss, 4),
        "sample_size": len(df_train),
        "calibration": "sigmoid"
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Exported metadata to '{METRICS_FILE}'.")

if __name__ == "__main__":
    train_and_export()
