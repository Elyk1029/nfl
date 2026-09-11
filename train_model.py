"""
train_model.py - Historical XGBoost Model Trainer for NFL Spread Prediction.
Produces the serialized nfl_model.json required by update_nfl.py.
"""
import os
import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# 1. Configuration & Training Seasons
TRAINING_SEASONS = [2022, 2023, 2024, 2025]
MODEL_OUTPUT_PATH = "nfl_model.json"

print(f"Ingesting historical play-by-play data for seasons: {TRAINING_SEASONS}...")
pbp = nfl.load_pbp(seasons=TRAINING_SEASONS).to_pandas()
schedules = nfl.load_schedules(seasons=TRAINING_SEASONS).to_pandas()

if pbp.empty or schedules.empty:
    raise ValueError("FATAL: Failed to ingest historical training data from nflreadpy.")

# Clean team abbreviations
TEAM_ABBR_MAP = {"LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX"}
def clean_abbr(s):
    if not isinstance(s, str): return s
    c = s.strip().upper()
    return TEAM_ABBR_MAP.get(c, c)

for df in [pbp, schedules]:
    for col in ["home_team", "away_team", "posteam", "defteam"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_abbr)

# 2. Compute Neutral-Script Rolling Team Efficiencies
print("Engineering rolling efficiency features...")
pbp_clean = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
if "home_wp" in pbp_clean.columns and "qtr" in pbp_clean.columns:
    pbp_clean = pbp_clean[(pbp_clean["qtr"] <= 3) | (pbp_clean["home_wp"].between(0.10, 0.90))]

pbp_clean["is_early_down"] = pbp_clean["down"].isin([1, 2]).astype(int) if "down" in pbp_clean.columns else 1
pbp_clean["is_late_down"] = pbp_clean["down"].isin([3, 4]).astype(int) if "down" in pbp_clean.columns else 0
pbp_clean["is_explosive"] = (
    ((pbp_clean["play_type"] == "pass") & (pbp_clean["yards_gained"] >= 15)) |
    ((pbp_clean["play_type"] == "run") & (pbp_clean["yards_gained"] >= 10))
).astype(int)

off_stats = pbp_clean.groupby(["season", "week", "posteam"]).agg(
    off_dropback_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "play_type"] == "pass"].mean()),
    off_rush_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "play_type"] == "run"].mean()),
    off_early_down_success=("success", lambda x: x[pbp_clean.loc[x.index, "is_early_down"] == 1].mean()),
    off_late_down_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "is_late_down"] == 1].mean()),
    off_explosive=("is_explosive", "mean"),
).reset_index().rename(columns={"posteam": "team"})

def_stats = pbp_clean.groupby(["season", "week", "defteam"]).agg(
    def_dropback_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "play_type"] == "pass"].mean()),
    def_rush_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "play_type"] == "run"].mean()),
    def_early_down_success=("success", lambda x: x[pbp_clean.loc[x.index, "is_early_down"] == 1].mean()),
    def_late_down_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "is_late_down"] == 1].mean()),
).reset_index().rename(columns={"defteam": "team"})

team_perf = pd.merge(off_stats, def_stats, on=["season", "week", "team"], how="outer").fillna(0)
team_perf.sort_values(["team", "season", "week"], inplace=True)

metric_cols = [
    "off_dropback_epa", "off_rush_epa", "off_early_down_success", "off_late_down_epa", "off_explosive",
    "def_dropback_epa", "def_rush_epa", "def_early_down_success", "def_late_down_epa",
]
for col in metric_cols:
    team_perf[f"roll_{col}"] = team_perf.groupby("team"][col].transform(
        lambda x: x.shift(1).ewm(span=6, min_periods=1).mean()
    )

def get_row(season, week, team):
    sub = team_perf[(team_perf["season"] == season) & (team_perf["week"] == week) & (team_perf["team"] == team)]
    if not sub.empty:
        return sub.iloc[0]
    # Fallback to latest prior week if exact match missing
    prior = team_perf[(team_perf["team"] == team) & ((team_perf["season"] < season) | ((team_perf["season"] == season) & (team_perf["week"] < week)))]
    return prior.tail(1).iloc[0] if not prior.empty else pd.Series()

# 3. Assemble Game-Level Training Matrix
print("Building game-level training dataset...")
training_rows = []

completed_games = schedules[schedules["result"].notna() & schedules["home_score"].notna() & schedules["away_score"].notna()].copy()

for _, g in completed_games.iterrows():
    season = int(g["season"])
    week = int(g["week"])
    home = g["home_team"]
    away = g["away_team"]
    
    h_row = get_row(season, week, home)
    a_row = get_row(season, week, away)
    if h_row.empty or a_row.empty:
        continue

    def get_val(series, key):
        return float(series[key]) if key in series and pd.notna(series[key]) else 0.0

    net_pass_edge = (get_val(h_row, "roll_off_dropback_epa") - get_val(a_row, "roll_def_dropback_epa")) - \
                    (get_val(a_row, "roll_off_dropback_epa") - get_val(h_row, "roll_def_dropback_epa"))
    net_rush_edge = (get_val(h_row, "roll_off_rush_epa") - get_val(a_row, "roll_def_rush_epa")) - \
                    (get_val(a_row, "roll_off_rush_epa") - get_val(h_row, "roll_def_rush_epa"))
    net_late_down_edge = (get_val(h_row, "roll_off_late_down_epa") - get_val(a_row, "roll_def_late_down_epa")) - \
                         (get_val(a_row, "roll_off_late_down_epa") - get_val(h_row, "roll_def_late_down_epa"))
    diff_success = get_val(h_row, "roll_off_early_down_success", 0.44) - get_val(a_row, "roll_off_early_down_success", 0.44)
    diff_explosive = get_val(h_row, "roll_off_explosive", 0.12) - get_val(a_row, "roll_off_explosive", 0.12)

    home_rest = float(g.get("home_rest", 7.0)) if pd.notna(g.get("home_rest")) else 7.0
    away_rest = float(g.get("away_rest", 7.0)) if pd.notna(g.get("away_rest")) else 7.0
    rest_diff = home_rest - away_rest
    is_divisional = int(g.get("div_game", 0)) if pd.notna(g.get("div_game")) else 0

    spread_line = float(g["spread_line"]) if pd.notna(g.get("spread_line")) else 0.0
    market_home_prob = float(norm.cdf(spread_line / 13.5))

    home_score = float(g["home_score"])
    away_score = float(g["away_score"])
    home_win = 1 if home_score > away_score else (0.5 if home_score == away_score else 0)

    training_rows.append({
        "net_pass_edge": net_pass_edge,
        "net_rush_edge": net_rush_edge,
        "net_late_down_edge": net_late_down_edge,
        "diff_success": diff_success,
        "diff_explosive": diff_explosive,
        "rest_diff": rest_diff,
        "is_divisional": is_divisional,
        "market_home_prob": market_home_prob,
        "home_win": home_win
    })

train_df = pd.DataFrame(training_rows)
train_df = train_df[train_df["home_win"] != 0.5].dropna() # Drop ties for binary classification

FEATURES = [
    "net_pass_edge", "net_rush_edge", "net_late_down_edge", "diff_success",
    "diff_explosive", "rest_diff", "is_divisional", "market_home_prob"
]

X = train_df[FEATURES]
y = train_df["home_win"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 4. Train XGBoost Classifier
print(f"Training XGBoost model on {len(X_train)} historical fixtures...")
clf = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

clf.fit(X_train, y_train)

# 5. Evaluate Performance
preds = clf.predict_proba(X_test)[:, 1]
loss = log_loss(y_test, preds)
auc = roc_auc_score(y_test, preds)
print(f"Training Complete | Out-of-Sample Log Loss: {loss:.4f} | AUC: {auc:.4f}")

# 6. Serialize Booster to Disk
clf.save_model(MODEL_OUTPUT_PATH)
print(f"Model successfully saved to {MODEL_OUTPUT_PATH}")
