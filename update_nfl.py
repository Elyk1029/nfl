import json
import os
import time
from google import genai
from google.genai import types
import nflreadpy as nfl
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import xgboost as xgb

# 1. Environment & API Setup
db_url = os.environ.get("DATABASE_URL")
gemini_key = os.environ.get("GEMINI_API_KEY")

if not db_url or not gemini_key:
    raise ValueError("Missing DATABASE_URL or GEMINI_API_KEY in environment.")

engine = create_engine(db_url)
client = genai.Client(api_key=gemini_key)

# 2. Load Trained XGBoost Model
MODEL_FILE = "nfl_model.json"
model = xgb.XGBClassifier()
if os.path.exists(MODEL_FILE):
    model.load_model(MODEL_FILE)
    print("Loaded nfl_model.json successfully.")
else:
    raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found in root directory.")

FEATURES = [
    "net_pass_edge", "net_rush_edge", "net_late_down_edge", "diff_success",
    "diff_explosive", "rest_diff", "is_divisional", "market_home_prob",
]

# 3. Pull Schedule & Baseline Play-by-Play Data
CURRENT_SEASON = 2026
DATA_SEASON = 2025 

print(f"Loading NFL data (Historical baseline: {DATA_SEASON})...")

try:
    schedules = nfl.load_schedules(seasons=[CURRENT_SEASON]).to_pandas()
except Exception:
    schedules = nfl.load_schedules(seasons=[DATA_SEASON]).to_pandas()

try:
    pbp = nfl.load_pbp(seasons=[DATA_SEASON]).to_pandas()
except Exception as e:
    print(f"Notice: Could not load PBP for {DATA_SEASON}: {e}")
    pbp = pd.DataFrame()
    
# Player Stats for Projections
try:
    player_stats = nfl.load_player_stats(seasons=[DATA_SEASON]).to_pandas()
except Exception as e:
    print(f"Notice: Could not load Player Stats for {DATA_SEASON}: {e}")
    player_stats = pd.DataFrame()

# Clean scrimmage plays and define down leverage
if not pbp.empty:
    pbp_scrimmage = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    pbp_scrimmage["is_late_down"] = (
        pbp_scrimmage["down"].isin([3, 4]).astype(int)
        if "down" in pbp_scrimmage.columns else 0
    )
else:
    pbp_scrimmage = pd.DataFrame()

metric_cols = [
    "off_dropback_epa", "off_rush_epa", "off_success", "off_late_down_epa", "off_explosive",
    "def_dropback_epa", "def_rush_epa", "def_success", "def_late_down_epa",
]

if not pbp_scrimmage.empty:
    off_stats = pbp_scrimmage.groupby(["week", "posteam"]).agg(
        off_dropback_epa=("epa", lambda x: x[pbp_scrimmage.loc[x.index, "play_type"] == "pass"].mean()),
        off_rush_epa=("epa", lambda x: x[pbp_scrimmage.loc[x.index, "play_type"] == "run"].mean()),
        off_success=("success", "mean"),
        off_late_down_epa=("epa", lambda x: x[pbp_scrimmage.loc[x.index, "is_late_down"] == 1].mean()),
        off_explosive=("yards_gained", "std"),
    ).reset_index().rename(columns={"posteam": "team"})

    def_stats = pbp_scrimmage.groupby(["week", "defteam"]).agg(
        def_dropback_epa=("epa", lambda x: x[pbp_scrimmage.loc[x.index, "play_type"] == "pass"].mean()),
        def_rush_epa=("epa", lambda x: x[pbp_scrimmage.loc[x.index, "play_type"] == "run"].mean()),
        def_success=("success", "mean"),
        def_late_down_epa=("epa", lambda x: x[pbp_scrimmage.loc[x.index, "is_late_down"] == 1].mean()),
    ).reset_index().rename(columns={"defteam": "team"})

    team_perf = pd.merge(off_stats, def_stats, on=["week", "team"], how="outer").fillna(0)
    team_perf.sort_values(["team", "week"], inplace=True)

    for col in metric_cols:
        team_perf[f"roll_{col}"] = team_perf.groupby("team")[col].transform(
            lambda x: x.shift(1).ewm(span=6, min_periods=1).mean()
        )
else:
    team_perf = pd.DataFrame(columns=["week", "team"] + [f"roll_{c}" for c in metric_cols])

# Aggregate rolling baselines for top skill positions
top_qbs = top_rbs = top_wrs = pd.DataFrame()
if not player_stats.empty:
    recent_stats = player_stats[player_stats['week'] >= player_stats['week'].max() - 4]
    player_baselines = recent_stats.groupby(['recent_team', 'player_id', 'player_name', 'position']).agg(
        avg_pass_yards=('passing_yards', 'mean'),
        avg_pass_tds=('passing_tds', 'mean'),
        avg_rush_yards=('rushing_yards', 'mean'),
        avg_rec_yards=('receiving_yards', 'mean'),
        avg_targets=('targets', 'mean')
    ).reset_index()
    
    top_qbs = player_baselines[player_baselines['position'] == 'QB'].sort_values('avg_pass_yards', ascending=False).groupby('recent_team').head(1)
    top_rbs = player_baselines[player_baselines['position'] == 'RB'].sort_values('avg_rush_yards', ascending=False).groupby('recent_team').head(1)
    top_wrs = player_baselines[player_baselines['position'] == 'WR'].sort_values('avg_targets', ascending=False).groupby('recent_team').head(1)

# 4. Filter Upcoming Unplayed Matchups (Pulls the entire next slate)
next_week = schedules[schedules["result"].isna()]["week"].min()
upcoming = schedules[(schedules["result"].isna()) & (schedules["week"] == next_week)].copy()

system_prompt = """
You are a lead quantitative NFL handicapper and schematic analyst. You are evaluating an upcoming matchup using Vegas lines, proprietary machine learning win probabilities, and player-level baselines.

Your task is to synthesize this data and output a strictly structured JSON object. You must leverage your internal knowledge of current NFL coaching staffs, defensive coordinators, base coverages (e.g., Cover 3, Quarters, Man-Match), and blitz tendencies to contextualize the matchups.

You must output a JSON object with the following exact keys:
{
  "executive_summary": "A 2-sentence breakdown of the quantitative edge and market value.",
  "schematic_matchup": {
    "away_offense_vs_home_defense": "Detailed analysis of how the away team's offensive scheme matches up against the home team's defensive playbook and coordinator tendencies.",
    "home_offense_vs_away_defense": "Detailed analysis of how the home team's offensive scheme matches up against the away team's defensive playbook and coordinator tendencies."
  },
  "player_projections": {
    "away_team": {
      "QB_projection": "Name - Projected Pass Yards, TDs. Include a 1-sentence rationale based on the opposing defense.",
      "RB_projection": "Name - Projected Rush Yards. Include a 1-sentence rationale.",
      "WR_projection": "Name - Projected Rec Yards. Include a 1-sentence rationale."
    },
    "home_team": {
      "QB_projection": "Name - Projected Pass Yards, TDs. Include a 1-sentence rationale based on the opposing defense.",
      "RB_projection": "Name - Projected Rush Yards. Include a 1-sentence rationale.",
      "WR_projection": "Name - Projected Rec Yards. Include a 1-sentence rationale."
    }
  },
  "actionable_verdict": "Your final betting recommendation on the Spread, Moneyline, or Total."
}
"""

def get_top_player_string(team_abbr, df, pos, stat_col, stat_name):
    player_row = df[df['recent_team'] == team_abbr]
    if not player_row.empty:
        name = player_row.iloc[0]['player_name']
        val = player_row.iloc[0][stat_col]
        return f"{name} (~{val:.1f} {stat_name}/gm)"
    return "Unknown"

records = []
for _, game in upcoming.iterrows():
    home_team = game["home_team"]
    away_team = game["away_team"]
    matchup = f"{away_team} @ {home_team}"
    week_num = int(game["week"]) if pd.notna(game["week"]) else 1

    home_row = team_perf[(team_perf["team"] == home_team) & (team_perf["week"] == week_num)]
    away_row = team_perf[(team_perf["team"] == away_team) & (team_perf["week"] == week_num)]

    if home_row.empty: 
        home_row = team_perf[team_perf["team"] == home_team].tail(1)
    if away_row.empty: 
        away_row = team_perf[team_perf["team"] == away_team].tail(1)

    def get_metric(df, col_name, default=0.0):
        if not df.empty and pd.notna(df[col_name].values[0]):
            return float(df[col_name].values[0])
        return default

    home_rest = float(game["home_rest"]) if pd.notna(game["home_rest"]) else 7.0
    away_rest = float(game["away_rest"]) if pd.notna(game["away_rest"]) else 7.0
    rest_diff = home_rest - away_rest
    is_divisional = int(game["div_game"]) if pd.notna(game.get("div_game")) else 0

    spread_line = float(game["spread_line"]) if pd.notna(game["spread_line"]) else 0.0
    total_line = float(game["total_line"]) if pd.notna(game["total_line"]) else 44.0

    market_home_prob = 1 / (1 + 10 ** (spread_line * 0.035))

    net_pass_edge = (get_metric(home_row, "roll_off_dropback_epa") - get_metric(away_row, "roll_def_dropback_epa")) - \
                    (get_metric(away_row, "roll_off_dropback_epa") - get_metric(home_row, "roll_def_dropback_epa"))

    net_rush_edge = (get_metric(home_row, "roll_off_rush_epa") - get_metric(away_row, "roll_def_rush_epa")) - \
                    (get_metric(away_row, "roll_off_rush_epa") - get_metric(home_row, "roll_def_rush_epa"))

    net_late_down_edge = (get_metric(home_row, "roll_off_late_down_epa") - get_metric(away_row, "roll_def_late_down_epa")) - \
                         (get_metric(away_row, "roll_off_late_down_epa") - get_metric(home_row, "roll_def_late_down_epa"))

    diff_success = get_metric(home_row, "roll_off_success", 0.45) - get_metric(away_row, "roll_off_success", 0.45)
    diff_explosive = get_metric(home_row, "roll_off_explosive", 8.0) - get_metric(away_row, "roll_off_explosive", 8.0)

    feature_row = pd.DataFrame([[net_pass_edge, net_rush_edge, net_late_down_edge, diff_success, diff_explosive, rest_diff, is_divisional, market_home_prob]], columns=FEATURES)

    model_home_prob = float(model.predict_proba(feature_row)[0][1])

    home_qb = get_top_player_string(home_team, top_qbs, 'QB', 'avg_pass_yards', 'pass yds')
    home_rb = get_top_player_string(home_team, top_rbs, 'RB', 'avg_rush_yards', 'rush yds')
    home_wr = get_top_player_string(home_team, top_wrs, 'WR', 'avg_rec_yards', 'rec yds')
  
    away_qb = get_top_player_string(away_team, top_qbs, 'QB', 'avg_pass_yards', 'pass yds')
    away_rb = get_top_player_string(away_team, top_rbs, 'RB', 'avg_rush_yards', 'rush yds')
    away_wr = get_top_player_string(away_team, top_wrs, 'WR', 'avg_rec_yards', 'rec yds')

    payload = {
        "matchup": matchup,
        "week": week_num,
        "spread_line": spread_line,
        "total_line": total_line,
        "market_home_win_probability": f"{market_home_prob:.1%}",
        "model_projections": {
            "home_win_probability": f"{model_home_prob:.1%}",
            "away_win_probability": f"{(1 - model_home_prob):.1%}",
            "probability_edge_vs_market": f"{(model_home_prob - market_home_prob):+.1%}",
            "net_pass_edge": round(net_pass_edge, 4),
            "net_rush_edge": round(net_rush_edge, 4),
        },
        "player_baselines": {
            "home_team": {"QB": home_qb, "RB": home_rb, "WR": home_wr},
            "away_team": {"QB": away_qb, "RB": away_rb, "WR": away_wr}
        }
    }

    max_retries = 3
    backoff = 4
    response = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=json.dumps(payload),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                    response_mime_type="application/json"
                ),
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API busy or error ({e}). Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
            else:
                print(f"API call failed after {max_retries} attempts: {e}")

    analysis_text = response.text if response else "{}"

    records.append({
        "game_id": str(game["game_id"]),
        "week": week_num,
        "matchup": matchup,
        "home_win_prob": model_home_prob,
        "market_prob": market_home_prob,
        "analysis": analysis_text,
    })
    print(f"Processed {matchup}")

if records:
    df_results = pd.DataFrame(records)
    df_results.to_sql("nfl_weekly_analysis", engine, if_exists="append", index=False)
    print("Neon database updated successfully with JSON model-backed predictions.")
