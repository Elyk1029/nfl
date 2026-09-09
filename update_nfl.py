import json
import os
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
  raise FileNotFoundError(
      f"Model file '{MODEL_FILE}' not found in the root directory."
  )

# Features expected by the self-trained walk-forward model
FEATURES = [
    "net_pass_edge",
    "net_rush_edge",
    "net_late_down_edge",
    "diff_success",
    "diff_explosive",
    "rest_diff",
    "is_divisional",
    "market_home_prob",
]

# 3. Pull Play-by-Play & Schedule Data
CURRENT_SEASON = 2026
DATA_SEASON = 2025  # Fallback to the latest available completed season for baseline stats

print(f"Loading NFL data (Schedule: {CURRENT_SEASON}, Historical PBP: {DATA_SEASON})...")

try:
    schedules = nfl.load_schedules(seasons=[CURRENT_SEASON]).to_pandas()
except Exception:
    schedules = nfl.load_schedules(seasons=[DATA_SEASON]).to_pandas()

try:
    pbp = nfl.load_pbp(seasons=[DATA_SEASON]).to_pandas()
except Exception as e:
    print(f"Notice: Could not load PBP for {DATA_SEASON}: {e}")
    pbp = pd.DataFrame()


metric_cols = [
    "off_dropback_epa",
    "off_rush_epa",
    "off_success",
    "off_late_down_epa",
    "off_explosive",
    "def_dropback_epa",
    "def_rush_epa",
    "def_success",
    "def_late_down_epa",
]

if not pbp_scrimmage.empty:
  # Aggregate team offensive performance
  off_stats = (
      pbp_scrimmage.groupby(["week", "posteam"])
      .agg(
          off_dropback_epa=(
              "epa",
              lambda x: (
                  x[pbp_scrimmage.loc[x.index, "play_type"] == "pass"].mean()
              ),
          ),
          off_rush_epa=(
              "epa",
              lambda x: (
                  x[pbp_scrimmage.loc[x.index, "play_type"] == "run"].mean()
              ),
          ),
          off_success=("success", "mean"),
          off_late_down_epa=(
              "epa",
              lambda x: (
                  x[pbp_scrimmage.loc[x.index, "is_late_down"] == 1].mean()
              ),
          ),
          off_explosive=("yards_gained", "std"),
      )
      .reset_index()
      .rename(columns={"posteam": "team"})
  )

  # Aggregate defensive performance allowed
  def_stats = (
      pbp_scrimmage.groupby(["week", "defteam"])
      .agg(
          def_dropback_epa=(
              "epa",
              lambda x: (
                  x[pbp_scrimmage.loc[x.index, "play_type"] == "pass"].mean()
              ),
          ),
          def_rush_epa=(
              "epa",
              lambda x: (
                  x[pbp_scrimmage.loc[x.index, "play_type"] == "run"].mean()
              ),
          ),
          def_success=("success", "mean"),
          def_late_down_epa=(
              "epa",
              lambda x: (
                  x[pbp_scrimmage.loc[x.index, "is_late_down"] == 1].mean()
              ),
          ),
      )
      .reset_index()
      .rename(columns={"defteam": "team"})
  )

  team_perf = pd.merge(
      off_stats, def_stats, on=["week", "team"], how="outer"
  ).fillna(0)
  team_perf.sort_values(["team", "week"], inplace=True)

  # Strictly lagged 6-game rolling window
  for col in metric_cols:
    team_perf[f"roll_{col}"] = team_perf.groupby("team")[col].transform(
        lambda x: x.shift(1).ewm(span=6, min_periods=1).mean()
    )
else:
  team_perf = pd.DataFrame(
      columns=["week", "team"] + [f"roll_{c}" for c in metric_cols]
  )

# 4. Filter Upcoming Unplayed Matchups
upcoming = schedules[schedules["result"].isna()].head(3).copy()

system_prompt = (
    "You are a quantitative sports handicapper. You are evaluating an upcoming"
    " NFL matchup using both Vegas market lines and a proprietary machine"
    " learning model's projected win probability. Compare the model's"
    " projection against the market line, highlight key situational and trench"
    " mismatches, and identify whether there is true betting value on the"
    " Spread or Total."
)

records = []
for _, game in upcoming.iterrows():
  home_team = game["home_team"]
  away_team = game["away_team"]
  matchup = f"{away_team} @ {home_team}"
  week_num = int(game["week"]) if pd.notna(game["week"]) else 1

  # Extract lagged rolling stats
  home_row = team_perf[
      (team_perf["team"] == home_team) & (team_perf["week"] == week_num)
  ]
  away_row = team_perf[
      (team_perf["team"] == away_team) & (team_perf["week"] == week_num)
  ]

  def get_metric(df, col_name, default=0.0):
    if not df.empty and pd.notna(df[col_name].values[0]):
      return float(df[col_name].values[0])
    return default

  # Situational & Contextual Features
  home_rest = float(game["home_rest"]) if pd.notna(game["home_rest"]) else 7.0
  away_rest = float(game["away_rest"]) if pd.notna(game["away_rest"]) else 7.0
  rest_diff = home_rest - away_rest
  is_divisional = (
      int(game["div_game"]) if pd.notna(game.get("div_game")) else 0
  )

  spread_line = (
      float(game["spread_line"]) if pd.notna(game["spread_line"]) else 0.0
  )
  total_line = (
      float(game["total_line"]) if pd.notna(game["total_line"]) else 44.0
  )

  # Market implied win probability based on closing spread
  market_home_prob = 1 / (1 + 10 ** (spread_line * 0.035))

  # Metric Differentials matching the 8-feature training model
  net_pass_edge = (
      get_metric(home_row, "roll_off_dropback_epa")
      - get_metric(away_row, "roll_def_dropback_epa")
  ) - (
      get_metric(away_row, "roll_off_dropback_epa")
      - get_metric(home_row, "roll_def_dropback_epa")
  )

  net_rush_edge = (
      get_metric(home_row, "roll_off_rush_epa")
      - get_metric(away_row, "roll_def_rush_epa")
  ) - (
      get_metric(away_row, "roll_off_rush_epa")
      - get_metric(home_row, "roll_def_rush_epa")
  )

  net_late_down_edge = (
      get_metric(home_row, "roll_off_late_down_epa")
      - get_metric(away_row, "roll_def_late_down_epa")
  ) - (
      get_metric(away_row, "roll_off_late_down_epa")
      - get_metric(home_row, "roll_def_late_down_epa")
  )

  diff_success = get_metric(
      home_row, "roll_off_success", 0.45
  ) - get_metric(away_row, "roll_off_success", 0.45)
  diff_explosive = get_metric(
      home_row, "roll_off_explosive", 8.0
  ) - get_metric(away_row, "roll_off_explosive", 8.0)

  # Build Feature Vector
  feature_row = pd.DataFrame(
      [[
          net_pass_edge,
          net_rush_edge,
          net_late_down_edge,
          diff_success,
          diff_explosive,
          rest_diff,
          is_divisional,
          market_home_prob,
      ]],
      columns=FEATURES,
  )

  # Model Prediction
  model_home_prob = float(model.predict_proba(feature_row)[0][1])

  print(
      f"{matchup} | Model Win Prob: {model_home_prob:.1%} | Market Implied:"
      f" {market_home_prob:.1%}"
  )

  # Payload for Gemini
  payload = {
      "matchup": matchup,
      "week": week_num,
      "spread_line": spread_line,
      "total_line": total_line,
      "market_home_win_probability": f"{market_home_prob:.1%}",
      "model_projections": {
          "home_win_probability": f"{model_home_prob:.1%}",
          "away_win_probability": f"{(1 - model_home_prob):.1%}",
          "probability_edge_vs_market": (
              f"{(model_home_prob - market_home_prob):+.1%}"
          ),
          "net_pass_edge": round(net_pass_edge, 4),
          "net_rush_edge": round(net_rush_edge, 4),
          "net_late_down_edge": round(net_late_down_edge, 4),
          "success_rate_edge": round(diff_success, 4),
      },
  }

  response = client.models.generate_content(
      model="gemini-3.6-flash",
      contents=json.dumps(payload),
      config=types.GenerateContentConfig(
          system_instruction=system_prompt, temperature=0.2
      ),
  )

  records.append({
      "game_id": str(game["game_id"]),
      "week": week_num,
      "matchup": matchup,
      "home_win_prob": model_home_prob,
      "market_prob": market_home_prob,
      "analysis": response.text,
  })

# 5. Overwrite / Insert Into Neon DB
if records:
  df_results = pd.DataFrame(records)
  df_results.to_sql(
      "nfl_weekly_analysis", engine, if_exists="append", index=False
  )
  print("Neon database updated successfully with model-backed predictions.")
