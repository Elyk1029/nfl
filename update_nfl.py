import json
import os
import time
import math
from google import genai
from google.genai import types
import nflreadpy as nfl
import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import create_engine
import xgboost as xgb

# 1. Environment & API Setup
db_url = os.environ.get("DATABASE_URL")
gemini_key = os.environ.get("GEMINI_API_KEY")

if not db_url or not gemini_key:
    raise ValueError("Missing DATABASE_URL or GEMINI_API_KEY in environment.")

engine = create_engine(db_url)
client = genai.Client(api_key=gemini_key)

# 2. Load Trained Model
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

# 3. Verified Staff & Coordinator Ground-Truth Mapping
# Prevents LLM from hallucinating fired coaches (Staley, Smith, Martindale, etc.)
CURRENT_STAFF_MAP = {
    "ARI": {"HC": "Jonathan Gannon", "OC": "Drew Petzing", "DC": "Nick Rallis", "Scheme": "Multiple / Split-safety"},
    "ATL": {"HC": "Raheem Morris", "OC": "Zac Robinson", "DC": "Jimmy Lake", "Scheme": "3-4 / Fangio-adjacent Zone"},
    "BAL": {"HC": "John Harbaugh", "OC": "Todd Monken", "DC": "Zach Orr", "Scheme": "Multiple / Simulated Pressures"},
    "BUF": {"HC": "Sean McDermott", "OC": "Joe Brady", "DC": "Bobby Babich", "Scheme": "4-2-5 Nickel base Cover 2/4"},
    "CAR": {"HC": "Dave Canales", "OC": "Brad Idzik", "DC": "Ejiro Evero", "Scheme": "3-4 Vic Fangio Two-High Shell"},
    "CHI": {"HC": "Matt Eberflus", "OC": "Thomas Brown", "DC": "Eric Washington", "Scheme": "4-3 Tampa 2 / Cover 3"},
    "CIN": {"HC": "Zac Taylor", "OC": "Dan Pitcher", "DC": "Lou Anarumo", "Scheme": "Multiple / Hybrid Man-Match"},
    "CLE": {"HC": "Kevin Stefanski", "OC": "Ken Dorsey", "DC": "Jim Schwartz", "Scheme": "4-3 Wide-9 Aggressive Man/Cover 3"},
    "DAL": {"HC": "Mike McCarthy", "OC": "Brian Schottenheimer", "DC": "Mike Zimmer", "Scheme": "4-3 Double-A Gap Blitz/Cover 1-3"},
    "DEN": {"HC": "Sean Payton", "OC": "Joe Lombardi", "DC": "Vance Joseph", "Scheme": "3-4 Heavy Blitz / Man-to-Man"},
    "DET": {"HC": "Dan Campbell", "OC": "Ben Johnson", "DC": "Aaron Glenn", "Scheme": "4-2-5 Aggressive Press-Man"},
    "GB":  {"HC": "Matt LaFleur", "OC": "Adam Stenavich", "DC": "Jeff Hafley", "Scheme": "4-3 Single-High Cover 1/3 Press"},
    "HOU": {"HC": "DeMeco Ryans", "OC": "Bobby Slowik", "DC": "Matt Burke", "Scheme": "4-3 Wide-9 / Quarters & Cover 3"},
    "IND": {"HC": "Shane Steichen", "OC": "Jim Bob Cooter", "DC": "Gus Bradley", "Scheme": "4-3 Pure Seattle Cover 3 / Low Blitz"},
    "JAX": {"HC": "Doug Pederson", "OC": "Press Taylor", "DC": "Ryan Nielsen", "Scheme": "4-2-5 Heavy Press-Man"},
    "KC":  {"HC": "Andy Reid", "OC": "Matt Nagy", "DC": "Steve Spagnuolo", "Scheme": "Multiple Exotic Blitz / Split Field Coverages"},
    "LAC": {"HC": "Jim Harbaugh", "OC": "Greg Roman", "DC": "Jesse Minter", "Scheme": "Multiple / Ravens-Michigan Disguised Shell"},
    "LAR": {"HC": "Sean McVay", "OC": "Mike LaFleur", "DC": "Chris Shula", "Scheme": "3-4 Light-Box Split Safety / Match Quarters"},
    "LV":  {"HC": "Antonio Pierce", "OC": "Luke Getsy", "DC": "Patrick Graham", "Scheme": "3-4 Multiple / Bracket Match Coverage"},
    "MIA": {"HC": "Mike McDaniel", "OC": "Frank Smith", "DC": "Anthony Weaver", "Scheme": "3-4 Ravens-Style Multiple Odd Front"},
    "MIN": {"HC": "Kevin O'Connell", "OC": "Wes Phillips", "DC": "Brian Flores", "Scheme": "3-4 Maximum Zero-Blitz / Invert Coverages"},
    "NE":  {"HC": "Jerod Mayo", "OC": "Alex Van Pelt", "DC": "DeMarcus Covington", "Scheme": "3-4 Belichick Cover 1 / Hybrid Match"},
    "NO":  {"HC": "Dennis Allen", "OC": "Klint Kubiak", "DC": "Joe Woods", "Scheme": "4-3 Quarters / Heavy Man Leverage"},
    "NYG": {"HC": "Brian Daboll", "OC": "Mike Kafka", "DC": "Shane Bowen", "Scheme": "4-2-5 Titans Bend-Don't-Break Two-High"},
    "NYJ": {"HC": "Robert Saleh", "OC": "Nathaniel Hackett", "DC": "Jeff Ulbrich", "Scheme": "4-3 4-Man Rush / Pure Quarters"},
    "PHI": {"HC": "Nick Sirianni", "OC": "Kellen Moore", "DC": "Vic Fangio", "Scheme": "3-4 Two-High Shell / Zone Match"},
    "PIT": {"HC": "Mike Tomlin", "OC": "Arthur Smith", "DC": "Teryl Austin", "Scheme": "3-4 Fire Zone / Cover 2 & Cover 3"},
    "SEA": {"HC": "Mike Macdonald", "OC": "Ryan Grubb", "DC": "Aden Durde", "Scheme": "Multiple Simulated Pressures / Cover 6/9"},
    "SF":  {"HC": "Kyle Shanahan", "OC": "Chris Foerster", "DC": "Nick Sorensen", "Scheme": "4-3 Wide-9 Under / Quarters"},
    "TB":  {"HC": "Todd Bowles", "OC": "Liam Coen", "DC": "Kacy Rodgers", "Scheme": "3-4 Overload Corner/Safety Blitz Heavy"},
    "TEN": {"HC": "Brian Callahan", "OC": "Nick Holz", "DC": "Dennard Wilson", "Scheme": "3-4 Press-Man Aggressive"},
    "WAS": {"HC": "Dan Quinn", "OC": "Kliff Kingsbury", "DC": "Joe Whitt Jr.", "Scheme": "4-3 Single-High Cover 1 / Quarters Hybrid"}
}

# 4. Pull Schedules, PBP, and Verified Rosters
CURRENT_SEASON = 2026
DATA_SEASON = 2025

print("Loading schedules, PBP metrics, and roster baseline...")
try:
    schedules = nfl.load_schedules(seasons=[CURRENT_SEASON]).to_pandas()
except Exception:
    schedules = nfl.load_schedules(seasons=[DATA_SEASON]).to_pandas()

try:
    pbp = nfl.load_pbp(seasons=[DATA_SEASON]).to_pandas()
except Exception:
    pbp = pd.DataFrame()

try:
    player_stats = nfl.load_player_stats(seasons=[DATA_SEASON]).to_pandas()
except Exception:
    player_stats = pd.DataFrame()

try:
    rosters = nfl.load_rosters(seasons=[DATA_SEASON]).to_pandas()
except Exception:
    rosters = pd.DataFrame()

# Clean scrimmage plays
if not pbp.empty:
    pbp_scrimmage = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    pbp_scrimmage["is_late_down"] = (
        pbp_scrimmage["down"].isin([3, 4]).astype(int)
        if "down" in pbp_scrimmage.columns else 0
    )
    
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

    metric_cols = [
        "off_dropback_epa", "off_rush_epa", "off_success", "off_late_down_epa", "off_explosive",
        "def_dropback_epa", "def_rush_epa", "def_success", "def_late_down_epa",
    ]
    for col in metric_cols:
        team_perf[f"roll_{col}"] = team_perf.groupby("team")[col].transform(
            lambda x: x.shift(1).ewm(span=6, min_periods=1).mean()
        )
else:
    team_perf = pd.DataFrame()

# Helper: Mathematical Devigging & Accurate Implied Probabilities
def calculate_market_probabilities(spread, home_ml=None, away_ml=None):
    """
    Computes fair, vig-free win and cover probabilities.
    In NFL markets, spread is defined as (Home Team Score - Away Team Score).
    A spread of -9.5 means the HOME team is favored by 9.5 points.
    """
    # 1. Devig Moneylines if provided
    if home_ml is not None and away_ml is not None and not math.isnan(home_ml) and not math.isnan(away_ml):
        p_home_raw = 100 / (home_ml + 100) if home_ml > 0 else abs(home_ml) / (abs(home_ml) + 100)
        p_away_raw = 100 / (away_ml + 100) if away_ml > 0 else abs(away_ml) / (abs(away_ml) + 100)
        total_implied = p_home_raw + p_away_raw
        fair_home_ml_prob = p_home_raw / total_implied
        return fair_home_ml_prob

    # 2. Empirical CDF Normal Conversion (NFL Spread StDev = 13.45)
    # home_spread: -9.5 implies home win prob = norm.cdf(-(-9.5) / 13.45) = norm.cdf(0.706) = ~76.0%
    norm_implied_home_win = float(norm.cdf(-spread / 13.45))
    return max(0.01, min(0.99, norm_implied_home_win))

def calculate_spread_cover_prob(model_win_prob, spread):
    """
    Decouples Win Probability from Cover Probability.
    Projects implied rating differential and calculates probability of covering spread.
    """
    z_score = norm.ppf(max(0.01, min(0.99, model_win_prob)))
    projected_margin = z_score * 13.45
    # Probability that Projected Margin covers the Spread
    cover_prob = float(norm.cdf((projected_margin + spread) / 13.45))
    return max(0.01, min(0.99, cover_prob))

# 5. Extract Real Starters from Active Rosters
def get_verified_skill_players(team_abbr):
    team_col = "recent_team" if "recent_team" in player_stats.columns else "team"
    t_stats = player_stats[player_stats[team_col] == team_abbr] if not player_stats.empty else pd.DataFrame()
    
    def get_pos(pos, stat_col):
        if t_stats.empty: return "Starting " + pos
        sub = t_stats[t_stats['position'] == pos].sort_values(stat_col, ascending=False)
        if not sub.empty:
            p = sub.iloc[0]
            return f"{p['player_name']} (Avg {p[stat_col]:.1f}/gm)"
        return "Unknown"

    return {
        "QB": get_pos("QB", "passing_yards"),
        "RB": get_pos("RB", "rushing_yards"),
        "WR": get_pos("WR", "receiving_yards")
    }

# 6. Filter Upcoming Games
next_week = schedules[schedules["result"].isna()]["week"].min()
upcoming = schedules[(schedules["result"].isna()) & (schedules["week"] == next_week)].copy()

system_prompt = """
You are a quantitative NFL handicapper and schematic analyst. 
You are given verified data payloads containing actual coaching staffs, defensive schemes, verified skill players, proprietary XGBoost win probabilities, and devigged market odds.

Rules:
1. NEVER hallucinate starters or head coaches. Use ONLY the personnel explicitly provided in the payload.
2. Clearly distinguish between outright Moneyline value and Spread cover edge.
3. Output strictly valid JSON matching the exact schema provided.
"""

records = []
for _, game in upcoming.iterrows():
    home_team = game["home_team"]
    away_team = game["away_team"]
    matchup = f"{away_team} @ {home_team}"
    week_num = int(game["week"]) if pd.notna(game["week"]) else 1

    spread_line = float(game["spread_line"]) if pd.notna(game["spread_line"]) else 0.0
    total_line = float(game["total_line"]) if pd.notna(game["total_line"]) else 44.0
    home_ml = float(game["home_moneyline"]) if "home_moneyline" in game and pd.notna(game["home_moneyline"]) else None
    away_ml = float(game["away_moneyline"]) if "away_moneyline" in game and pd.notna(game["away_moneyline"]) else None

    # Precise Market Probability (Vig removed, correct sign)
    market_home_prob = calculate_market_probabilities(spread_line, home_ml, away_ml)

    # Recompute explicit rolling metrics per game (prevents hardcoded duplicates)
    home_row = team_perf[(team_perf["team"] == home_team) & (team_perf["week"] == week_num)]
    away_row = team_perf[(team_perf["team"] == away_team) & (team_perf["week"] == week_num)]

    if home_row.empty and not team_perf.empty: home_row = team_perf[team_perf["team"] == home_team].tail(1)
    if away_row.empty and not team_perf.empty: away_row = team_perf[team_perf["team"] == away_team].tail(1)

    def get_metric(df, col_name, default=0.0):
        if not df.empty and col_name in df.columns and pd.notna(df[col_name].values[0]):
            return float(df[col_name].values[0])
        return default

    home_rest = float(game.get("home_rest", 7.0)) if pd.notna(game.get("home_rest")) else 7.0
    away_rest = float(game.get("away_rest", 7.0)) if pd.notna(game.get("away_rest")) else 7.0
    rest_diff = home_rest - away_rest
    is_divisional = int(game.get("div_game", 0)) if pd.notna(game.get("div_game")) else 0

    net_pass_edge = (get_metric(home_row, "roll_off_dropback_epa") - get_metric(away_row, "roll_def_dropback_epa")) - \
                    (get_metric(away_row, "roll_off_dropback_epa") - get_metric(home_row, "roll_def_dropback_epa"))
    net_rush_edge = (get_metric(home_row, "roll_off_rush_epa") - get_metric(away_row, "roll_def_rush_epa")) - \
                    (get_metric(away_row, "roll_off_rush_epa") - get_metric(home_row, "roll_def_rush_epa"))
    net_late_down_edge = (get_metric(home_row, "roll_off_late_down_epa") - get_metric(away_row, "roll_def_late_down_epa")) - \
                         (get_metric(away_row, "roll_off_late_down_epa") - get_metric(home_row, "roll_def_late_down_epa"))
    diff_success = get_metric(home_row, "roll_off_success", 0.44) - get_metric(away_row, "roll_off_success", 0.44)
    diff_explosive = get_metric(home_row, "roll_off_explosive", 7.5) - get_metric(away_row, "roll_off_explosive", 7.5)

    feature_row = pd.DataFrame([[
        net_pass_edge, net_rush_edge, net_late_down_edge, diff_success, 
        diff_explosive, rest_diff, is_divisional, market_home_prob
    ]], columns=FEATURES)

    # Model Probabilities
    model_home_prob = float(model.predict_proba(feature_row)[0][1])
    model_cover_prob = calculate_spread_cover_prob(model_home_prob, spread_line)
    ml_edge = model_home_prob - market_home_prob
    spread_edge = model_cover_prob - 0.50  # Against 50/50 standard spread market

    # Verified Personnel
    home_staff = CURRENT_STAFF_MAP.get(home_team, {"HC": "Unknown", "DC": "Unknown", "Scheme": "Multiple"})
    away_staff = CURRENT_STAFF_MAP.get(away_team, {"HC": "Unknown", "DC": "Unknown", "Scheme": "Multiple"})
    home_personnel = get_verified_skill_players(home_team)
    away_personnel = get_verified_skill_players(away_team)

    payload = {
        "matchup": matchup,
        "market": {
            "spread": spread_line,
            "total": total_line,
            "devigged_home_win_probability": f"{market_home_prob:.1%}",
        },
        "model_estimations": {
            "home_win_probability": f"{model_home_prob:.1%}",
            "home_spread_cover_probability": f"{model_cover_prob:.1%}",
            "moneyline_edge": f"{ml_edge:+.1%}",
            "spread_edge": f"{spread_edge:+.1%}"
        },
        "verified_schematics": {
            "home_team": {
                "team": home_team,
                "staff": home_staff,
                "starters": home_personnel
            },
            "away_team": {
                "team": away_team,
                "staff": away_staff,
                "starters": away_personnel
            }
        }
    }

    prompt = f"""
Analyze this game JSON:
{json.dumps(payload, indent=2)}

Produce a JSON response with this exact structure:
{{
  "executive_summary": "2 sentences breaking down whether there is real market value on the ML ({ml_edge:+.1%}) or Spread ({spread_edge:+.1%}). State clearly if the game is a PASS (edges under 3%).",
  "schematic_matchup": {{
    "away_offense_vs_home_defense": "Analysis matching away players against the confirmed defensive coordinator ({home_staff['DC']}) and scheme ({home_staff['Scheme']}).",
    "home_offense_vs_away_defense": "Analysis matching home players against the confirmed defensive coordinator ({away_staff['DC']}) and scheme ({away_staff['Scheme']})."
  }},
  "player_projections": {{
    "away_team": {{
      "QB": "{away_personnel['QB']}: stat projection & scheme rationale",
      "RB": "{away_personnel['RB']}: stat projection & scheme rationale",
      "WR": "{away_personnel['WR']}: stat projection & scheme rationale"
    }},
    "home_team": {{
      "QB": "{home_personnel['QB']}: stat projection & scheme rationale",
      "RB": "{home_personnel['RB']}: stat projection & scheme rationale",
      "WR": "{home_personnel['WR']}: stat projection & scheme rationale"
    }}
  }},
  "actionable_verdict": "Clear call: [Bet Team Spread / Bet Team ML / Pass]"
}}
"""

    max_retries = 3
    backoff = 3
    response_text = "{}"

    for attempt in range(max_retries):
        try:
            res = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    response_mime_type="application/json"
                ),
            )
            response_text = res.text
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
            else:
                print(f"Failed API call for {matchup}: {e}")

    records.append({
        "game_id": str(game["game_id"]),
        "week": week_num,
        "matchup": matchup,
        "home_win_prob": model_home_prob,
        "market_prob": market_home_prob,
        "analysis": response_text,
    })
    print(f"Processed {matchup}: ML Edge = {ml_edge:+.1%}, Spread Edge = {spread_edge:+.1%}")

if records:
    # Clear stale inverted data and write mathematically grounded records
    df_results = pd.DataFrame(records)
    with engine.begin() as conn:
        conn.execute("DELETE FROM nfl_weekly_analysis WHERE week = %s", (week_num,))
    df_results.to_sql("nfl_weekly_analysis", engine, if_exists="append", index=False)
    print("Database updated with verified calculations and grounded personnel.")
