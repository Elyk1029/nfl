import json
import math
import os
import time
from google import genai
from google.genai import types
import nflreadpy as nfl
import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import create_engine, text
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

NFL_KEY_PUSH_RATES = {
    3: 0.148,
    7: 0.094,
    6: 0.059,
    10: 0.057,
    4: 0.052,
    14: 0.046,
    1: 0.038,
    2: 0.036
}

# Standardize NFL team abbreviations
TEAM_ABBR_MAP = {
    "LAR": "LA",
    "WSH": "WAS",
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "JAC": "JAX"
}

def clean_team_abbr(team_str):
    if not isinstance(team_str, str):
        return team_str
    team_str = team_str.strip().upper()
    return TEAM_ABBR_MAP.get(team_str, team_str)

BASE_STAFF_MAP = {
    "ARI": {"HC": "Mike LaFleur", "OC": "Nathaniel Hackett", "DC": "Nick Rallis", "Scheme": "Multiple / Split-safety"},
    "ATL": {"HC": "Kevin Stefanski", "OC": "Tommy Rees", "DC": "Jimmy Lake", "Scheme": "3-4 / Fangio-adjacent Zone"},
    "BAL": {"HC": "Jesse Minter", "OC": "Declan Doyle", "DC": "Anthony Weaver", "Scheme": "Multiple / Ravens Disguised Shell"},
    "BUF": {"HC": "Joe Brady", "OC": "Pete Carmichael Jr.", "DC": "Jim Leonhard", "Scheme": "4-2-5 Nickel Base Cover 2/4"},
    "CAR": {"HC": "Dave Canales", "OC": "Brad Idzik", "DC": "Ejiro Evero", "Scheme": "3-4 Vic Fangio Two-High Shell"},
    "CHI": {"HC": "Matt Eberflus", "OC": "Press Taylor", "DC": "Eric Washington", "Scheme": "4-3 Tampa 2 / Cover 3"},
    "CIN": {"HC": "Zac Taylor", "OC": "Dan Pitcher", "DC": "Lou Anarumo", "Scheme": "Multiple / Hybrid Man-Match"},
    "CLE": {"HC": "Todd Monken", "OC": "Travis Switzer", "DC": "Mike Rutenberg", "Scheme": "4-3 Wide-9 Aggressive Man/Cover 3"},
    "DAL": {"HC": "Brian Schottenheimer", "OC": "Scott Tolzien", "DC": "Mike Zimmer", "Scheme": "4-3 Double-A Gap Blitz/Cover 1-3"},
    "DEN": {"HC": "Sean Payton", "OC": "Davis Webb", "DC": "Vance Joseph", "Scheme": "3-4 Heavy Blitz / Man-to-Man"},
    "DET": {"HC": "Dan Campbell", "OC": "Drew Petzing", "DC": "Aaron Glenn", "Scheme": "4-2-5 Aggressive Press-Man"},
    "GB":  {"HC": "Matt LaFleur", "OC": "Adam Stenavich", "DC": "Jonathan Gannon", "Scheme": "4-3 Single-High Cover 1/3 Press"},
    "HOU": {"HC": "DeMeco Ryans", "OC": "Bobby Slowik", "DC": "Matt Burke", "Scheme": "4-3 Wide-9 / Quarters & Cover 3"},
    "IND": {"HC": "Shane Steichen", "OC": "Jim Bob Cooter", "DC": "Gus Bradley", "Scheme": "4-3 Pure Seattle Cover 3 / Low Blitz"},
    "JAX": {"HC": "Doug Pederson", "OC": "Mike McCoy", "DC": "Ryan Nielsen", "Scheme": "4-2-5 Heavy Press-Man"},
    "KC":  {"HC": "Andy Reid", "OC": "Eric Bieniemy", "DC": "Steve Spagnuolo", "Scheme": "Multiple Exotic Blitz / Split Field Coverages"},
    "LAC": {"HC": "Jim Harbaugh", "OC": "Mike McDaniel", "DC": "Chris O'Leary", "Scheme": "Multiple / Disguised Shell"},
    "LA":  {"HC": "Sean McVay", "OC": "Nathan Scheelhaase", "DC": "Chris Shula", "Scheme": "3-4 Light-Box Split Safety / Match Quarters"},
    "LV":  {"HC": "Klint Kubiak", "OC": "Andrew Janocko", "DC": "Rob Leonard", "Scheme": "3-4 Multiple / Bracket Match Coverage"},
    "MIA": {"HC": "Jeff Hafley", "OC": "Bobby Slowik", "DC": "Sean Duggan", "Scheme": "3-4 Multiple Odd Front / Press-Zone"},
    "MIN": {"HC": "Kevin O'Connell", "OC": "Wes Phillips", "DC": "Brian Flores", "Scheme": "3-4 Maximum Zero-Blitz / Invert Coverages"},
    "NE":  {"HC": "Jerod Mayo", "OC": "Alex Van Pelt", "DC": "Zak Kuhr", "Scheme": "3-4 Belichick Cover 1 / Hybrid Match"},
    "NO":  {"HC": "Dennis Allen", "OC": "Frank Reich", "DC": "Joe Woods", "Scheme": "4-3 Quarters / Heavy Man Leverage"},
    "NYG": {"HC": "John Harbaugh", "OC": "Matt Nagy", "DC": "Dennard Wilson", "Scheme": "4-2-5 Two-High Zone Shell"},
    "NYJ": {"HC": "Robert Saleh", "OC": "Nathaniel Hackett", "DC": "Jeff Ulbrich", "Scheme": "4-3 4-Man Rush / Pure Quarters"},
    "PHI": {"HC": "Nick Sirianni", "OC": "Sean Mannion", "DC": "Vic Fangio", "Scheme": "3-4 Two-High Shell / Zone Match"},
    "PIT": {"HC": "Mike McCarthy", "OC": "Brian Angelichio", "DC": "Patrick Graham", "Scheme": "3-4 Fire Zone / Cover 2 & Cover 3"},
    "SEA": {"HC": "Mike Macdonald", "OC": "Brian Fleury", "DC": "Aden Durde", "Scheme": "Multiple Simulated Pressures / Cover 6/9"},
    "SF":  {"HC": "Kyle Shanahan", "OC": "Chris Foerster", "DC": "Raheem Morris", "Scheme": "4-3 Wide-9 Under / Quarters"},
    "TB":  {"HC": "Todd Bowles", "OC": "Josh Grizzard", "DC": "Kacy Rodgers", "Scheme": "3-4 Overload Corner/Safety Blitz Heavy"},
    "TEN": {"HC": "Robert Saleh", "OC": "Brian Daboll", "DC": "Gus Bradley", "Scheme": "4-3 Aggressive Front / Single-High & Quarters"},
    "WAS": {"HC": "Dan Quinn", "OC": "David Blough", "DC": "Joe Whitt Jr.", "Scheme": "4-3 Single-High Cover 1 / Quarters Hybrid"}
}

# Duplicate aliases for lookup safety
CURRENT_STAFF_MAP = {**BASE_STAFF_MAP}
CURRENT_STAFF_MAP["LAR"] = BASE_STAFF_MAP["LA"]
CURRENT_STAFF_MAP["WSH"] = BASE_STAFF_MAP["WAS"]
CURRENT_STAFF_MAP["JAC"] = BASE_STAFF_MAP["JAX"]

# 3. Ingestion Pipeline
CURRENT_SEASON = 2026
DATA_SEASON = 2025

print("Loading NFL datasets...")
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
    injuries = nfl.load_injuries(seasons=[CURRENT_SEASON]).to_pandas()
except Exception:
    try:
        injuries = nfl.load_injuries(seasons=[DATA_SEASON]).to_pandas()
    except Exception:
        injuries = pd.DataFrame()

# Clean team abbreviations across all ingested data
if not schedules.empty:
    schedules["home_team"] = schedules["home_team"].apply(clean_team_abbr)
    schedules["away_team"] = schedules["away_team"].apply(clean_team_abbr)

if not pbp.empty:
    pbp["posteam"] = pbp["posteam"].apply(clean_team_abbr)
    pbp["defteam"] = pbp["defteam"].apply(clean_team_abbr)

if not player_stats.empty:
    team_col = "recent_team" if "recent_team" in player_stats.columns else "team"
    player_stats[team_col] = player_stats[team_col].apply(clean_team_abbr)

if not injuries.empty and "team" in injuries.columns:
    injuries["team"] = injuries["team"].apply(clean_team_abbr)

# Clean Garbage Time EPA
if not pbp.empty:
    pbp_clean = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    if "home_wp" in pbp_clean.columns and "qtr" in pbp_clean.columns:
        leverage_mask = (pbp_clean["qtr"] <= 2) | (pbp_clean["home_wp"].between(0.05, 0.95))
        pbp_clean = pbp_clean[leverage_mask]

    pbp_clean["is_late_down"] = pbp_clean["down"].isin([3, 4]).astype(int) if "down" in pbp_clean.columns else 0

    off_stats = pbp_clean.groupby(["week", "posteam"]).agg(
        off_dropback_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "play_type"] == "pass"].mean()),
        off_rush_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "play_type"] == "run"].mean()),
        off_success=("success", "mean"),
        off_late_down_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "is_late_down"] == 1].mean()),
        off_explosive=("yards_gained", "std"),
    ).reset_index().rename(columns={"posteam": "team"})

    def_stats = pbp_clean.groupby(["week", "defteam"]).agg(
        def_dropback_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "play_type"] == "pass"].mean()),
        def_rush_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "play_type"] == "run"].mean()),
        def_success=("success", "mean"),
        def_late_down_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "is_late_down"] == 1].mean()),
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

# 4. Odds, Edge Calibration & Risk Formulas
def get_devigged_market_home_prob(spread_line, home_ml=None, away_ml=None):
    if home_ml is not None and away_ml is not None and not math.isnan(home_ml) and not math.isnan(away_ml):
        p_home = 100.0 / (home_ml + 100.0) if home_ml > 0 else abs(home_ml) / (abs(home_ml) + 100.0)
        p_away = 100.0 / (away_ml + 100.0) if away_ml > 0 else abs(away_ml) / (abs(away_ml) + 100.0)
        tot = p_home + p_away
        if tot > 0:
            return float(p_home / tot)
    return float(norm.cdf(spread_line / 13.8))

def calibrate_model_probability(raw_model_prob, market_prob, market_weight=0.70):
    """
    Bayesian shrinkage anchor (70% market weight) to prevent extreme Week 1 edge outliers.
    """
    return float((1.0 - market_weight) * raw_model_prob + market_weight * market_prob)

def calculate_spread_cover_distribution(calibrated_home_win_prob, market_home_prob, spread_line):
    """
    Calculates empirical cover probability based on model vs market edge.
    Prevents double-normal CDF tail inflation on wide spreads.
    """
    abs_spread = round(abs(spread_line))
    push_rate = NFL_KEY_PUSH_RATES.get(abs_spread, 0.015) if float(spread_line).is_integer() else 0.0
    
    # Delta of model expectation vs consensus market win probability
    prob_delta = calibrated_home_win_prob - market_home_prob
    
    # Each 1.0% win probability advantage translates to ~0.55% spread cover advantage
    home_cover_prob = 0.50 + (prob_delta * 0.55)
    
    # Adjust for push likelihood
    home_cover_prob = home_cover_prob * (1.0 - push_rate)
    away_cover_prob = (1.0 - home_cover_prob) * (1.0 - push_rate)
    
    # Realistic operational bounds [35%, 65%]
    home_cover_prob = max(0.35, min(0.65, home_cover_prob))
    away_cover_prob = max(0.35, min(0.65, away_cover_prob))
    
    # Invariant: Favorite cover probability cannot exceed its outright win probability
    if spread_line > 0:
        home_cover_prob = min(home_cover_prob, calibrated_home_win_prob)
    elif spread_line < 0:
        away_cover_prob = min(away_cover_prob, 1.0 - calibrated_home_win_prob)
        
    return float(home_cover_prob), float(away_cover_prob), float(push_rate)

def calculate_quarter_kelly(prob_win, decimal_odds=1.9091, max_cap=2.00):
    if prob_win <= 0.5238:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - prob_win
    raw_kelly = (b * prob_win - q) / b
    fractional = raw_kelly * 0.25 * 100.0  # Quarter-Kelly
    return round(float(min(max_cap, max(0.0, fractional))), 2)

# 5. Player Stat Sanitizer (Excludes Retired / Historical Players)
EXCLUDED_HISTORICAL_PLAYERS = {
    "Aaron Donald", "Tom Brady", "J.J. Watt", "Rob Gronkowski", 
    "Drew Brees", "Matt Ryan", "Ben Roethlisberger", "Jason Kelce"
}

def get_sanitized_player_baselines(team_abbr):
    team_col = "recent_team" if "recent_team" in player_stats.columns else "team"
    t_stats = player_stats[player_stats[team_col] == team_abbr] if not player_stats.empty and team_col in player_stats.columns else pd.DataFrame()
    
    scratches = []
    if not injuries.empty and "team" in injuries.columns:
        t_inj = injuries[(injuries["team"] == team_abbr) & (injuries["report_status"].isin(["Out", "Doubtful", "IR"]))]
        if "full_name" in t_inj.columns:
            scratches = [p for p in t_inj["full_name"].dropna().unique().tolist() if p not in EXCLUDED_HISTORICAL_PLAYERS][:6]

    def get_pos(pos, stat_col, min_val, max_val, unit_str):
        if t_stats.empty or stat_col not in t_stats.columns:
            return f"Starting {pos} (~0.0 {unit_str}/gm)"
            
        sub = t_stats[(t_stats['position'] == pos) & (~t_stats['player_name'].isin(scratches + list(EXCLUDED_HISTORICAL_PLAYERS)))].copy()
        if not sub.empty:
            grouped = sub.groupby('player_name').agg({stat_col: 'mean'}).reset_index()
            grouped = grouped.sort_values(stat_col, ascending=False)
            top = grouped.iloc[0]
            val = float(top[stat_col])
            sanitized_val = max(min_val, min(max_val, val))
            return f"{top['player_name']} (~{sanitized_val:.1f} {unit_str}/gm)"
        return f"Starting {pos} (~0.0 {unit_str}/gm)"

    return {
        "starters": {
            "QB": get_pos("QB", "passing_yards", 120.0, 310.0, "pass yds"),
            "RB": get_pos("RB", "rushing_yards", 25.0, 115.0, "rush yds"),
            "WR": get_pos("WR", "receiving_yards", 20.0, 105.0, "rec yds")
        },
        "scratches": scratches if scratches else ["None Reported"]
    }

# 6. Main Execution Loop
upcoming = pd.DataFrame()
target_week = 1

if not schedules.empty:
    unplayed = schedules[schedules["result"].isna()]
    if not unplayed.empty:
        target_week = int(unplayed["week"].min())
        upcoming = unplayed[unplayed["week"] == target_week].copy()

records = []
system_prompt = """
You are an institutional NFL sports betting syndicate analyst.
Synthesize the provided quantitative edges, key number push rates, confirmed schemes, and inactive scratches.

Strict Constraints:
1. Reference ONLY explicitly confirmed starters and coaches provided in the payload.
2. If the recommendation is PASS, state PASS clearly with 0.00u sizing.
3. If an edge exists on a team, the recommendation must match the exact team and Vegas line provided in the payload.
4. Output strictly valid JSON matching the exact schema.
"""

for _, game in upcoming.iterrows():
    home_team = clean_team_abbr(str(game["home_team"]))
    away_team = clean_team_abbr(str(game["away_team"]))
    matchup = f"{away_team} @ {home_team}"
    week_num = int(game["week"]) if pd.notna(game["week"]) else target_week

    spread_line = float(game["spread_line"]) if pd.notna(game.get("spread_line")) else 0.0
    total_line = float(game["total_line"]) if pd.notna(game.get("total_line")) else 44.0
    home_ml = float(game["home_moneyline"]) if pd.notna(game.get("home_moneyline")) else None
    away_ml = float(game["away_moneyline"]) if pd.notna(game.get("away_moneyline")) else None

    market_home_prob = get_devigged_market_home_prob(spread_line, home_ml, away_ml)

    home_row = team_perf[(team_perf["team"] == home_team) & (team_perf["week"] == week_num)] if not team_perf.empty else pd.DataFrame()
    away_row = team_perf[(team_perf["team"] == away_team) & (team_perf["week"] == week_num)] if not team_perf.empty else pd.DataFrame()

    if home_row.empty and not team_perf.empty: home_row = team_perf[team_perf["team"] == home_team].tail(1)
    if away_row.empty and not team_perf.empty: away_row = team_perf[team_perf["team"] == away_team].tail(1)

    def get_metric(df, col_name, default=0.0):
        if not df.empty and col_name in df.columns and pd.notna(df[col_name].values[0]):
            return float(df[col_name].values[0])
        return float(default)

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

    raw_model_home_prob = float(model.predict_proba(feature_row)[0][1])
    
    # Bayesian market shrinkage (70% anchor)
    calibrated_home_win_prob = calibrate_model_probability(raw_model_home_prob, market_home_prob, market_weight=0.70)

    # Calculate cover distribution
    home_cover_prob, away_cover_prob, push_prob = calculate_spread_cover_distribution(
        calibrated_home_win_prob, market_home_prob, spread_line
    )
    
    # Standard -110 break-even: 52.38%
    home_spread_edge = home_cover_prob - 0.5238
    away_spread_edge = away_cover_prob - 0.5238
    
    home_line_formatted = f"{home_team} {-spread_line:+g}"
    away_line_formatted = f"{away_team} {+spread_line:+g}"
    
    # 2.0% minimum threshold with a realistic 5.0% display cap
    if home_spread_edge > 0.02 and home_spread_edge > away_spread_edge:
        recommended_team = home_team
        recommended_line = home_line_formatted
        chosen_cover_prob = home_cover_prob
        chosen_edge = min(0.050, home_spread_edge)
        kelly_units = calculate_quarter_kelly(home_cover_prob)
    elif away_spread_edge > 0.02 and away_spread_edge > home_spread_edge:
        recommended_team = away_team
        recommended_line = away_line_formatted
        chosen_cover_prob = away_cover_prob
        chosen_edge = min(0.050, away_spread_edge)
        kelly_units = calculate_quarter_kelly(away_cover_prob)
    else:
        recommended_team = "PASS"
        recommended_line = "No Value"
        chosen_cover_prob = max(home_cover_prob, away_cover_prob)
        chosen_edge = max(home_spread_edge, away_spread_edge)
        kelly_units = 0.00

    home_ctx = get_sanitized_player_baselines(home_team)
    away_ctx = get_sanitized_player_baselines(away_team)
    
    home_staff = CURRENT_STAFF_MAP.get(home_team, {"HC": "Head Coach", "DC": "Defensive Coordinator", "Scheme": "Nickel 4-2-5 Base"})
    away_staff = CURRENT_STAFF_MAP.get(away_team, {"HC": "Head Coach", "DC": "Defensive Coordinator", "Scheme": "Nickel 4-2-5 Base"})

    payload = {
        "matchup": matchup,
        "market": {
            "vegas_spread": f"{home_team} {-spread_line:+g}",
            "total": total_line,
            "devigged_home_ml_prob": f"{market_home_prob:.1%}"
        },
        "model_calculations": {
            "calibrated_home_win_prob": f"{calibrated_home_win_prob:.1%}",
            "home_cover_prob": f"{home_cover_prob:.1%}",
            "away_cover_prob": f"{away_cover_prob:.1%}",
            "home_edge_vs_juice": f"{home_spread_edge:+.1%}",
            "away_edge_vs_juice": f"{away_spread_edge:+.1%}",
            "recommended_side": recommended_team,
            "recommended_line": recommended_line,
            "suggested_kelly_units": f"{kelly_units:.2f}u"
        },
        "schematics": {
            "home_team": {
                "team": home_team,
                "staff": home_staff,
                "starters": home_ctx["starters"],
                "injuries": home_ctx["scratches"]
            },
            "away_team": {
                "team": away_team,
                "staff": away_staff,
                "starters": away_ctx["starters"],
                "injuries": away_ctx["scratches"]
            }
        }
    }

    prompt = f"""
Analyze this game JSON:
{json.dumps(payload, indent=2)}

Output strictly valid JSON with this schema:
{{
  "executive_summary": "State whether this game is a BET ({recommended_line} at {chosen_edge:+.1%} edge) or a PASS.",
  "schematic_matchup": {{
    "away_offense_vs_home_defense": "Analysis matching away players against DC {home_staff['DC']} and scheme {home_staff['Scheme']}.",
    "home_offense_vs_away_defense": "Analysis matching home players against DC {away_staff['DC']} and scheme {away_staff['Scheme']}."
  }},
  "player_projections": {{
    "away_team": {{
      "QB": "{away_ctx['starters']['QB']}: analysis",
      "RB": "{away_ctx['starters']['RB']}: analysis",
      "WR": "{away_ctx['starters']['WR']}: analysis"
    }},
    "home_team": {{
      "QB": "{home_ctx['starters']['QB']}: analysis",
      "RB": "{home_ctx['starters']['RB']}: analysis",
      "WR": "{home_ctx['starters']['WR']}: analysis"
    }}
  }},
  "actionable_verdict": "{'PASS - 0.00u' if recommended_team == 'PASS' else 'Bet ' + recommended_line + ' - ' + str(kelly_units) + 'u'}"
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
                print(f"API error on {matchup}: {e}")

    records.append({
        "game_id": str(game.get("game_id", f"2026_{week_num}_{away_team}_{home_team}")),
        "week": int(week_num),
        "matchup": str(matchup),
        "home_win_prob": float(calibrated_home_win_prob),
        "market_prob": float(market_home_prob),
        "spread_cover_prob": float(chosen_cover_prob),
        "spread_edge": float(chosen_edge),
        "kelly_units": float(kelly_units),
        "analysis": str(response_text),
    })
    print(f"Processed {matchup} | Pick: {recommended_line} | Edge: {chosen_edge:+.1%} | Kelly: {kelly_units}u")

# 7. Database Sync & Auto-Migration
if records:
    df_results = pd.DataFrame(records)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS nfl_weekly_analysis (
                game_id TEXT PRIMARY KEY,
                week INTEGER,
                matchup TEXT,
                home_win_prob NUMERIC,
                market_prob NUMERIC,
                spread_cover_prob NUMERIC,
                spread_edge NUMERIC,
                kelly_units NUMERIC,
                analysis TEXT
            );
        """))
        conn.execute(text("""
            ALTER TABLE nfl_weekly_analysis 
            ADD COLUMN IF NOT EXISTS spread_cover_prob NUMERIC,
            ADD COLUMN IF NOT EXISTS spread_edge NUMERIC,
            ADD COLUMN IF NOT EXISTS kelly_units NUMERIC;
        """))
        conn.execute(
            text("DELETE FROM nfl_weekly_analysis WHERE week = :target_week"),
            {"target_week": target_week}
        )
    df_results.to_sql("nfl_weekly_analysis", engine, if_exists="append", index=False)
    print(f"Database updated for Week {target_week} with {len(df_results)} calibrated records.")
