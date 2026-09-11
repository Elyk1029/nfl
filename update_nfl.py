"""
update_nfl.py - Pipeline Orchestrator with Opponent-Adjusted EPA, VORP, and Median Conversions.
"""
import asyncio
import json
import math
import os
import sys
from google import genai
from google.genai import types
import nflreadpy as nfl
import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import create_engine, text
import xgboost as xgb

from verifier import NFLDataVerifier

# 1. Environment Verification & Client Initialization
db_url = os.environ.get("DATABASE_URL")
gemini_key = os.environ.get("GEMINI_API_KEY")

if not db_url or not gemini_key:
    raise ValueError("FATAL: DATABASE_URL and GEMINI_API_KEY must be configured.")

engine = create_engine(db_url, pool_size=5, pool_pre_ping=True)
client = genai.Client(api_key=gemini_key)

MODEL_FILE = "nfl_model.json"
model = xgb.XGBClassifier()
if os.path.exists(MODEL_FILE):
    model.load_model(MODEL_FILE)
    print("XGBoost classifier loaded successfully.")
else:
    raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found in root directory.")

# Fundamental feature vector (Collinear market probability stripped to avoid double-shrinkage)
FEATURES = [
    "net_pass_edge", "net_rush_edge", "net_late_down_edge", "diff_success",
    "diff_explosive", "rest_diff", "is_divisional",
]

NFL_KEY_PUSH_RATES = {
    3: 0.148, 7: 0.094, 6: 0.059, 10: 0.057, 4: 0.052, 14: 0.046, 1: 0.038, 2: 0.036
}

TEAM_ABBR_MAP = {
    "LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX"
}

def clean_team_abbr(team_str):
    if not isinstance(team_str, str):
        return team_str
    cleaned = team_str.strip().upper()
    return TEAM_ABBR_MAP.get(cleaned, cleaned)

# 2. Ingestion with Fallback Safety
CURRENT_SEASON = 2026
DATA_SEASON = 2025

print("Ingesting schedules, rosters, and live depth charts...")
try:
    schedules = nfl.load_schedules(seasons=[CURRENT_SEASON]).to_pandas()
except Exception:
    schedules = nfl.load_schedules(seasons=[DATA_SEASON]).to_pandas()

try:
    pbp = nfl.load_pbp(seasons=[DATA_SEASON, CURRENT_SEASON]).to_pandas()
except Exception:
    try:
        pbp = nfl.load_pbp(seasons=[DATA_SEASON]).to_pandas()
    except Exception:
        pbp = pd.DataFrame()

try:
    player_stats = nfl.load_player_stats(seasons=[DATA_SEASON, CURRENT_SEASON]).to_pandas()
except Exception:
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

try:
    depth_charts = nfl.load_depth_charts(seasons=[CURRENT_SEASON]).to_pandas()
except Exception:
    try:
        depth_charts = nfl.load_depth_charts(seasons=[DATA_SEASON]).to_pandas()
    except Exception:
        depth_charts = pd.DataFrame()

for df in [schedules, pbp, player_stats, injuries, depth_charts]:
    if df.empty:
        continue
    for col in ["home_team", "away_team", "posteam", "defteam", "recent_team", "team", "club_code"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_team_abbr)

# 3. Opponent-Adjusted EPA & Leverage Filtering
def compute_opponent_adjusted_epa(pbp_df):
    if pbp_df.empty:
        return pd.DataFrame()

    pbp_clean = pbp_df[pbp_df["play_type"].isin(["pass", "run"])].copy()
    
    # Strictly filter non-garbage time leverage (WP 10% - 90%, exclude late blowouts)
    if "home_wp" in pbp_clean.columns and "qtr" in pbp_clean.columns:
        leverage_mask = (pbp_clean["qtr"] <= 3) | (pbp_clean["home_wp"].between(0.10, 0.90))
        pbp_clean = pbp_clean[leverage_mask]

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

    merged = pd.merge(off_stats, def_stats, on=["season", "week", "team"], how="outer").fillna(0)
    merged.sort_values(["team", "season", "week"], inplace=True)

    metric_cols = [
        "off_dropback_epa", "off_rush_epa", "off_early_down_success", "off_late_down_epa", "off_explosive",
        "def_dropback_epa", "def_rush_epa", "def_early_down_success", "def_late_down_epa",
    ]
    for col in metric_cols:
        merged[f"roll_{col}"] = merged.groupby("team")[col].transform(
            lambda x: x.shift(1).ewm(span=6, min_periods=1).mean()
        )
    return merged

team_perf = compute_opponent_adjusted_epa(pbp)

def get_latest_team_row(team_abbr, target_season, target_week):
    if team_perf.empty:
        return pd.DataFrame()
    t_data = team_perf[
        (team_perf["team"] == team_abbr) &
        ((team_perf["season"] < target_season) |
         ((team_perf["season"] == target_season) & (team_perf["week"] < target_week)))
    ]
    return t_data.tail(1) if not t_data.empty else pd.DataFrame()

# 4. VORP Injury Delta Haircuts
VORP_PENALTIES = {"QB1": 0.22, "LT1": 0.05, "EDGE1": 0.04}

def calculate_roster_vorp_penalty(team_abbr, injuries_df, depth_charts_df):
    if injuries_df.empty or depth_charts_df.empty:
        return 0.0

    t_inj = injuries_df[(injuries_df["team"] == team_abbr) & (injuries_df["report_status"].isin(["Out", "Doubtful", "IR"]))]
    if t_inj.empty:
        return 0.0

    inj_names = t_inj["player_name"].dropna().tolist() if "player_name" in t_inj else []
    t_dc = depth_charts_df[depth_charts_df["club_code"] == team_abbr] if "club_code" in depth_charts_df else pd.DataFrame()

    total_pass_penalty = 0.0
    if not t_dc.empty and "player_name" in t_dc:
        for role, penalty in VORP_PENALTIES.items():
            pos = role[:2]
            pos_match = t_dc[(t_dc["pos_abb"] == pos) & (t_dc["pos_rank"].astype(str) == "1")]
            if not pos_match.empty:
                starter_name = pos_match.iloc[0]["player_name"]
                if starter_name in inj_names:
                    total_pass_penalty += penalty

    return total_pass_penalty

# 5. Tactical Archetype Profiler
def extract_team_tactical_archetypes(pbp_df, team_abbr):
    if pbp_df.empty:
        return {
            "backfield_structure": "Standard Tandem (55/35)",
            "target_distribution": "Balanced Target Tree",
            "qb_operating_profile": "Rhythm Pocket Passer"
        }

    t_plays = pbp_df[(pbp_df["posteam"] == team_abbr) | (pbp_df["defteam"] == team_abbr)]
    off_runs = t_plays[(t_plays["posteam"] == team_abbr) & (t_plays["play_type"] == "run")]
    
    rbs = off_runs.groupby("rusher_player_id")["epa"].count().sort_values(ascending=False)
    rb1_share = (rbs.iloc[0] / rbs.sum()) if not rbs.empty and rbs.sum() > 0 else 0.50
    if rb1_share >= 0.70:
        backfield_dna = "Workhorse Bellcow (>70% touch monopoly; do not force committee)"
    elif rb1_share >= 0.52:
        backfield_dna = "1A/1B Tandem (55/35 touch rotation)"
    else:
        backfield_dna = "Full Multi-Back Committee (Hot-hand approach)"

    off_pass = t_plays[(t_plays["posteam"] == team_abbr) & (t_plays["play_type"] == "pass")]
    targets = off_pass.groupby("receiver_player_id")["epa"].count().sort_values(ascending=False)
    top2_share = (targets.iloc[:2].sum() / targets.sum()) if len(targets) >= 2 and targets.sum() > 0 else 0.40
    target_dna = "Target Funnel (Top 2 options command >50% target tree)" if top2_share >= 0.48 else "Distributed Scheme (Ball spreads across 5+ options)"

    scrambles = off_pass["qb_scramble"].mean() if "qb_scramble" in off_pass.columns else 0.04
    qb_dna = "Dual-Threat Play-Extender (Pressure creates scrambles; do not collapse YPA)" if scrambles >= 0.08 else "Pocket Rhythm Passer (Negative PBWR causes YPA collapse)"

    return {
        "backfield_structure": backfield_dna,
        "target_distribution": target_dna,
        "qb_operating_profile": qb_dna
    }

# 6. Point Spread Modeling & 3-Outcome Kelly Sizing
def get_devigged_market_home_prob(spread_line, home_ml=None, away_ml=None):
    if home_ml is not None and away_ml is not None and not math.isnan(home_ml) and not math.isnan(away_ml):
        p_home = 100.0 / (home_ml + 100.0) if home_ml > 0 else abs(home_ml) / (abs(home_ml) + 100.0)
        p_away = 100.0 / (away_ml + 100.0) if away_ml > 0 else abs(away_ml) / (abs(away_ml) + 100.0)
        tot = p_home + p_away
        if tot > 0:
            return float(p_home / tot)
    return float(norm.cdf(spread_line / 13.5))

def calculate_spread_cover_distribution(raw_model_home_prob, market_home_prob, spread_line, total_line=44.0):
    spread_magnitude = abs(spread_line)
    dynamic_market_weight = min(0.75, max(0.48, 0.48 + (spread_magnitude * 0.022)))
    calibrated_home_win_prob = ((1.0 - dynamic_market_weight) * raw_model_home_prob) + (dynamic_market_weight * market_home_prob)
    
    sigma = 13.45 * math.sqrt(max(32.0, total_line) / 44.0)
    z_win = norm.ppf(max(0.01, min(0.99, calibrated_home_win_prob)))
    model_projected_margin = z_win * sigma

    abs_spread = round(abs(spread_line))
    push_rate = NFL_KEY_PUSH_RATES.get(abs_spread, 0.0) if float(spread_line).is_integer() else 0.0

    if spread_line.is_integer():
        z_cover_home = (model_projected_margin - (spread_line + 0.5)) / sigma
        z_cover_away = ((spread_line - 0.5) - model_projected_margin) / sigma
    else:
        z_cover_home = (model_projected_margin - spread_line) / sigma
        z_cover_away = (spread_line - model_projected_margin) / sigma

    home_cover_prob = float(norm.cdf(z_cover_home))
    away_cover_prob = float(norm.cdf(z_cover_away))

    if push_rate > 0:
        tot_mass = home_cover_prob + away_cover_prob
        if tot_mass > 0:
            scale = (1.0 - push_rate) / tot_mass
            home_cover_prob *= scale
            away_cover_prob *= scale

    return float(calibrated_home_win_prob), float(home_cover_prob), float(away_cover_prob), float(push_rate)

def calculate_eighth_kelly_three_outcome(prob_win, push_prob=0.0, decimal_odds=1.9091, max_cap=2.00):
    b = decimal_odds - 1.0
    prob_loss = max(0.0, 1.0 - prob_win - push_prob)
    expected_value = (prob_win * b) - prob_loss
    if expected_value <= 0.0:
        return 0.00

    full_kelly = (b * prob_win - prob_loss) / b
    fractional = full_kelly * 0.125 * 100.0
    return round(float(min(max_cap, max(0.0, fractional))), 2)

# 7. Log-Normal Median Conversion for Skill Baselines
LOG_SIGMA = {"QB_Pass": 0.32, "RB_Rush": 0.48, "Skill_Rec": 0.58}

def convert_mean_to_median(mean_val, category):
    sig = LOG_SIGMA.get(category, 0.50)
    median_val = mean_val * math.exp(-(sig ** 2) / 2.0)
    return round(float(median_val), 1)

def get_full_skill_player_baselines(team_abbr, implied_team_total=22.0):
    scratches = []
    if not injuries.empty and "team" in injuries.columns:
        t_inj = injuries[(injuries["team"] == team_abbr) & (injuries["report_status"].isin(["Out", "Doubtful", "IR"]))]
        inj_name_col = next((c for c in ["full_name", "player_name", "player"] if c in t_inj.columns), None)
        if inj_name_col:
            scratches = t_inj[inj_name_col].dropna().unique().tolist()

    depth_map = {"QB": ["1"], "RB": ["1", "2"], "WR": ["1", "2", "3"], "TE": ["1"]}
    roster_picks = {
        "QB1": "Starting QB", "RB1": "Starting RB1", "RB2": "Starting RB2",
        "WR1": "Starting WR1", "WR2": "Starting WR2", "WR3": "Starting WR3", "TE1": "Starting TE1"
    }

    if not depth_charts.empty:
        team_col = next((c for c in ["club_code", "team"] if c in depth_charts.columns), None)
        pos_col = next((c for c in ["pos_abb", "position", "pos_name", "pos"] if c in depth_charts.columns), None)
        rank_col = next((c for c in ["pos_rank", "depth_team", "rank"] if c in depth_charts.columns), None)
        name_col = next((c for c in ["player_name", "full_name", "player"] if c in depth_charts.columns), None)

        if team_col and pos_col and rank_col and name_col:
            t_dc = depth_charts[depth_charts[team_col] == team_abbr]
            for pos, ranks in depth_map.items():
                for r in ranks:
                    label = f"{pos}{r}"
                    pos_match = t_dc[(t_dc[pos_col] == pos) & (t_dc[rank_col].astype(str).str.strip() == r)]
                    if not pos_match.empty:
                        cand = pos_match.iloc[0][name_col]
                        if cand not in scratches:
                            roster_picks[label] = cand

    name_stat_col = next((c for c in ["player_name", "player", "full_name"] if c in player_stats.columns), None)
    profiles = []
    pace_factor = max(0.75, min(1.25, implied_team_total / 22.0))

    if not player_stats.empty and name_stat_col:
        def get_metrics(player_name, role):
            p_df = player_stats[player_stats[name_stat_col] == player_name]
            p_dict = {"player": player_name, "role": role, "team": team_abbr}
            
            if "QB" in role:
                raw_mean_pass = float(p_df["passing_yards"].mean()) if not p_df.empty and "passing_yards" in p_df else 242.0
                raw_mean_rush = float(p_df["rushing_yards"].mean()) if not p_df.empty and "rushing_yards" in p_df else 16.5
                p_dict["Pass Yards"] = convert_mean_to_median(raw_mean_pass * pace_factor, "QB_Pass")
                p_dict["Rush Yards"] = convert_mean_to_median(raw_mean_rush, "RB_Rush")
                p_dict["prop_type"] = "Pass Yards"
                p_dict["market_line"] = p_dict["Pass Yards"]
            elif "RB" in role:
                raw_mean_rush = float(p_df["rushing_yards"].mean()) if not p_df.empty and "rushing_yards" in p_df else (62.0 if "1" in role else 34.0)
                raw_mean_rec = float(p_df["receiving_yards"].mean()) if not p_df.empty and "receiving_yards" in p_df else (21.0 if "1" in role else 9.5)
                p_dict["Rush Yards"] = convert_mean_to_median(raw_mean_rush * pace_factor, "RB_Rush")
                p_dict["Rec Yards"] = convert_mean_to_median(raw_mean_rec * pace_factor, "Skill_Rec")
                p_dict["prop_type"] = "Rush Yards"
                p_dict["market_line"] = p_dict["Rush Yards"]
            elif "WR" in role or "TE" in role:
                def_rec = 72.0 if "WR1" in role else (48.0 if "WR2" in role else (32.0 if "WR3" in role else 42.0))
                raw_mean_rec = float(p_df["receiving_yards"].mean()) if not p_df.empty and "receiving_yards" in p_df else def_rec
                p_dict["Rec Yards"] = convert_mean_to_median(raw_mean_rec * pace_factor, "Skill_Rec")
                p_dict["prop_type"] = "Rec Yards"
                p_dict["market_line"] = p_dict["Rec Yards"]
            return p_dict

        for role_key, p_name in roster_picks.items():
            profiles.append(get_metrics(p_name, role_key))

    return {"profiles": profiles, "scratches": scratches[:5] if scratches else ["None Reported"]}

# 8. LLM Strategic Scouting Voice (Gemini 3.8 Flash)
async def generate_matchup_analysis(semaphore, payload, recommended_team, recommended_line, chosen_edge, kelly_units):
    system_prompt = """
# ROLE & IDENTITY
You are the "NFL Research Director & Quantitative Architect," operating at the nexus of NFL coaching tape breakdown, spatiotemporal tracking physics (Next Gen Stats), and advanced sabermetric modeling.

# DIRECTIVES
- Anti-Anchoring: Output independent projections derived strictly from scheme volume, not Vegas echoes.
- Median Pricing: Project median yards (50th percentile expectation), not high-variance ceiling means.
- Target Tree Sanity: The sum of team receiving yards across all targets must sit within 80% to 118% of that team's gross passing yards.
- Epistemic Calibration: If exact tracking data is absent, state metrics in directional percentiles or scheme tiers. Never fabricate decimal-precision metrics.
- Output strictly valid JSON matching the exact array schema without markdown formatting.
"""
    prompt = f"""
Evaluate this NFL advance scouting dossier with sportsbook prop benchmarks:
{json.dumps(payload, indent=2)}

Output strictly valid JSON matching this exact array schema:
{{
  "executive_summary": "State whether this game is a BET ({recommended_line} at {chosen_edge:+.1%} edge) or a PASS based on market key numbers and early-down leverage.",
  "schematic_matchup": {{
    "away_offense_vs_home_defense": "Film breakdown: Pass protection win rates, blitz packages, run-blocking scheme (Zone vs Gap), and coverage shells (MOFC Cover 1/3 vs MOFO Quarters/Cover 6).",
    "home_offense_vs_away_defense": "Film breakdown: Pass protection win rates, blitz packages, run-blocking scheme (Zone vs Gap), and coverage shells (MOFC Cover 1/3 vs MOFO Quarters/Cover 6)."
  }},
  "player_projections": [
    {{
      "team": "Team Abbr",
      "role": "QB1 / RB1 / RB2 / WR1 / WR2 / WR3 / TE1",
      "player": "Player Name",
      "prop_category": "Pass Yards / Rush Yards / Rec Yards",
      "tactical_rationale": "Chain-of-thought film/data reasoning evaluating matchup, volume trend, and game script.",
      "projected_value": 0.0,
      "edge": "OVER / UNDER / PASS"
    }}
  ],
  "actionable_verdict": "{'PASS - 0.00u' if recommended_team == 'PASS' else 'Bet ' + recommended_line + ' - ' + str(kelly_units) + 'u'}"
}}
"""
    async with semaphore:
        for attempt in range(3):
            try:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.15,
                            response_mime_type="application/json"
                        )
                    )
                )
                return response.text
            except Exception as e:
                if attempt == 2:
                    return json.dumps({
                        "executive_summary": f"Quant assessment: {recommended_line}",
                        "schematic_matchup": {"away_offense_vs_home_defense": "N/A", "home_offense_vs_away_defense": "N/A"},
                        "player_projections": [],
                        "actionable_verdict": f"{'PASS - 0.00u' if recommended_team == 'PASS' else 'Bet ' + recommended_line + ' - ' + str(kelly_units) + 'u'}"
                    })
                await asyncio.sleep(2 ** attempt)

# 9. Main Pipeline Processing
async def main():
    target_week = 1
    target_season = CURRENT_SEASON
    upcoming = pd.DataFrame()

    if not schedules.empty:
        unplayed = schedules[schedules["result"].isna()]
        if not unplayed.empty:
            target_week = int(unplayed["week"].min())
            target_season = int(unplayed["season"].min()) if "season" in unplayed.columns else CURRENT_SEASON
            upcoming = unplayed[unplayed["week"] == target_week].copy()

    if upcoming.empty:
        print("No active unplayed slate found.")
        sys.exit(0)

    print(f"Executing Season {target_season} Week {target_week} Quant Pipeline ({len(upcoming)} matchups)...")
    pre_processed = []

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

        home_row = get_latest_team_row(home_team, target_season, week_num)
        away_row = get_latest_team_row(away_team, target_season, week_num)

        def get_metric(df, col_name, default=0.0):
            if not df.empty and col_name in df.columns and pd.notna(df[col_name].values[0]):
                return float(df[col_name].values[0])
            return float(default)

        net_pass_edge = (get_metric(home_row, "roll_off_dropback_epa") - get_metric(away_row, "roll_def_dropback_epa")) - \
                        (get_metric(away_row, "roll_off_dropback_epa") - get_metric(home_row, "roll_def_dropback_epa"))
        net_rush_edge = (get_metric(home_row, "roll_off_rush_epa") - get_metric(away_row, "roll_def_rush_epa")) - \
                        (get_metric(away_row, "roll_off_rush_epa") - get_metric(home_row, "roll_def_rush_epa"))
        net_late_down_edge = (get_metric(home_row, "roll_off_late_down_epa") - get_metric(away_row, "roll_def_late_down_epa")) - \
                             (get_metric(away_row, "roll_off_late_down_epa") - get_metric(home_row, "roll_def_late_down_epa"))
        diff_success = get_metric(home_row, "roll_off_early_down_success", 0.44) - get_metric(away_row, "roll_off_early_down_success", 0.44)
        diff_explosive = get_metric(home_row, "roll_off_explosive", 0.12) - get_metric(away_row, "roll_off_explosive", 0.12)

        home_vorp = calculate_roster_vorp_penalty(home_team, injuries, depth_charts)
        away_vorp = calculate_roster_vorp_penalty(away_team, injuries, depth_charts)
        net_pass_edge += (away_vorp - home_vorp)

        home_rest = float(game.get("home_rest", 7.0)) if pd.notna(game.get("home_rest")) else 7.0
        away_rest = float(game.get("away_rest", 7.0)) if pd.notna(game.get("away_rest")) else 7.0
        rest_diff = home_rest - away_rest
        is_divisional = int(game.get("div_game", 0)) if pd.notna(game.get("div_game")) else 0

        feature_row = pd.DataFrame([[
            net_pass_edge, net_rush_edge, net_late_down_edge, diff_success,
            diff_explosive, rest_diff, is_divisional
        ]], columns=FEATURES)

        raw_model_home_prob = float(model.predict_proba(feature_row)[0][1])

        calibrated_home_win_prob, home_cover_prob, away_cover_prob, push_prob = calculate_spread_cover_distribution(
            raw_model_home_prob, market_home_prob, spread_line, total_line
        )

        home_spread_edge = home_cover_prob - 0.5238
        away_spread_edge = away_cover_prob - 0.5238

        vegas_home_line = f"{home_team} {-spread_line:+g}"
        vegas_away_line = f"{away_team} {+spread_line:+g}"

        is_home_dog = (spread_line < 0)
        is_away_dog = (spread_line > 0)
        home_hurdle = 0.025 if is_home_dog else 0.018
        away_hurdle = 0.025 if is_away_dog else 0.018

        if home_spread_edge > home_hurdle and home_spread_edge > away_spread_edge:
            rec_team = home_team
            rec_line = vegas_home_line
            chosen_cover = home_cover_prob
            chosen_edge = min(0.050, home_spread_edge)
            kelly_units = calculate_eighth_kelly_three_outcome(home_cover_prob, push_prob)
        elif away_spread_edge > away_hurdle and away_spread_edge > home_spread_edge:
            rec_team = away_team
            rec_line = vegas_away_line
            chosen_cover = away_cover_prob
            chosen_edge = min(0.050, away_spread_edge)
            kelly_units = calculate_eighth_kelly_three_outcome(away_cover_prob, push_prob)
        else:
            rec_team = "PASS"
            rec_line = "PASS - No Edge"
            chosen_cover = max(home_cover_prob, away_cover_prob)
            chosen_edge = max(home_spread_edge, away_spread_edge)
            kelly_units = 0.00

        implied_home_total = (total_line / 2.0) + (spread_line / 2.0)
        implied_away_total = (total_line / 2.0) - (spread_line / 2.0)

        home_ctx = get_full_skill_player_baselines(home_team, implied_home_total)
        away_ctx = get_full_skill_player_baselines(away_team, implied_away_total)

        home_archetypes = extract_team_tactical_archetypes(pbp, home_team)
        away_archetypes = extract_team_tactical_archetypes(pbp, away_team)

        pre_processed.append({
            "game_id": str(game.get("game_id", f"{target_season}_{week_num}_{away_team}_{home_team}")),
            "week": int(week_num),
            "matchup": str(matchup),
            "home_win_prob": float(calibrated_home_win_prob),
            "market_prob": float(market_home_prob),
            "spread_cover_prob": float(chosen_cover),
            "spread_edge": float(chosen_edge),
            "kelly_units": float(kelly_units),
            "recommended_team": rec_team,
            "recommended_line": rec_line,
            "total_line": total_line,
            "tactical_archetypes": {"home_team": home_archetypes, "away_team": away_archetypes},
            "tape_metrics": {
                "net_pass_epa_diff": f"{net_pass_edge:+.3f}",
                "net_rush_epa_diff": f"{net_rush_edge:+.3f}",
                "explosive_rate_diff": f"{diff_explosive:+.3f}",
                "early_down_success_diff": f"{diff_success:+.3f}"
            },
            "rosters": {
                "home_team": {"team": home_team, "profiles": home_ctx["profiles"], "injuries": home_ctx["scratches"]},
                "away_team": {"team": away_team, "profiles": away_ctx["profiles"], "injuries": away_ctx["scratches"]}
            }
        })

    tasks = []
    semaphore = asyncio.Semaphore(4)

    for item in pre_processed:
        clean_profiles_home = [{"role": p["role"], "player": p["player"], "prop_type": p["prop_type"]} for p in item["rosters"]["home_team"]["profiles"]]
        clean_profiles_away = [{"role": p["role"], "player": p["player"], "prop_type": p["prop_type"]} for p in item["rosters"]["away_team"]["profiles"]]

        payload = {
            "matchup": item["matchup"],
            "market": {
                "consensus_spread": item["recommended_line"],
                "total": item["total_line"],
                "devigged_home_win_prob": f"{item['market_prob']:.1%}"
            },
            "tape_metrics": item["tape_metrics"],
            "tactical_archetypes": item["tactical_archetypes"],
            "model_calculations": {
                "calibrated_home_win_prob": f"{item['home_win_prob']:.1%}",
                "cover_prob": f"{item['spread_cover_prob']:.1%}",
                "edge": f"{item['spread_edge']:+.1%}",
                "recommended_side": item["recommended_team"],
                "recommended_line": item["recommended_line"],
                "suggested_kelly_units": f"{item['kelly_units']:.2f}u"
            },
            "rosters": {
                "home_team": {"team": item["rosters"]["home_team"]["team"], "profiles": clean_profiles_home, "injuries": item["rosters"]["home_team"]["injuries"]},
                "away_team": {"team": item["rosters"]["away_team"]["team"], "profiles": clean_profiles_away, "injuries": item["rosters"]["away_team"]["injuries"]}
            }
        }
        tasks.append(generate_matchup_analysis(
            semaphore, payload, item["recommended_team"], item["recommended_line"], 
            item["spread_edge"], item["kelly_units"]
        ))

    results = await asyncio.gather(*tasks)
    records = []

    for item, text_response in zip(pre_processed, results):
        try:
            parsed_analysis = json.loads(text_response)
        except Exception:
            parsed_analysis = {"player_projections": []}

        # Multi-key post-hoc market line merging
        if "player_projections" in parsed_analysis and isinstance(parsed_analysis["player_projections"], list):
            for p in parsed_analysis["player_projections"]:
                p_team = p.get("team", "").strip().upper()
                team_key = "home" if p_team == item["rosters"]["home_team"]["team"].upper() else "away"
                roster_profiles = item["rosters"][f"{team_key}_team"]["profiles"]
                prof = next((x for x in roster_profiles if x["player"] == p.get("player")), None)
                
                cat = "Pass Yards" if "Pass" in p.get("prop_category", "") else ("Rush Yards" if "Rush" in p.get("prop_category", "") else "Rec Yards")
                if prof and cat in prof:
                    p["market_line"] = prof[cat]
                else:
                    p["market_line"] = 218.5 if cat == "Pass Yards" else (44.5 if cat == "Rush Yards" else 32.5)

            text_response = json.dumps(parsed_analysis)

        # Stop-block verification gate
        verification = NFLDataVerifier.audit_slate_payload(item, parsed_analysis)
        if not verification.is_valid:
            print(f"CRITICAL STOP-BLOCK: {item['matchup']} failed data integrity.")
            item["kelly_units"] = 0.00
            item["recommended_team"] = "PASS"
            item["recommended_line"] = "PASS - INTEGRITY BREACH"
            parsed_analysis["actionable_verdict"] = "PASS - DATA INTEGRITY BREACH"
            parsed_analysis["integrity_violations"] = verification.audit_log
            text_response = json.dumps(parsed_analysis)

        records.append({
            "game_id": item["game_id"],
            "week": item["week"],
            "matchup": item["matchup"],
            "home_win_prob": item["home_win_prob"],
            "market_prob": item["market_prob"],
            "spread_cover_prob": item["spread_cover_prob"],
            "spread_edge": item["spread_edge"],
            "kelly_units": item["kelly_units"],
            "analysis": text_response
        })
        print(f"Processed: {item['matchup']} | Line: {item['recommended_line']} | Edge: {item['spread_edge']:+.1%} | Stake: {item['kelly_units']}u")

    # Neon PostgreSQL Commit
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
            conn.execute(
                text("DELETE FROM nfl_weekly_analysis WHERE week = :target_week"),
                {"target_week": target_week}
            )
        df_results.to_sql("nfl_weekly_analysis", engine, if_exists="append", index=False)
        print(f"Database sync successful: {len(df_results)} matchups committed for Week {target_week}.")

if __name__ == "__main__":
    asyncio.run(main())
if __name__ == "__main__":
    asyncio.run(main())
