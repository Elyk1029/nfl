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

# 1. Environment Verification & Client Initialization
db_url = os.environ.get("DATABASE_URL")
gemini_key = os.environ.get("GEMINI_API_KEY")

if not db_url or not gemini_key:
    raise ValueError("FATAL: DATABASE_URL and GEMINI_API_KEY must be configured in environment.")

engine = create_engine(db_url, pool_size=5, pool_pre_ping=True)
client = genai.Client(api_key=gemini_key)

MODEL_FILE = "nfl_model.json"
model = xgb.XGBClassifier()
if os.path.exists(MODEL_FILE):
    model.load_model(MODEL_FILE)
    print("XGBoost classifier loaded successfully.")
else:
    raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found in root directory.")

FEATURES = [
    "net_pass_edge", "net_rush_edge", "net_late_down_edge", "diff_success",
    "diff_explosive", "rest_diff", "is_divisional", "market_home_prob",
]

# Discrete empirical NFL key-number push frequencies
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

# 2. Dynamic Ingestion via nflreadpy
CURRENT_SEASON = 2026
DATA_SEASON = 2025

print("Ingesting schedules, rosters, and live depth charts...")
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

# 3. Clean EPA & Early-Down Success Rate (EDSR)
if not pbp.empty:
    pbp_clean = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    if "home_wp" in pbp_clean.columns and "qtr" in pbp_clean.columns:
        leverage_mask = (pbp_clean["qtr"] <= 2) | (pbp_clean["home_wp"].between(0.15, 0.85))
        pbp_clean = pbp_clean[leverage_mask]

    pbp_clean["is_early_down"] = pbp_clean["down"].isin([1, 2]).astype(int) if "down" in pbp_clean.columns else 1
    pbp_clean["is_late_down"] = pbp_clean["down"].isin([3, 4]).astype(int) if "down" in pbp_clean.columns else 0
    pbp_clean["is_explosive"] = (
        ((pbp_clean["play_type"] == "pass") & (pbp_clean["yards_gained"] >= 15)) |
        ((pbp_clean["play_type"] == "run") & (pbp_clean["yards_gained"] >= 10))
    ).astype(int)

    off_stats = pbp_clean.groupby(["week", "posteam"]).agg(
        off_dropback_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "play_type"] == "pass"].mean()),
        off_rush_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "play_type"] == "run"].mean()),
        off_early_down_success=("success", lambda x: x[pbp_clean.loc[x.index, "is_early_down"] == 1].mean()),
        off_late_down_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "is_late_down"] == 1].mean()),
        off_explosive=("is_explosive", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})

    def_stats = pbp_clean.groupby(["week", "defteam"]).agg(
        def_dropback_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "play_type"] == "pass"].mean()),
        def_rush_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "play_type"] == "run"].mean()),
        def_early_down_success=("success", lambda x: x[pbp_clean.loc[x.index, "is_early_down"] == 1].mean()),
        def_late_down_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "is_late_down"] == 1].mean()),
    ).reset_index().rename(columns={"defteam": "team"})

    team_perf = pd.merge(off_stats, def_stats, on=["week", "team"], how="outer").fillna(0)
    team_perf.sort_values(["team", "week"], inplace=True)

    metric_cols = [
        "off_dropback_epa", "off_rush_epa", "off_early_down_success", "off_late_down_epa", "off_explosive",
        "def_dropback_epa", "def_rush_epa", "def_early_down_success", "def_late_down_epa",
    ]
    for col in metric_cols:
        team_perf[f"roll_{col}"] = team_perf.groupby("team")[col].transform(
            lambda x: x.shift(1).ewm(span=6, min_periods=1).mean()
        )
else:
    team_perf = pd.DataFrame()

# 4. Point Spread Engine & Eighth-Kelly Risk Sizing
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
    dynamic_market_weight = min(0.78, max(0.52, 0.52 + (spread_magnitude * 0.020)))
    calibrated_home_win_prob = ((1.0 - dynamic_market_weight) * raw_model_home_prob) + (dynamic_market_weight * market_home_prob)
    
    sigma = 13.5 * math.sqrt(max(30.0, total_line) / 44.0)
    z_win = norm.ppf(max(0.01, min(0.99, calibrated_home_win_prob)))
    model_projected_margin = z_win * sigma

    z_home_cover = (model_projected_margin - spread_line) / sigma
    continuous_home_cover = float(norm.cdf(z_home_cover))
    
    abs_spread = round(abs(spread_line))
    push_rate = NFL_KEY_PUSH_RATES.get(abs_spread, 0.015) if float(spread_line).is_integer() else 0.0
    
    home_cover_prob = continuous_home_cover * (1.0 - (push_rate * 0.5))
    away_cover_prob = (1.0 - continuous_home_cover) * (1.0 - (push_rate * 0.5))
    
    home_cover_prob = max(0.30, min(0.70, home_cover_prob))
    away_cover_prob = max(0.30, min(0.70, away_cover_prob))
    
    return float(calibrated_home_win_prob), float(home_cover_prob), float(away_cover_prob), float(push_rate)

def calculate_eighth_kelly(prob_win, decimal_odds=1.9091, max_cap=2.00):
    if prob_win <= 0.5238:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - prob_win
    raw_kelly = (b * prob_win - q) / b
    fractional = raw_kelly * 0.125 * 100.0
    return round(float(min(max_cap, max(0.0, fractional))), 2)

# 5. Full Skill-Group Target Tree & Benchmark Aggregation
def get_full_skill_player_baselines(team_abbr, implied_team_total=22.0):
    scratches = []
    if not injuries.empty and "team" in injuries.columns:
        t_inj = injuries[(injuries["team"] == team_abbr) & (injuries["report_status"].isin(["Out", "Doubtful", "IR"]))]
        inj_name_col = next((c for c in ["full_name", "player_name", "player"] if c in t_inj.columns), None)
        if inj_name_col:
            scratches = t_inj[inj_name_col].dropna().unique().tolist()

    depth_map = {
        "QB": ["1"], "RB": ["1", "2"], "WR": ["1", "2", "3"], "TE": ["1"]
    }
    
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
                pass_yds = round(float(p_df["passing_yards"].mean()), 1) if not p_df.empty and "passing_yards" in p_df else 235.5
                p_dict.update({
                    "prop_type": "Pass Yards",
                    "market_line": round(pass_yds * pace_factor, 1)
                })
            elif "RB" in role:
                rush_yds = round(float(p_df["rushing_yards"].mean()), 1) if not p_df.empty and "rushing_yards" in p_df else (58.5 if "1" in role else 32.5)
                p_dict.update({
                    "prop_type": "Rush Yards",
                    "market_line": round(rush_yds * pace_factor, 1)
                })
            elif "WR" in role or "TE" in role:
                yds_def = 68.5 if "WR1" in role else (44.5 if "WR2" in role else (28.5 if "WR3" in role else 38.5))
                rec_yds = round(float(p_df["receiving_yards"].mean()), 1) if not p_df.empty and "receiving_yards" in p_df else yds_def
                p_dict.update({
                    "prop_type": "Rec Yards",
                    "market_line": round(rec_yds * pace_factor, 1)
                })
            return p_dict

        for role_key, p_name in roster_picks.items():
            profiles.append(get_metrics(p_name, role_key))

    return {
        "profiles": profiles,
        "scratches": scratches[:5] if scratches else ["None Reported"]
    }

# 6. Strategic Scouting Voice Evaluator (Gemini 3.8 Flash)
async def generate_matchup_analysis(semaphore, payload, recommended_team, recommended_line, chosen_edge, kelly_units):
    system_prompt = """
# ROLE & PERSONA
You are the "NFL Analytics Coordinator & Strategic Guru," operating at the intersection of advanced football sabermetrics and high-level coaching tape analysis. You possess elite-level fluency in both traditional football film study (schemes, coverages, run fits, route concepts) and modern predictive analytics (EPA/play, CPOE, Success Rate, DVOA, pressure rate vs. quick game, NGS tracking data).

Your dual mandate:
1. Deliver razor-sharp, objective, and analytically grounded NFL football analysis.
2. Act as an expert AI evaluator: Continuously review user-submitted AI prompts, analytical frameworks, model outputs, or predictive systems to pinpoint blind spots, eliminate statistical noise, and recommend improvements.

# CORE COMPETENCIES & KNOWLEDGE BASE
- Scheme & Tactical Fluency: Personnel groupings (11, 12, 21 personnel), pass-pro schemes, run-blocking schemes (Inside/Outside Zone, Duo, Power/Counter), route distribution vs. MOFO/MOFC (Middle of Field Open/Closed), coverage shells (Cover 1, 2-Man, Quarters, Palms, Cover 3 Match).
- Advanced Metrics & Modeling: EPA per play, Success Rate, CPOE, DVOA, Adjusted Net Yards Per Attempt (ANY/A), explosive play rate, win probability models, high-leverage 4th-down decision curves.
- Data Hygiene: Sample size discipline, regressing unstable metrics (turnover luck, fumble recovery rates, red zone TD% variance) toward the mean, distinguishing process from outcome.

# OPERATIONAL DIRECTIVES
- Anti-Anchoring Mandate: Do NOT default to market benchmarks or copy numbers. Your projection must reflect your own independent read of usage share, game script, and matchup.
- Chain-of-Thought First: You must formulate the `tactical_rationale` explaining your film/data read BEFORE determining the final `projected_value`.
- Output strictly valid JSON matching the exact array schema without markdown formatting.
"""
    prompt = f"""
Evaluate this NFL advance scouting dossier:
{json.dumps(payload, indent=2)}

Output strictly valid JSON matching this exact array schema (note that tactical_rationale must come BEFORE projected_value):
{{
  "executive_summary": "State whether this game is a BET ({recommended_line} at {chosen_edge:+.1%} edge) or a PASS based on market key numbers and early-down leverage.",
  "schematic_matchup": {{
    "away_offense_vs_home_defense": "Film analysis of protection rates, blitz packages, and coverage families (MOFC vs MOFO).",
    "home_offense_vs_away_defense": "Film analysis of protection rates, blitz packages, and coverage families (MOFC vs MOFO)."
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

# 7. Portfolio Governance Engine
def apply_portfolio_risk_governance(slate_data, max_slate_bets=4, max_dog_units=3.5):
    slate_data.sort(key=lambda x: x["spread_edge"], reverse=True)
    governed = []
    dog_units_staked = 0.0
    action_count = 0

    for item in slate_data:
        is_dog = (item["recommended_line"].find("+") != -1)
        stake = item["kelly_units"]
        edge = item["spread_edge"]

        if edge >= 0.018 and action_count < max_slate_bets and stake > 0:
            if is_dog:
                if dog_units_staked >= max_dog_units:
                    item["kelly_units"] = 0.00
                    item["recommended_team"] = "PASS"
                    item["recommended_line"] = "PASS - Portfolio Cap Exceeded"
                elif (dog_units_staked + stake) > max_dog_units:
                    item["kelly_units"] = round(max_dog_units - dog_units_staked, 2)
                    dog_units_staked += item["kelly_units"]
                    action_count += 1
                else:
                    dog_units_staked += stake
                    action_count += 1
            else:
                action_count += 1
        else:
            item["kelly_units"] = 0.00
            item["recommended_team"] = "PASS"
            item["recommended_line"] = "PASS - No Edge"

        governed.append(item)

    return governed

# 8. Main Pipeline Processing
async def main():
    target_week = 1
    upcoming = pd.DataFrame()

    if not schedules.empty:
        unplayed = schedules[schedules["result"].isna()]
        if not unplayed.empty:
            target_week = int(unplayed["week"].min())
            upcoming = unplayed[unplayed["week"] == target_week].copy()

    if upcoming.empty:
        print("No active unplayed slate found.")
        sys.exit(0)

    print(f"Executing Week {target_week} Quant Pipeline ({len(upcoming)} matchups)...")

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
        diff_success = get_metric(home_row, "roll_off_early_down_success", 0.44) - get_metric(away_row, "roll_off_early_down_success", 0.44)
        diff_explosive = get_metric(home_row, "roll_off_explosive", 0.12) - get_metric(away_row, "roll_off_explosive", 0.12)

        feature_row = pd.DataFrame([[
            net_pass_edge, net_rush_edge, net_late_down_edge, diff_success,
            diff_explosive, rest_diff, is_divisional, market_home_prob
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
            kelly_units = calculate_eighth_kelly(home_cover_prob)
        elif away_spread_edge > away_hurdle and away_spread_edge > home_spread_edge:
            rec_team = away_team
            rec_line = vegas_away_line
            chosen_cover = away_cover_prob
            chosen_edge = min(0.050, away_spread_edge)
            kelly_units = calculate_eighth_kelly(away_cover_prob)
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

        # Baseline lookup maps for post-hoc calculation
        home_market_map = {f"{p['role']}_{p['prop_type']}": p['market_line'] for p in home_ctx["profiles"]}
        away_market_map = {f"{p['role']}_{p['prop_type']}": p['market_line'] for p in away_ctx["profiles"]}

        pre_processed.append({
            "game_id": str(game.get("game_id", f"2026_{week_num}_{away_team}_{home_team}")),
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
            "market_maps": {"home": home_market_map, "away": away_market_map},
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

    governed_slate = apply_portfolio_risk_governance(pre_processed, max_slate_bets=4, max_dog_units=3.5)

    tasks = []
    semaphore = asyncio.Semaphore(4)

    for item in governed_slate:
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
    for item, text_response in zip(governed_slate, results):
        try:
            parsed_analysis = json.loads(text_response)
            if "player_projections" in parsed_analysis and isinstance(parsed_analysis["player_projections"], list):
                for p in parsed_analysis["player_projections"]:
                    team_key = "home" if p.get("team","").upper() == item["rosters"]["home_team"]["team"].upper() else "away"
                    m_map = item["market_maps"][team_key]
                    cat = p.get('prop_category', 'Yards')
                    if "Pass" in cat:
                        map_key = f"{p.get('role')}_Pass Yards"
                    elif "Rush" in cat:
                        map_key = f"{p.get('role')}_Rush Yards"
                    else:
                        map_key = f"{p.get('role')}_Rec Yards"
                    
                    p["market_line"] = m_map.get(map_key, 45.0)
                text_response = json.dumps(parsed_analysis)
        except Exception as e:
            print(f"Error merging post-hoc market lines for {item['matchup']}: {e}")

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
        print(f"Executed: {item['matchup']} | Play: {item['recommended_line']} | Edge: {item['spread_edge']:+.1%} | Stake: {item['kelly_units']}u")

    # 9. Database Synchronization
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
        print(f"Database synchronized: {len(df_results)} records committed for Week {target_week}.")

if __name__ == "__main__":
    asyncio.run(main())
