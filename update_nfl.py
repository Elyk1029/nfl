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

# 1. Environment & Database Verification
db_url = os.environ.get("DATABASE_URL")
gemini_key = os.environ.get("GEMINI_API_KEY")

if not db_url or not gemini_key:
    raise ValueError("FATAL: DATABASE_URL and GEMINI_API_KEY must be exported in environment.")

engine = create_engine(db_url, pool_size=5, pool_pre_ping=True)
client = genai.Client(api_key=gemini_key)

MODEL_FILE = "nfl_model.json"
model = xgb.XGBClassifier()
if os.path.exists(MODEL_FILE):
    model.load_model(MODEL_FILE)
    print("XGBoost model loaded.")
else:
    raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found.")

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

print("Pulling live NFL schedules and rosters...")
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

# 3. EPA Filtering & Explosive Play Feature Pipeline
if not pbp.empty:
    pbp_clean = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    
    # Filter Garbage Time (Win Prob between 5% and 95% unless 1st Half)
    if "home_wp" in pbp_clean.columns and "qtr" in pbp_clean.columns:
        leverage_mask = (pbp_clean["qtr"] <= 2) | (pbp_clean["home_wp"].between(0.05, 0.95))
        pbp_clean = pbp_clean[leverage_mask]

    pbp_clean["is_late_down"] = pbp_clean["down"].isin([3, 4]).astype(int) if "down" in pbp_clean.columns else 0
    
    # Isolate true explosive plays: Pass 15+ yards, Rush 10+ yards
    pbp_clean["is_explosive"] = (
        ((pbp_clean["play_type"] == "pass") & (pbp_clean["yards_gained"] >= 15)) |
        ((pbp_clean["play_type"] == "run") & (pbp_clean["yards_gained"] >= 10))
    ).astype(int)

    off_stats = pbp_clean.groupby(["week", "posteam"]).agg(
        off_dropback_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "play_type"] == "pass"].mean()),
        off_rush_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "play_type"] == "run"].mean()),
        off_success=("success", "mean"),
        off_late_down_epa=("epa", lambda x: x[pbp_clean.loc[x.index, "is_late_down"] == 1].mean()),
        off_explosive=("is_explosive", "mean"),
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

# 4. Mathematical Modeling & Spread Cover Distribution
def get_devigged_market_home_prob(spread_line, home_ml=None, away_ml=None):
    if home_ml is not None and away_ml is not None and not math.isnan(home_ml) and not math.isnan(away_ml):
        p_home = 100.0 / (home_ml + 100.0) if home_ml > 0 else abs(home_ml) / (abs(home_ml) + 100.0)
        p_away = 100.0 / (away_ml + 100.0) if away_ml > 0 else abs(away_ml) / (abs(away_ml) + 100.0)
        tot = p_home + p_away
        if tot > 0:
            return float(p_home / tot)
    return float(norm.cdf(-spread_line / 13.5))

def calculate_spread_cover_distribution(raw_model_home_prob, market_home_prob, spread_line, total_line=44.0):
    """
    Point spread margin distribution model with discrete key-number push calibration.
    Converts model vs market probabilities into an implied projected point spread differential.
    """
    # Shrink raw model edge by 65% toward market consensus to prevent early-season overreaction
    calibrated_home_win_prob = 0.35 * raw_model_home_prob + 0.65 * market_home_prob
    
    # Margin volatility scales with game total (sigma ~ 13.5 adjusted by total line)
    sigma = 13.5 * math.sqrt(total_line / 44.0)
    
    # Invert normal win prob into implied margin: mu = -sigma * norm.ppf(1 - win_prob)
    z_win = norm.ppf(max(0.01, min(0.99, calibrated_home_win_prob)))
    model_projected_margin = z_win * sigma  # Positive indicates home favorite
    
    # Cover probability calculation: Home covers if (Actual Margin + spread_line) > 0
    # Vegas notation: home_team -3.5 means spread_line = -3.5 (or +3.5 depending on provider).
    # Here spread_line is the point adjustment added to the home score.
    z_cover = (model_projected_margin + spread_line) / sigma
    continuous_home_cover = float(norm.cdf(z_cover))
    
    # Key-number push rate allocation
    abs_spread = round(abs(spread_line))
    push_rate = NFL_KEY_PUSH_RATES.get(abs_spread, 0.015) if float(spread_line).is_integer() else 0.0
    
    home_cover_prob = continuous_home_cover * (1.0 - (push_rate * 0.5))
    away_cover_prob = (1.0 - continuous_home_cover) * (1.0 - (push_rate * 0.5))
    
    # Clamp bounds to institutional sanity thresholds
    home_cover_prob = max(0.30, min(0.70, home_cover_prob))
    away_cover_prob = max(0.30, min(0.70, away_cover_prob))
    
    return float(calibrated_home_win_prob), float(home_cover_prob), float(away_cover_prob), float(push_rate)

def calculate_quarter_kelly(prob_win, decimal_odds=1.9091, max_cap=2.00):
    if prob_win <= 0.5238:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - prob_win
    raw_kelly = (b * prob_win - q) / b
    fractional = raw_kelly * 0.25 * 100.0
    return round(float(min(max_cap, max(0.0, fractional))), 2)

# 5. Dynamic Personnel & Starter Identification
def get_active_starters(team_abbr):
    scratches = []
    if not injuries.empty and "team" in injuries.columns:
        t_inj = injuries[(injuries["team"] == team_abbr) & (injuries["report_status"].isin(["Out", "Doubtful", "IR"]))]
        if "full_name" in t_inj.columns:
            scratches = t_inj["full_name"].dropna().unique().tolist()

    starters = {"QB": "Starting QB", "RB": "Starting RB", "WR": "Starting WR"}
    
    # Query depth chart if available
    team_col = "club_code" if "club_code" in depth_charts.columns else "team"
    if not depth_charts.empty and team_col in depth_charts.columns:
        t_dc = depth_charts[depth_charts[team_col] == team_abbr]
        for pos in ["QB", "RB", "WR"]:
            pos_match = t_dc[(t_dc["position"] == pos) & (t_dc["depth_team"] == "1")]
            if not pos_match.empty and "full_name" in pos_match.columns:
                cand = pos_match.iloc[0]["full_name"]
                if cand not in scratches:
                    starters[pos] = cand

    # Enrich with player stat averages
    team_stat_col = "recent_team" if "recent_team" in player_stats.columns else "team"
    if not player_stats.empty and team_stat_col in player_stats.columns:
        t_stats = player_stats[player_stats[team_stat_col] == team_abbr]
        for pos, metric, unit in [("QB", "passing_yards", "pass yds"), ("RB", "rushing_yards", "rush yds"), ("WR", "receiving_yards", "rec yds")]:
            current_name = starters[pos]
            sub = t_stats[t_stats["player_name"] == current_name]
            if not sub.empty and metric in sub.columns:
                avg_stat = sub[metric].mean()
                starters[pos] = f"{current_name} (~{avg_stat:.1f} {unit}/gm)"
            elif current_name != f"Starting {pos}":
                starters[pos] = f"{current_name} (Active)"

    return {
        "starters": starters,
        "scratches": scratches[:5] if scratches else ["None Reported"]
    }

# 6. Asynchronous LLM Execution Engine
async def generate_matchup_analysis(semaphore, payload, recommended_team, recommended_line, chosen_edge, kelly_units):
    system_prompt = """
You are an NFL Strategic Research Director and quantitative betting syndicate analyst.
Synthesize the provided Expected Points Added (EPA), Success Rates, explosive play ratios, personnel scratches, and key-number spread edges.

Evaluation Protocol:
1. Pinpoint the primary tactical matchup: Evaluate passing offense dropback EPA vs. defense pass rush win rate and secondary coverage shell.
2. In the player projections, reference strictly the confirmed active personnel provided in the JSON input.
3. State whether the model warrants a BET or PASS based on key number pricing.
4. Output strictly valid JSON matching the specified schema.
"""
    prompt = f"""
Analyze this NFL game payload:
{json.dumps(payload, indent=2)}

Output strictly valid JSON with this exact schema:
{{
  "executive_summary": "State whether this game is a BET ({recommended_line} at {chosen_edge:+.1%} edge) or a PASS based on market key numbers.",
  "schematic_matchup": {{
    "away_offense_vs_home_defense": "Film-grounded analysis of away passing/rushing concepts vs. home front and coverage shell.",
    "home_offense_vs_away_defense": "Film-grounded analysis of home passing/rushing concepts vs. away front and coverage shell."
  }},
  "player_projections": {{
    "away_team": {{
      "QB": "Projection statement",
      "RB": "Projection statement",
      "WR": "Projection statement"
    }},
    "home_team": {{
      "QB": "Projection statement",
      "RB": "Projection statement",
      "WR": "Projection statement"
    }}
  }},
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
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.1,
                            response_mime_type="application/json"
                        )
                    )
                )
                return response.text
            except Exception as e:
                if attempt == 2:
                    print(f"Failed analysis for {payload['matchup']}: {e}")
                    return json.dumps({
                        "executive_summary": f"Quant assessment generated: {recommended_line}",
                        "schematic_matchup": {"away_offense_vs_home_defense": "Data unparsed", "home_offense_vs_away_defense": "Data unparsed"},
                        "player_projections": {"away_team": {}, "home_team": {}},
                        "actionable_verdict": f"{'PASS - 0.00u' if recommended_team == 'PASS' else 'Bet ' + recommended_line + ' - ' + str(kelly_units) + 'u'}"
                    })
                await asyncio.sleep(2 ** attempt)

# 7. Main Pipeline Processing
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

    tasks = []
    metadata = []
    semaphore = asyncio.Semaphore(4)  # Concurrency cap to respect API limits

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

        home_line_formatted = f"{home_team} {spread_line:+g}"
        away_line_formatted = f"{away_team} {-spread_line:+g}"

        # 2.0% clear threshold
        if home_spread_edge > 0.02 and home_spread_edge > away_spread_edge:
            recommended_team = home_team
            recommended_line = home_line_formatted
            chosen_cover_prob = home_cover_prob
            chosen_edge = min(0.060, home_spread_edge)
            kelly_units = calculate_quarter_kelly(home_cover_prob)
        elif away_spread_edge > 0.02 and away_spread_edge > home_spread_edge:
            recommended_team = away_team
            recommended_line = away_line_formatted
            chosen_cover_prob = away_cover_prob
            chosen_edge = min(0.060, away_spread_edge)
            kelly_units = calculate_quarter_kelly(away_cover_prob)
        else:
            recommended_team = "PASS"
            recommended_line = "No Value"
            chosen_cover_prob = max(home_cover_prob, away_cover_prob)
            chosen_edge = max(home_spread_edge, away_spread_edge)
            kelly_units = 0.00

        home_ctx = get_active_starters(home_team)
        away_ctx = get_active_starters(away_team)

        payload = {
            "matchup": matchup,
            "market": {
                "vegas_spread": home_line_formatted,
                "total": total_line,
                "devigged_home_ml_prob": f"{market_home_prob:.1%}"
            },
            "metrics": {
                "net_pass_epa_differential": f"{net_pass_edge:+.3f}",
                "net_rush_epa_differential": f"{net_rush_edge:+.3f}",
                "explosive_play_differential": f"{diff_explosive:+.3f}",
            },
            "model_calculations": {
                "calibrated_home_win_prob": f"{calibrated_home_win_prob:.1%}",
                "home_cover_prob": f"{home_cover_prob:.1%}",
                "away_cover_prob": f"{away_cover_prob:.1%}",
                "recommended_side": recommended_team,
                "recommended_line": recommended_line,
                "suggested_kelly_units": f"{kelly_units:.2f}u"
            },
            "rosters": {
                "home_team": {"team": home_team, "starters": home_ctx["starters"], "injuries": home_ctx["scratches"]},
                "away_team": {"team": away_team, "starters": away_ctx["starters"], "injuries": away_ctx["scratches"]}
            }
        }

        tasks.append(generate_matchup_analysis(semaphore, payload, recommended_team, recommended_line, chosen_edge, kelly_units))
        metadata.append({
            "game_id": str(game.get("game_id", f"2026_{week_num}_{away_team}_{home_team}")),
            "week": int(week_num),
            "matchup": str(matchup),
            "home_win_prob": float(calibrated_home_win_prob),
            "market_prob": float(market_home_prob),
            "spread_cover_prob": float(chosen_cover_prob),
            "spread_edge": float(chosen_edge),
            "kelly_units": float(kelly_units),
            "recommended_line": recommended_line
        })

    # Await concurrent LLM analysis
    results = await asyncio.gather(*tasks)

    records = []
    for meta, text_response in zip(metadata, results):
        rec = dict(meta)
        rec["analysis"] = text_response
        del rec["recommended_line"]
        records.append(rec)
        print(f"Processed: {meta['matchup']} | Line: {meta['recommended_line']} | Kelly: {meta['kelly_units']}u")

    # 8. Database Upsert & Migration
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
        print(f"Neon database synchronized for Week {target_week} with {len(df_results)} records.")

if __name__ == "__main__":
    asyncio.run(main())
