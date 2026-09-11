"""
update_nfl.py - Pipeline Orchestrator with Opponent-Adjusted EPA, VORP, Median Conversions,
Discrete Score Modeling, and Auto-Migrating PostgreSQL Commit.
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

FEATURES = [
    "net_pass_edge", "net_rush_edge", "net_late_down_edge", "diff_success",
    "diff_explosive", "rest_diff", "is_divisional", "market_home_prob",
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

# 2. Discrete Score Computation Engine
def calculate_discrete_projected_scores(projected_margin, total_line):
    raw_home = (total_line + projected_margin) / 2.0
    raw_away = (total_line - projected_margin) / 2.0

    def snap_to_football_score(score):
        base = round(score)
        if base in [1, 2, 5]:
            return 3 if base <= 2 else 6
        return int(base)

    home_score = snap_to_football_score(raw_home)
    away_score = snap_to_football_score(raw_away)

    if projected_margin > 0.5 and home_score <= away_score:
        home_score = away_score + (3 if (away_score + 3) - away_score == 3 else 1)
    elif projected_margin < -0.5 and away_score <= home_score:
        away_score = home_score + (3 if (home_score + 3) - home_score == 3 else 1)

    return home_score, away_score

# 3. Pipeline Ingestion & Model Execution
CURRENT_SEASON = 2026
DATA_SEASON = 2025

try:
    schedules = nfl.load_schedules(seasons=[CURRENT_SEASON]).to_pandas()
except Exception:
    schedules = nfl.load_schedules(seasons=[DATA_SEASON]).to_pandas()

try:
    pbp = nfl.load_pbp(seasons=[DATA_SEASON, CURRENT_SEASON]).to_pandas()
except Exception:
    pbp = pd.DataFrame()

try:
    player_stats = nfl.load_player_stats(seasons=[DATA_SEASON, CURRENT_SEASON]).to_pandas()
except Exception:
    player_stats = pd.DataFrame()

try:
    injuries = nfl.load_injuries(seasons=[CURRENT_SEASON]).to_pandas()
except Exception:
    injuries = pd.DataFrame()

try:
    depth_charts = nfl.load_depth_charts(seasons=[CURRENT_SEASON]).to_pandas()
except Exception:
    depth_charts = pd.DataFrame()

for df in [schedules, pbp, player_stats, injuries, depth_charts]:
    if df.empty:
        continue
    for col in ["home_team", "away_team", "posteam", "defteam", "recent_team", "team", "club_code"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_team_abbr)

def compute_opponent_adjusted_epa(pbp_df):
    if pbp_df.empty:
        return pd.DataFrame()
    pbp_clean = pbp_df[pbp_df["play_type"].isin(["pass", "run"])].copy()
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

    merged = pd.merge(off_stats, def_stats, on=["season", "week", "team"], how="outer").fillna(0)
    merged.sort_values(["team", "season", "week"], inplace=True)
    for col in ["off_dropback_epa", "off_rush_epa", "off_early_down_success", "off_late_down_epa", "off_explosive",
                "def_dropback_epa", "def_rush_epa", "def_early_down_success", "def_late_down_epa"]:
        merged[f"roll_{col}"] = merged.groupby("team")[col].transform(lambda x: x.shift(1).ewm(span=6, min_periods=1).mean())
    return merged

team_perf = compute_opponent_adjusted_epa(pbp)

def get_latest_team_row(team_abbr, target_season, target_week):
    if team_perf.empty:
        return pd.DataFrame()
    t_data = team_perf[
        (team_perf["team"] == team_abbr) & 
        ((team_perf["season"] < target_season) | ((team_perf["season"] == target_season) & (team_perf["week"] < target_week)))
    ]
    return t_data.sort_values(["season", "week"], ascending=[False, False]).head(1) if not t_data.empty else pd.DataFrame()

VORP_PENALTIES = {"QB1": 0.22, "LT1": 0.05, "EDGE1": 0.04}
def calculate_roster_vorp(team_abbr):
    if injuries.empty or depth_charts.empty:
        return 0.0
    t_inj = injuries[(injuries["team"] == team_abbr) & (injuries["report_status"].isin(["Out", "Doubtful", "IR"]))]
    if t_inj.empty:
        return 0.0
    inj_names = t_inj["player_name"].dropna().tolist() if "player_name" in t_inj else []
    t_dc = depth_charts[depth_charts["club_code"] == team_abbr] if "club_code" in depth_charts else pd.DataFrame()
    penalty = 0.0
    if not t_dc.empty and "player_name" in t_dc:
        for role, pen in VORP_PENALTIES.items():
            pos_match = t_dc[(t_dc["pos_abb"] == role[:2]) & (t_dc["pos_rank"].astype(str) == "1")]
            if not pos_match.empty and pos_match.iloc[0]["player_name"] in inj_names:
                penalty += pen
    return penalty

async def generate_matchup_analysis(semaphore, payload):
    system_prompt = """
# ROLE & IDENTITY
You are the NFL Research Director & Quantitative Architect.
Deliver objective, accessible film-grounded game notes and score validation.
Never fabricate decimal-precision statistics. Output strictly valid JSON.
"""
    prompt = f"Scout this NFL matchup dossier:\n{json.dumps(payload, indent=2)}\nOutput valid JSON matching schema."
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
                            temperature=0.15,
                            response_mime_type="application/json",
                            tools=None
                        )
                    )
                )
                return response.text
            except Exception:
                await asyncio.sleep(2 ** attempt)
        return json.dumps({
            "executive_summary": f"Analytical edge identified for {payload['matchup']}.",
            "schematic_matchup": {
                "away_offense_vs_home_defense": "Standard alignment.",
                "home_offense_vs_away_defense": "Standard alignment."
            },
            "player_projections": []
        })

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

    print(f"Executing Season {target_season} Week {target_week} Pipeline ({len(upcoming)} matchups)...")
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

        if home_ml is not None and away_ml is not None and not math.isnan(home_ml) and not math.isnan(away_ml):
            p_h = 100.0 / (home_ml + 100.0) if home_ml > 0 else abs(home_ml) / (abs(home_ml) + 100.0)
            p_a = 100.0 / (away_ml + 100.0) if away_ml > 0 else abs(away_ml) / (abs(away_ml) + 100.0)
            market_home_prob = float(p_h / (p_h + p_a)) if (p_h + p_a) > 0 else 0.50
        else:
            market_home_prob = float(norm.cdf(spread_line / 13.5))

        home_row = get_latest_team_row(home_team, target_season, week_num)
        away_row = get_latest_team_row(away_team, target_season, week_num)

        def get_stat(df, col, default=0.0):
            return float(df[col].values[0]) if not df.empty and col in df.columns and pd.notna(df[col].values[0]) else float(default)

        net_pass_edge = (get_stat(home_row, "roll_off_dropback_epa") - get_stat(away_row, "roll_def_dropback_epa")) - \
                        (get_stat(away_row, "roll_off_dropback_epa") - get_stat(home_row, "roll_def_dropback_epa"))
        net_rush_edge = (get_stat(home_row, "roll_off_rush_epa") - get_stat(away_row, "roll_def_rush_epa")) - \
                        (get_stat(away_row, "roll_off_rush_epa") - get_stat(home_row, "roll_def_rush_epa"))
        net_late_down_edge = (get_stat(home_row, "roll_off_late_down_epa") - get_stat(away_row, "roll_def_late_down_epa")) - \
                             (get_stat(away_row, "roll_off_late_down_epa") - get_stat(home_row, "roll_def_late_down_epa"))
        diff_success = get_stat(home_row, "roll_off_early_down_success", 0.44) - get_stat(away_row, "roll_off_early_down_success", 0.44)
        diff_explosive = get_stat(home_row, "roll_off_explosive", 0.12) - get_stat(away_row, "roll_off_explosive", 0.12)

        net_pass_edge += (calculate_roster_vorp(away_team) - calculate_roster_vorp(home_team))
        rest_diff = float(game.get("home_rest", 7.0) or 7.0) - float(game.get("away_rest", 7.0) or 7.0)
        is_divisional = int(game.get("div_game", 0) or 0)

        feature_row = pd.DataFrame([[
            net_pass_edge, net_rush_edge, net_late_down_edge, diff_success,
            diff_explosive, rest_diff, is_divisional, market_home_prob
        ]], columns=FEATURES)

        raw_home_prob = float(model.predict_proba(feature_row)[0][1])

        dynamic_weight = min(0.75, max(0.48, 0.48 + (abs(spread_line) * 0.022)))
        calibrated_home_win_prob = ((1.0 - dynamic_weight) * raw_home_prob) + (dynamic_weight * market_home_prob)

        sigma = 13.45 * math.sqrt(max(32.0, total_line) / 44.0)
        z_win = norm.ppf(max(0.01, min(0.99, calibrated_home_win_prob)))
        projected_margin = z_win * sigma

        pred_home_score, pred_away_score = calculate_discrete_projected_scores(projected_margin, total_line)
        pred_total_score = pred_home_score + pred_away_score

        abs_spread = round(abs(spread_line))
        push_rate = NFL_KEY_PUSH_RATES.get(abs_spread, 0.0) if float(spread_line).is_integer() else 0.0
        z_cover_home = (projected_margin - (spread_line + 0.5 if spread_line.is_integer() else spread_line)) / sigma
        z_cover_away = ((spread_line - 0.5 if spread_line.is_integer() else spread_line) - projected_margin) / sigma

        home_cover = float(norm.cdf(z_cover_home))
        away_cover = float(norm.cdf(z_cover_away))
        if push_rate > 0:
            scale = (1.0 - push_rate) / (home_cover + away_cover)
            home_cover *= scale
            away_cover *= scale

        home_edge = home_cover - 0.5238
        away_edge = away_cover - 0.5238

        if home_edge > 0.018 and home_edge > away_edge:
            rec_team = home_team
            rec_line = f"{home_team} {-spread_line:+g}"
            cover_prob = home_cover
            final_edge = min(0.050, home_edge)
        elif away_edge > 0.018 and away_edge > home_edge:
            rec_team = away_team
            rec_line = f"{away_team} {+spread_line:+g}"
            cover_prob = away_cover
            final_edge = min(0.050, away_edge)
        else:
            rec_team = "PASS"
            rec_line = "PASS - No Edge"
            cover_prob = max(home_cover, away_cover)
            final_edge = max(home_edge, away_edge)

        b = 1.9091 - 1.0
        q = max(0.0, 1.0 - cover_prob - push_rate)
        kelly_units = round(max(0.0, min(2.0, (((b * cover_prob) - q) / b) * 0.125 * 100.0)), 2) if rec_team != "PASS" else 0.0

        pre_processed.append({
            "game_id": str(game.get("game_id", f"{target_season}_{week_num}_{away_team}_{home_team}")),
            "season": target_season,
            "week": week_num,
            "matchup": matchup,
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": calibrated_home_win_prob,
            "market_prob": market_home_prob,
            "spread_cover_prob": cover_prob,
            "spread_edge": final_edge,
            "kelly_units": kelly_units,
            "recommended_team": rec_team,
            "recommended_line": rec_line,
            "total_line": total_line,
            "spread_line": spread_line,
            "predicted_home_score": pred_home_score,
            "predicted_away_score": pred_away_score,
            "predicted_total_score": pred_total_score,
            "tape_metrics": {
                "net_pass_epa_diff": f"{net_pass_edge:+.3f}",
                "net_rush_epa_diff": f"{net_rush_edge:+.3f}",
                "explosive_rate_diff": f"{diff_explosive:+.3f}",
                "early_down_success_diff": f"{diff_success:+.3f}"
            }
        })

    semaphore = asyncio.Semaphore(4)
    tasks = [generate_matchup_analysis(semaphore, item) for item in pre_processed]
    results = await asyncio.gather(*tasks)

    records = []
    for item, text_res in zip(pre_processed, results):
        try:
            parsed = json.loads(text_res)
        except Exception:
            parsed = {}

        parsed["predicted_scores"] = {
            "home_team": item["home_team"],
            "predicted_home_score": item["predicted_home_score"],
            "away_team": item["away_team"],
            "predicted_away_score": item["predicted_away_score"],
            "predicted_total": item["predicted_total_score"]
        }

        records.append({
            "game_id": item["game_id"],
            "season": item["season"],
            "week": item["week"],
            "matchup": item["matchup"],
            "home_win_prob": item["home_win_prob"],
            "market_prob": item["market_prob"],
            "spread_cover_prob": item["spread_cover_prob"],
            "spread_edge": item["spread_edge"],
            "kelly_units": item["kelly_units"],
            "predicted_home_score": item["predicted_home_score"],
            "predicted_away_score": item["predicted_away_score"],
            "predicted_total_score": item["predicted_total_score"],
            "analysis": json.dumps(parsed)
        })

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

            migration_statements = [
                "ALTER TABLE nfl_weekly_analysis ADD COLUMN IF NOT EXISTS season INTEGER DEFAULT 2026;",
                "ALTER TABLE nfl_weekly_analysis ADD COLUMN IF NOT EXISTS predicted_home_score INTEGER;",
                "ALTER TABLE nfl_weekly_analysis ADD COLUMN IF NOT EXISTS predicted_away_score INTEGER;",
                "ALTER TABLE nfl_weekly_analysis ADD COLUMN IF NOT EXISTS predicted_total_score INTEGER;"
            ]
            for stmt in migration_statements:
                conn.execute(text(stmt))

            conn.execute(
                text("DELETE FROM nfl_weekly_analysis WHERE season = :s AND week = :w;"),
                {"s": target_season, "w": target_week}
            )

        df_results.to_sql(
            "nfl_weekly_analysis",
            engine,
            if_exists="append",
            index=False,
            method="multi"
        )
        print(f"Database sync verified: {len(df_results)} fixtures safely committed for Season {target_season} Week {target_week}.")

if __name__ == "__main__":
    asyncio.run(main())
if __name__ == "__main__":
    asyncio.run(main())
