"""
update_nfl.py - Autonomous Temporal Live Slate Ingestion & Execution Engine.
Calibrated for Dynamic Drive-Level Monte Carlo Scoring & Automated Active Week Ingestion.
"""

import asyncio
from datetime import datetime, timezone, timedelta
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
import nflreadpy as nfl
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import norm
from sqlalchemy import create_engine, text
import xgboost as xgb

from nfl_guru import NFL_GURU_FULL_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def sanitize_db_url(raw_url: str) -> str:
    if not raw_url:
        return ""
    url = raw_url.strip()
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://") and not url.startswith("postgresql+"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url

RAW_DB_URL = os.environ.get("DATABASE_URL", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

if not RAW_DB_URL or not GEMINI_KEY:
    raise ValueError("FATAL: DATABASE_URL and GEMINI_API_KEY must be configured in environment.")

DB_URL = sanitize_db_url(RAW_DB_URL)
engine = create_engine(DB_URL, pool_size=5, max_overflow=10, pool_pre_ping=True)
ai_client = genai.Client(api_key=GEMINI_KEY)

MODEL_FILE = "nfl_model.json"
model = xgb.XGBClassifier()
if os.path.exists(MODEL_FILE):
    model.load_model(MODEL_FILE)
    EXPECTED_FEATURES = model.get_booster().feature_names
else:
    raise FileNotFoundError(f"Required model artifact '{MODEL_FILE}' not found.")

TEAM_ABBR_MAP = {"LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX"}
MIN_BETTABLE_EDGE_PCT = 1.8

LOG_SIGMA = {
    "QB_Pass": 0.32, "QB_Rush": 0.52, "RB_Rush": 0.48,
    "RB_Rec": 0.55, "WR_Rec": 0.58, "TE_Rec": 0.54
}

class SchematicMatchup(BaseModel):
    away_offense_vs_home_defense: str = Field(..., description="Tactical trench, coverage, and explosive play breakdown.")
    home_offense_vs_away_defense: str = Field(..., description="Tactical trench, coverage, and explosive play breakdown.")

class MatchupDossier(BaseModel):
    executive_summary: str = Field(..., description="Two-sentence analytical verdict synthesizing schematic leverage.")
    schematic_matchup: SchematicMatchup
    actionable_verdict: str = Field(..., description="Executable ticket recommendation and stake.")

def clean_team_abbr(team_str: str) -> str:
    if not isinstance(team_str, str):
        return ""
    val = str(team_str).strip().upper()
    return TEAM_ABBR_MAP.get(val, val)

def determine_active_nfl_week(schedules_df: pd.DataFrame) -> Tuple[int, int]:
    now_utc = datetime.now(timezone.utc)
    target_season = 2026

    df_season = schedules_df[schedules_df["season"] == target_season].copy()
    if df_season.empty:
        return target_season, 3

    def parse_kickoff(row):
        gameday = str(row.get("gameday", "")).strip()
        gametime = str(row.get("gametime", "13:00")).strip()
        if not gameday or gameday == "None":
            return datetime(target_season, 9, 27, tzinfo=timezone.utc)
        try:
            time_str = f"{gameday} {gametime}"
            dt_naive = pd.to_datetime(time_str)
            return dt_naive.tz_localize("America/New_York").tz_convert("UTC")
        except Exception:
            return pd.to_datetime(gameday).tz_localize("UTC")

    df_season["kickoff_utc"] = df_season.apply(parse_kickoff, axis=1)
    active_unplayed = df_season[df_season["kickoff_utc"] + timedelta(hours=4) > now_utc]

    if not active_unplayed.empty:
        active_week = int(active_unplayed["week"].min())
    else:
        unplayed = df_season[df_season["result"].isna()]
        active_week = int(unplayed["week"].min()) if not unplayed.empty else 3

    logging.info(f"Temporal calibration: Current UTC {now_utc.isoformat()} -> Target Active NFL Week: {active_week}")
    return target_season, active_week

def resolve_directional_market_context(total_line: float, spread_line: float) -> Tuple[float, float, float, float]:
    home_margin = float(spread_line)
    implied_home = (total_line + home_margin) / 2.0
    implied_away = (total_line - home_margin) / 2.0
    sigma = 13.45 * math.sqrt(max(32.0, total_line) / 44.0)
    market_home_prob = float(norm.cdf(home_margin / sigma))
    return home_margin, round(implied_home, 2), round(implied_away, 2), market_home_prob

def blend_log_odds(p_model: float, p_mkt: float, w_mkt: float = 0.55) -> float:
    eps = 1e-4
    p_mod_clipped = np.clip(p_model, eps, 1.0 - eps)
    p_mkt_clipped = np.clip(p_mkt, eps, 1.0 - eps)
    lo_model = np.log(p_mod_clipped / (1.0 - p_mod_clipped))
    lo_mkt = np.log(p_mkt_clipped / (1.0 - p_mkt_clipped))
    blended_lo = ((1.0 - w_mkt) * lo_model) + (w_mkt * lo_mkt)
    return float(1.0 / (1.0 + np.exp(-blended_lo)))

def simulate_possession_drive(
    off_drop_epa: float,
    def_drop_epa: float,
    off_rush_epa: float,
    def_rush_epa: float,
    off_explosive_rate: float,
    def_explosive_allowed: float,
    trench_pressure_delta: float,
    rz_td_rate: float,
    starting_yardline: float,
    score_diff: int,
    quarter: int,
    clock_seconds: float,
    rng: np.random.Generator
) -> Tuple[int, float, float]:
    yards_to_goal = 100.0 - starting_yardline
    pass_edge = off_drop_epa - def_drop_epa
    rush_edge = off_rush_epa - def_rush_epa
    net_efficiency = (0.65 * pass_edge) + (0.35 * rush_edge) - (0.30 * trench_pressure_delta)
    
    net_explosive = np.clip(off_explosive_rate + (def_explosive_allowed - 0.12), 0.05, 0.25)
    turnover_prob = np.clip(0.10 + (0.09 * trench_pressure_delta) - (0.04 * pass_edge), 0.02, 0.24)
    
    if rng.random() < turnover_prob:
        duration = rng.uniform(30.0, 110.0)
        end_pos = np.clip(starting_yardline + rng.normal(12.0, 8.0), 5.0, 95.0)
        return 0, duration, end_pos

    if rng.random() < net_explosive:
        chunk_yards = rng.uniform(25.0, 75.0)
        yards_to_goal -= chunk_yards
        if yards_to_goal <= 0:
            duration = rng.uniform(20.0, 65.0)
            pat = 7 if (rng.random() < 0.95) else 6
            return pat, duration, 25.0

    down_conversion_prob = np.clip(0.44 + (0.35 * net_efficiency), 0.20, 0.72)
    if quarter == 4 and score_diff < 0:
        down_conversion_prob = np.clip(down_conversion_prob + 0.07, 0.25, 0.78)

    accumulated_yards = 0.0
    drive_plays = 0
    max_series = 6
    
    for series in range(max_series):
        drive_plays += rng.integers(3, 5)
        if rng.random() < down_conversion_prob:
            gain = rng.uniform(10.0, 22.0)
            accumulated_yards += gain
            yards_to_goal -= gain
            if yards_to_goal <= 20.0:
                break
        else:
            break

    duration = np.clip(drive_plays * rng.uniform(24.0, 38.0), 45.0, 360.0)
    final_yardline = 100.0 - yards_to_goal

    if yards_to_goal <= 20.0:
        effective_td_rate = np.clip(rz_td_rate + (0.18 * rush_edge), 0.38, 0.78)
        if rng.random() < effective_td_rate:
            go_for_two = False
            if quarter == 4:
                if score_diff + 6 in [-2, 0, 1]:
                    go_for_two = True
                elif score_diff + 6 == -9 and clock_seconds < 300.0:
                    go_for_two = True

            if go_for_two:
                pts = 8 if (rng.random() < 0.485) else 6
            else:
                pts = 7 if (rng.random() < 0.955) else 6
            return pts, duration, 25.0
        else:
            if quarter == 4 and score_diff < -3 and clock_seconds < 180.0:
                td_gamble = rng.random() < 0.40
                return (7 if td_gamble else 0), duration, 25.0
            return 3, duration, 25.0

    elif yards_to_goal <= 38.0:
        fg_distance = yards_to_goal + 17.0
        fg_prob = np.clip(0.92 - (fg_distance - 35.0) * 0.022, 0.40, 0.88)
        if quarter == 4 and score_diff < -3 and clock_seconds < 180.0:
            return 0, duration, final_yardline
        if rng.random() < fg_prob:
            return 3, duration, 25.0
        else:
            return 0, duration, final_yardline

    net_punt = rng.normal(41.5, 6.0)
    opp_field_pos = np.clip(final_yardline + net_punt, 1.0, 80.0)
    return 0, duration, 100.0 - opp_field_pos

def execute_possession_matchup_sim(
    home_drop_epa: float,
    away_drop_epa: float,
    home_def_drop_epa: float,
    away_def_drop_epa: float,
    home_rush_epa: float,
    away_rush_epa: float,
    home_def_rush_epa: float,
    away_def_rush_epa: float,
    home_explosive: float = 0.12,
    away_explosive: float = 0.11,
    home_def_explosive: float = 0.11,
    away_def_explosive: float = 0.12,
    home_prwr: float = 0.42,
    away_prwr: float = 0.40,
    home_pbwr: float = 0.60,
    away_pbwr: float = 0.58,
    home_rz_td: float = 0.58,
    away_rz_td: float = 0.52,
    home_pace_sec: float = 27.5,
    away_pace_sec: float = 28.2,
    num_simulations: int = 5000,
    seed: int = 42
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    home_scores = np.zeros(num_simulations, dtype=np.int32)
    away_scores = np.zeros(num_simulations, dtype=np.int32)

    home_trench_delta = float(home_prwr - away_pbwr)
    away_trench_delta = float(away_prwr - home_pbwr)

    for i in range(num_simulations):
        game_clock = 3600.0
        h_pts, a_pts = 0, 0
        curr_field_pos = 25.0
        possession = "HOME" if rng.random() < 0.50 else "AWAY"

        while game_clock > 0:
            quarter = 4 - int(game_clock // 900)
            score_diff = (h_pts - a_pts) if possession == "HOME" else (a_pts - h_pts)

            if possession == "HOME":
                pts, duration, next_field_pos = simulate_possession_drive(
                    home_drop_epa, away_def_drop_epa, home_rush_epa, away_def_rush_epa,
                    home_explosive, away_def_explosive, away_trench_delta, home_rz_td,
                    curr_field_pos, score_diff, quarter, game_clock, rng
                )
                h_pts += pts
                possession = "AWAY"
                curr_field_pos = next_field_pos
                pace = home_pace_sec
            else:
                pts, duration, next_field_pos = simulate_possession_drive(
                    away_drop_epa, home_def_drop_epa, away_rush_epa, home_def_rush_epa,
                    away_explosive, home_def_explosive, home_trench_delta, away_rz_td,
                    curr_field_pos, score_diff, quarter, game_clock, rng
                )
                a_pts += pts
                possession = "HOME"
                curr_field_pos = next_field_pos
                pace = away_pace_sec

            if quarter == 4 and score_diff < 0:
                pace = max(18.0, pace - 8.0)
            elif quarter == 4 and score_diff > 0:
                pace = min(38.0, pace + 7.0)

            game_clock -= (duration + pace)

        if h_pts == a_pts:
            if rng.random() < 0.53:
                h_pts += 3
            else:
                a_pts += 3

        home_scores[i] = h_pts
        away_scores[i] = a_pts

    margins = home_scores - away_scores
    totals = home_scores + away_scores

    pred_home = int(np.round(np.median(home_scores)))
    pred_away = int(np.round(np.median(away_scores)))
    
    if pred_home == pred_away:
        if np.mean(margins) >= 0:
            pred_home += 3
        else:
            pred_away += 3

    return {
        "predicted_home_score": pred_home,
        "predicted_away_score": pred_away,
        "mean_margin": float(np.mean(margins)),
        "mean_total": float(np.mean(totals)),
        "home_win_prob": float(np.mean(margins > 0)),
        "margins_raw": margins
    }

def compute_monte_carlo_ats_metrics(
    margins: np.ndarray,
    canonical_spread: float
) -> Dict[str, float]:
    hurdle = -float(canonical_spread)
    pushes = np.isclose(margins, hurdle, atol=1e-5)
    home_covers = margins > hurdle
    away_covers = margins < hurdle

    p_push = float(np.mean(pushes))
    p_home_cover = float(np.mean(home_covers))
    p_away_cover = float(np.mean(away_covers))

    break_even = 0.5238
    home_net_edge = p_home_cover - break_even
    away_net_edge = p_away_cover - break_even

    b = 0.90909
    def calc_kelly(p_win: float) -> float:
        q = max(0.0, 1.0 - p_win - p_push)
        raw_kelly = ((b * p_win) - q) / b
        return round(max(0.0, min(2.0, raw_kelly * 0.125 * 100.0)), 2)

    if home_net_edge >= (MIN_BETTABLE_EDGE_PCT / 100.0) and home_net_edge > away_net_edge:
        rec_side = "HOME"
        cover_prob = p_home_cover
        final_edge = min(0.080, home_net_edge)
        stake_units = calc_kelly(p_home_cover)
    elif away_net_edge >= (MIN_BETTABLE_EDGE_PCT / 100.0) and away_net_edge > home_net_edge:
        rec_side = "AWAY"
        cover_prob = p_away_cover
        final_edge = min(0.080, away_net_edge)
        stake_units = calc_kelly(p_away_cover)
    else:
        rec_side = "PASS"
        cover_prob = max(p_home_cover, p_away_cover)
        final_edge = max(home_net_edge, away_net_edge)
        stake_units = 0.0

    return {
        "home_cover_prob": round(p_home_cover, 4),
        "away_cover_prob": round(p_away_cover, 4),
        "push_prob": round(p_push, 4),
        "recommended_side": rec_side,
        "spread_cover_prob": round(cover_prob, 4),
        "spread_edge": round(final_edge, 4),
        "kelly_units": float(stake_units)
    }

def convert_mean_to_median(mean_val: float, role_key: str) -> float:
    if mean_val <= 0.0:
        return 0.0
    sig = LOG_SIGMA.get(role_key, 0.50)
    return round(max(0.0, float(mean_val * math.exp(-(sig ** 2) / 2.0))), 1)

def calculate_lognormal_cover_probability(mean_val: float, line: float, sigma: float) -> float:
    if mean_val <= 0.0 or line <= 0.0:
        return 0.0
    mu = math.log(mean_val) - (sigma ** 2) / 2.0
    z = (math.log(line) - mu) / sigma
    return round(1.0 - float(norm.cdf(z)), 4)

def filter_neutral_game_states(pbp_df: pd.DataFrame) -> pd.DataFrame:
    if pbp_df.empty:
        return pd.DataFrame()
    clean = pbp_df[pbp_df["play_type"].isin(["pass", "run"])].copy()
    if "home_wp" in clean.columns and "score_differential" in clean.columns and "qtr" in clean.columns:
        return clean[
            (clean["home_wp"].between(0.10, 0.90)) &
            ~((clean["qtr"] == 4) & (clean["score_differential"].abs() >= 16))
        ]
    return clean

def compute_opponent_adjusted_epa(pbp_df: pd.DataFrame) -> pd.DataFrame:
    neutral_pbp = filter_neutral_game_states(pbp_df)
    if neutral_pbp.empty:
        return pd.DataFrame()

    neutral_pbp["is_early_down"] = neutral_pbp["down"].isin([1, 2]).astype(int) if "down" in neutral_pbp.columns else 1
    neutral_pbp["is_late_down"] = neutral_pbp["down"].isin([3, 4]).astype(int) if "down" in neutral_pbp.columns else 0
    neutral_pbp["is_explosive"] = (
        ((neutral_pbp["play_type"] == "pass") & (neutral_pbp["yards_gained"] >= 15)) |
        ((neutral_pbp["play_type"] == "run") & (neutral_pbp["yards_gained"] >= 10))
    ).astype(int)

    off_stats = neutral_pbp.groupby(["season", "week", "posteam"]).agg(
        off_dropback_epa=("epa", lambda x: x[neutral_pbp.loc[x.index, "play_type"] == "pass"].mean()),
        off_rush_epa=("epa", lambda x: x[neutral_pbp.loc[x.index, "play_type"] == "run"].mean()),
        off_early_down_success=("success", lambda x: x[neutral_pbp.loc[x.index, "is_early_down"] == 1].mean()),
        off_late_down_epa=("epa", lambda x: x[neutral_pbp.loc[x.index, "is_late_down"] == 1].mean()),
        off_explosive=("is_explosive", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})

    def_stats = neutral_pbp.groupby(["season", "week", "defteam"]).agg(
        def_dropback_epa=("epa", lambda x: x[neutral_pbp.loc[x.index, "play_type"] == "pass"].mean()),
        def_rush_epa=("epa", lambda x: x[neutral_pbp.loc[x.index, "play_type"] == "run"].mean()),
        def_early_down_success=("success", lambda x: x[neutral_pbp.loc[x.index, "is_early_down"] == 1].mean()),
        def_late_down_epa=("epa", lambda x: x[neutral_pbp.loc[x.index, "is_late_down"] == 1].mean()),
        def_explosive=("is_explosive", "mean"),
    ).reset_index().rename(columns={"defteam": "team"})

    merged = pd.merge(off_stats, def_stats, on=["season", "week", "team"], how="outer").fillna(0)
    merged.sort_values(["team", "season", "week"], inplace=True)
    for col in [
        "off_dropback_epa", "off_rush_epa", "off_early_down_success", "off_late_down_epa", "off_explosive",
        "def_dropback_epa", "def_rush_epa", "def_early_down_success", "def_late_down_epa", "def_explosive"
    ]:
        merged[f"roll_{col}"] = merged.groupby("team")[col].transform(lambda x: x.shift(1).ewm(span=6, min_periods=1).mean())
    return merged

def extract_empirical_team_pace(pbp_df: pd.DataFrame, team_abbr: str) -> Dict[str, float]:
    neutral_pbp = filter_neutral_game_states(pbp_df)
    if neutral_pbp.empty:
        return {"neutral_plays": 63.5, "neutral_pass_rate": 0.56, "ypa": 7.10, "ypc": 4.20, "pace_sec": 27.5}
    t_pbp = neutral_pbp[neutral_pbp["posteam"] == team_abbr]
    if t_pbp.empty:
        return {"neutral_plays": 63.5, "neutral_pass_rate": 0.56, "ypa": 7.10, "ypc": 4.20, "pace_sec": 27.5}

    recent = t_pbp[t_pbp["week"].isin(sorted(t_pbp["week"].unique())[-6:])]
    n_plays = len(recent)
    passes = recent[recent["play_type"] == "pass"]
    runs = recent[recent["play_type"] == "run"]

    return {
        "neutral_plays": float(max(55.0, min(75.0, n_plays / max(1, recent["game_id"].nunique())))),
        "neutral_pass_rate": float(max(0.44, min(0.70, len(passes) / max(1, n_plays)))),
        "ypa": float(max(5.5, min(9.5, passes["yards_gained"].mean() if not passes.empty else 7.10))),
        "ypc": float(max(3.2, min(5.8, runs["yards_gained"].mean() if not runs.empty else 4.20))),
        "pace_sec": float(max(22.0, min(33.0, 27.5)))
    }

def generate_closed_loop_skill_projections(
    team_abbr: str, implied_total: float, team_spread_margin: float,
    pass_edge: float, rush_edge: float, depth_names: Dict[str, str],
    pbp_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    pace = extract_empirical_team_pace(pbp_df, team_abbr)

    script_logit = 1.0 / (1.0 + math.exp(0.12 * team_spread_margin))
    pass_rate = max(0.48, min(0.68, pace["neutral_pass_rate"] + 0.10 * (script_logit - 0.50) + 0.025 * (pass_edge - rush_edge)))
    run_rate = 1.0 - pass_rate

    trailing_drag = 0.0 if team_spread_margin >= 0 else max(-0.85, team_spread_margin * 0.045)
    ypa = max(5.8, min(8.8, 7.35 + (pass_edge * 2.8) + trailing_drag))
    ypc = max(3.4, min(5.4, 4.30 + (rush_edge * 2.2)))

    base_plays = pace["neutral_plays"] * (implied_total / 22.0) ** 0.25
    gross_pass_mean = base_plays * pass_rate * ypa
    gross_rush_mean = base_plays * run_rate * ypc
    gross_total = gross_pass_mean + gross_rush_mean

    min_yds = implied_total * 12.5
    max_yds = implied_total * 16.5
    if gross_total < min_yds:
        scale = min_yds / max(1.0, gross_total)
        gross_pass_mean *= scale
        gross_rush_mean *= scale
    elif gross_total > max_yds:
        scale = max_yds / gross_total
        gross_pass_mean *= scale
        gross_rush_mean *= scale

    team_td_budget = max(0.85, (implied_total * 0.78) / 7.0)
    pass_td_share = max(0.38, min(0.75, 0.58 + (pass_edge - rush_edge) * 0.15))
    team_pass_tds = team_td_budget * pass_td_share
    team_rush_tds = team_td_budget * (1.0 - pass_td_share)

    roles = ["WR1", "WR2", "WR3", "TE1", "RB1"]
    emp_targets = {"WR1": 0.29, "WR2": 0.20, "WR3": 0.13, "TE1": 0.22, "RB1": 0.16}
    weighted_rec = {r: emp_targets[r] * (ypa / max(1.0, ypa)) for r in roles}
    norm_rec_shares = {r: weighted_rec[r] / sum(weighted_rec.values()) for r in roles}
    rec_means = {r: gross_pass_mean * norm_rec_shares[r] for r in roles}

    rush_means = {"RB1": gross_rush_mean * 0.70, "QB1": gross_rush_mean * 0.15}

    def calc_anytime_td(lam: float) -> float:
        return round(float((1.0 - math.exp(-max(0.001, lam))) * 100.0), 1)

    def synthesize_line(stat_type: str, role: str, model_med: float) -> float:
        if stat_type == "Pass Yds":
            base = implied_total * 9.85
        elif stat_type == "Rush Yds":
            base = implied_total * (2.55 if role == "RB1" else 1.10)
        else:
            base = implied_total * (2.65 if role == "WR1" else (1.75 if role == "TE1" else 1.65))
        return float(max(0.5, round(((0.60 * base) + (0.40 * model_med)) * 2.0) / 2.0))

    def build_entry(role: str, player: str, stat_type: str, model_med: float, p_yds: float, r_yds: float, rc_yds: float, td_lam: float, p_over: float) -> dict:
        sb_line = synthesize_line(stat_type, role, model_med)
        delta = round(model_med - sb_line, 1)
        rec = f"OVER ({p_over*100:.1f}%)" if p_over >= 0.555 else (f"UNDER ({(1.0 - p_over)*100:.1f}%)" if p_over <= 0.445 else "PASS")
        return {
            "role": role, "player": player, "primary_stat_type": stat_type,
            "model_median": model_med, "sportsbook_line": sb_line, "edge_delta": delta,
            "prop_recommendation": rec, "pass_yards": p_yds, "rush_yards": r_yds,
            "rec_yards": rc_yds, "total_tds": round(td_lam, 2), "anytime_td_prob": calc_anytime_td(td_lam)
        }

    qb_pass = convert_mean_to_median(gross_pass_mean, "QB_Pass")
    qb_rush = convert_mean_to_median(rush_means["QB1"], "QB_Rush")
    rb1_rush = convert_mean_to_median(rush_means["RB1"], "RB_Rush")
    rb1_rec = convert_mean_to_median(rec_means["RB1"], "RB_Rec")
    wr1_rec = convert_mean_to_median(rec_means["WR1"], "WR_Rec")
    wr2_rec = convert_mean_to_median(rec_means["WR2"], "WR_Rec")
    te1_rec = convert_mean_to_median(rec_means["TE1"], "TE_Rec")

    p_qb = calculate_lognormal_cover_probability(gross_pass_mean, synthesize_line("Pass Yds", "QB1", qb_pass), LOG_SIGMA["QB_Pass"])
    p_rb1 = calculate_lognormal_cover_probability(rush_means["RB1"], synthesize_line("Rush Yds", "RB1", rb1_rush), LOG_SIGMA["RB_Rush"])
    p_wr1 = calculate_lognormal_cover_probability(rec_means["WR1"], synthesize_line("Rec Yds", "WR1", wr1_rec), LOG_SIGMA["WR_Rec"])
    p_wr2 = calculate_lognormal_cover_probability(rec_means["WR2"], synthesize_line("Rec Yds", "WR2", wr2_rec), LOG_SIGMA["WR_Rec"])
    p_te1 = calculate_lognormal_cover_probability(rec_means["TE1"], synthesize_line("Rec Yds", "TE1", te1_rec), LOG_SIGMA["TE_Rec"])

    return [
        build_entry("WR1", depth_names.get("WR1", f"{team_abbr} WR1"), "Rec Yds", wr1_rec, 0.0, 0.0, wr1_rec, team_pass_tds * 0.35, p_wr1),
        build_entry("WR2", depth_names.get("WR2", f"{team_abbr} WR2"), "Rec Yds", wr2_rec, 0.0, 0.0, wr2_rec, team_pass_tds * 0.20, p_wr2),
        build_entry("TE1", depth_names.get("TE1", f"{team_abbr} TE1"), "Rec Yds", te1_rec, 0.0, 0.0, te1_rec, team_pass_tds * 0.25, p_te1),
        build_entry("RB1", depth_names.get("RB1", f"{team_abbr} RB1"), "Rush Yds", rb1_rush, 0.0, rb1_rush, rb1_rec, team_rush_tds * 0.65, p_rb1),
        build_entry("RB1_REC", f"{depth_names.get('RB1', 'RB1')} (Rec)", "Rec Yds", rb1_rec, 0.0, 0.0, rb1_rec, 0.0, p_rb1),
        build_entry("QB1", depth_names.get("QB1", f"{team_abbr} QB"), "Pass Yds", qb_pass, qb_pass, qb_rush, 0.0, team_rush_tds * 0.15, p_qb),
    ]

def resolve_depth(depth_charts: pd.DataFrame, team: str) -> Dict[str, str]:
    picks = {"QB1": f"{team} QB", "RB1": f"{team} RB1", "WR1": f"{team} WR1", "WR2": f"{team} WR2", "TE1": f"{team} TE1"}
    if depth_charts.empty:
        return picks
    t_dc = depth_charts[depth_charts["club_code"] == team] if "club_code" in depth_charts.columns else pd.DataFrame()
    if t_dc.empty:
        return picks
    for pos, role in [("QB", "QB1"), ("RB", "RB1"), ("WR", "WR1"), ("TE", "TE1")]:
        cands = t_dc[t_dc["pos_abb"] == pos].sort_values("pos_rank", ascending=True) if "pos_abb" in t_dc.columns else pd.DataFrame()
        if not cands.empty:
            picks[role] = str(cands.iloc[0].get("player_name", f"{team} {role}"))
    return picks

async def main():
    logging.info("Querying live nflverse schedules for Season 2026...")
    try:
        schedules = nfl.load_schedules(seasons=[2026]).to_pandas()
    except Exception as e:
        logging.error(f"Schedule pull failed: {e}")
        sys.exit(1)

    for col in ["home_team", "away_team"]:
        if col in schedules.columns:
            schedules[col] = schedules[col].apply(clean_team_abbr)

    target_season, target_week = determine_active_nfl_week(schedules)
    upcoming_slate = schedules[(schedules["season"] == target_season) & (schedules["week"] == target_week)].copy()

    logging.info(f"Targeting Season {target_season} Week {target_week} with {len(upcoming_slate)} active fixture(s)...")

    try:
        pbp = nfl.load_pbp(seasons=[target_season - 1, target_season]).to_pandas()
    except Exception:
        pbp = pd.DataFrame()

    try:
        depth_charts = nfl.load_depth_charts(seasons=[target_season]).to_pandas()
        if "club_code" in depth_charts.columns:
            depth_charts["club_code"] = depth_charts["club_code"].apply(clean_team_abbr)
    except Exception:
        depth_charts = pd.DataFrame()

    team_perf = compute_opponent_adjusted_epa(pbp)
    pre_processed = []

    for _, game in upcoming_slate.iterrows():
        home_team = clean_team_abbr(str(game["home_team"]))
        away_team = clean_team_abbr(str(game["away_team"]))
        matchup = f"{away_team} @ {home_team}"
        game_id = str(game.get("game_id", f"{target_season}_{target_week:02d}_{away_team}_{home_team}"))

        raw_spread = float(game.get("spread_line", 0.0) or 0.0)
        raw_total = float(game.get("total_line", 44.0) or 44.0)

        def get_stat(team_abbr: str, col: str) -> float:
            if team_perf.empty:
                return 0.0
            row = team_perf[team_perf["team"] == team_abbr]
            if row.empty:
                return 0.0
            val = row.sort_values(["season", "week"], ascending=[False, False]).iloc[0].get(col, 0.0)
            return float(val) if pd.notna(val) else 0.0

        h_off_drop = get_stat(home_team, "roll_off_dropback_epa")
        a_off_drop = get_stat(away_team, "roll_off_dropback_epa")
        h_def_drop = get_stat(home_team, "roll_def_dropback_epa")
        a_def_drop = get_stat(away_team, "roll_def_dropback_epa")

        h_off_rush = get_stat(home_team, "roll_off_rush_epa")
        a_off_rush = get_stat(away_team, "roll_off_rush_epa")
        h_def_rush = get_stat(home_team, "roll_def_rush_epa")
        a_def_rush = get_stat(away_team, "roll_def_rush_epa")

        h_off_expl = get_stat(home_team, "roll_off_explosive") or 0.12
        a_off_expl = get_stat(away_team, "roll_off_explosive") or 0.11
        h_def_expl = get_stat(home_team, "roll_def_explosive") or 0.11
        a_def_expl = get_stat(away_team, "roll_def_explosive") or 0.12

        net_pass = (h_off_drop - a_def_drop) - (a_off_drop - h_def_drop)
        net_rush = (h_off_rush - a_def_rush) - (a_off_rush - h_def_rush)
        net_late = (get_stat(home_team, "roll_off_late_down_epa") - get_stat(away_team, "roll_def_late_down_epa")) - \
                   (get_stat(away_team, "roll_off_late_down_epa") - get_stat(home_team, "roll_def_late_down_epa"))

        diff_succ = get_stat(home_team, "roll_off_early_down_success") - get_stat(away_team, "roll_off_early_down_success")
        diff_expl = h_off_expl - a_off_expl
        rest_diff = float(game.get("home_rest", 7.0) or 7.0) - float(game.get("away_rest", 7.0) or 7.0)
        is_div = int(game.get("div_game", 0) or 0)

        canonical_spread, implied_home, implied_away, market_prob = resolve_directional_market_context(raw_total, raw_spread)

        feature_dict = {
            "net_pass_edge": net_pass, "net_rush_edge": net_rush, "net_late_down_edge": net_late,
            "diff_success": diff_succ, "diff_explosive": diff_expl, "rest_diff": rest_diff,
            "is_divisional": is_div, "market_home_prob": market_prob
        }

        feature_row = pd.DataFrame([[feature_dict[f] for f in EXPECTED_FEATURES]], columns=EXPECTED_FEATURES)
        raw_prob = float(model.predict_proba(feature_row)[0][1])

        calibrated_win_prob = blend_log_odds(raw_prob, market_prob, w_mkt=0.55)

        h_pace = extract_empirical_team_pace(pbp, home_team)
        a_pace = extract_empirical_team_pace(pbp, away_team)

        sim_res = execute_possession_matchup_sim(
            home_drop_epa=h_off_drop,
            away_drop_epa=a_off_drop,
            home_def_drop_epa=h_def_drop,
            away_def_drop_epa=a_def_drop,
            home_rush_epa=h_off_rush,
            away_rush_epa=a_off_rush,
            home_def_rush_epa=h_def_rush,
            away_def_rush_epa=a_def_rush,
            home_explosive=h_off_expl,
            away_explosive=a_off_expl,
            home_def_explosive=h_def_expl,
            away_def_explosive=a_def_expl,
            home_prwr=0.42,
            away_prwr=0.40,
            home_pbwr=0.60,
            away_pbwr=0.58,
            home_rz_td=0.58,
            away_rz_td=0.52,
            home_pace_sec=h_pace["pace_sec"],
            away_pace_sec=a_pace["pace_sec"],
            num_simulations=5000
        )

        pred_home = sim_res["predicted_home_score"]
        pred_away = sim_res["predicted_away_score"]
        pred_total = pred_home + pred_away
        model_projected_margin = sim_res["mean_margin"]

        ats_metrics = compute_monte_carlo_ats_metrics(sim_res["margins_raw"], canonical_spread)
        rec_side = ats_metrics["recommended_side"]

        if rec_side == "HOME":
            rec_team = home_team
            rec_line = f"{home_team} {-canonical_spread:+g}"
        elif rec_side == "AWAY":
            rec_team = away_team
            rec_line = f"{away_team} {+canonical_spread:+g}"
        else:
            rec_team = "PASS"
            rec_line = "PASS - No Edge"

        cover_prob = ats_metrics["spread_cover_prob"]
        final_edge = ats_metrics["spread_edge"]
        kelly_units = ats_metrics["kelly_units"]

        home_depth = resolve_depth(depth_charts, home_team)
        away_depth = resolve_depth(depth_charts, away_team)

        home_skills = generate_closed_loop_skill_projections(home_team, implied_home, model_projected_margin, net_pass, net_rush, home_depth, pbp)
        away_skills = generate_closed_loop_skill_projections(away_team, implied_away, -model_projected_margin, -net_pass, -net_rush, away_depth, pbp)

        pre_processed.append({
            "game_id": game_id, "season": target_season, "week": target_week,
            "matchup": matchup, "home_team": home_team, "away_team": away_team,
            "home_win_prob": calibrated_win_prob, "market_prob": market_prob,
            "spread_cover_prob": cover_prob, "spread_edge": final_edge,
            "kelly_units": kelly_units, "recommended_team": rec_team,
            "recommended_line": rec_line, "total_line": raw_total,
            "spread_line": model_projected_margin,
            "predicted_home_score": pred_home, "predicted_away_score": pred_away,
            "predicted_total_score": pred_total,
            "push_prob": ats_metrics["push_prob"],
            "player_projections": {"home": home_skills, "away": away_skills}
        })

    semaphore = asyncio.Semaphore(5)

    async def generate_matchup_analysis(item):
        verdict_str = f"Bet {item['recommended_line']} - {item['kelly_units']:.2f}u" if item['recommended_team'] != "PASS" and item['kelly_units'] > 0.0 else "PASS - 0.00u"
        prompt = f"""[MODE 1: NFL TACTICAL & STATISTICAL BREAKDOWN]
SUBJECT: {item['matchup']} Quantitative Evaluation Week {item['week']}
DOSSIER PAYLOAD:
{json.dumps(item, indent=2)}

TASK:
Provide an institutional film and sabermetric analysis detailing:
1. Executive summary evaluating whether the line value represents actionable market inefficiency.
2. Schematic breakdown for both offensive dropback scripts against opponent coverage shells.
3. Actionable verdict string confirming: '{verdict_str}'."""

        async with semaphore:
            for attempt in range(3):
                try:
                    response = await ai_client.aio.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=NFL_GURU_FULL_SYSTEM_PROMPT,
                            temperature=0.15,
                            response_mime_type="application/json",
                            response_schema=MatchupDossier
                        )
                    )
                    parsed = json.loads(response.text)
                    parsed["actionable_verdict"] = verdict_str
                    return json.dumps(parsed)
                except Exception as ex:
                    logging.warning(f"Async LLM inference attempt {attempt+1} failed for {item['matchup']}: {ex}")
                    await asyncio.sleep(2 ** attempt)

            fallback = MatchupDossier(
                executive_summary=f"Line-of-scrimmage leverage on neutral downs establishes baseline edge on {item['matchup']}.",
                schematic_matchup=SchematicMatchup(
                    away_offense_vs_home_defense=f"{item['away_team']} must sustain early-down push to keep dropbacks on schedule.",
                    home_offense_vs_away_defense=f"{item['home_team']} attacks intermediate boundary voids against split-safety shells."
                ),
                actionable_verdict=verdict_str
            )
            return fallback.model_dump_json()

    results = await asyncio.gather(*[generate_matchup_analysis(item) for item in pre_processed])

    records = []
    for item, text_res in zip(pre_processed, results):
        try:
            parsed = json.loads(text_res)
        except Exception:
            parsed = {}

        parsed["predicted_scores"] = {
            "home_team": item["home_team"], "predicted_home_score": item["predicted_home_score"],
            "away_team": item["away_team"], "predicted_away_score": item["predicted_away_score"],
            "predicted_total": item["predicted_total_score"]
        }
        parsed["player_projections"] = item["player_projections"]

        records.append({
            "game_id": item["game_id"], "season": item["season"], "week": item["week"],
            "matchup": item["matchup"], "home_win_prob": item["home_win_prob"],
            "market_prob": item["market_prob"], "spread_cover_prob": item["spread_cover_prob"],
            "spread_edge": item["spread_edge"], "kelly_units": item["kelly_units"],
            "predicted_home_score": item["predicted_home_score"],
            "predicted_away_score": item["predicted_away_score"],
            "predicted_total_score": item["predicted_total_score"],
            "analysis": json.dumps(parsed)
        })

    df_results = pd.DataFrame(records)
    df_results = df_results.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS nfl_weekly_analysis (
                game_id TEXT PRIMARY KEY, season INTEGER DEFAULT 2026, week INTEGER,
                matchup TEXT, home_win_prob NUMERIC, market_prob NUMERIC,
                spread_cover_prob NUMERIC, spread_edge NUMERIC, kelly_units NUMERIC,
                predicted_home_score INTEGER, predicted_away_score INTEGER,
                predicted_total_score INTEGER, analysis TEXT
            );
        """))
        conn.execute(
            text("DELETE FROM nfl_weekly_analysis WHERE season = :s AND week = :w;"),
            {"s": target_season, "w": target_week}
        )
        df_results.to_sql("nfl_weekly_analysis", con=conn, if_exists="append", index=False, method="multi")

    logging.info(f"Database successfully updated with Week {target_week} fixtures: {df_results['matchup'].tolist()}")
    return target_season, target_week

if __name__ == "__main__":
    asyncio.run(main())
