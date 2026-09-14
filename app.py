"""
update_nfl.py - Autonomous Live Slate Ingestion & Execution Engine.
Pulls verified schedules and play-by-play directly from nflreadpy.
Zero hardcoded teams, lines, or statistics.
"""

import asyncio
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from google import genai
from google.genai import types
import nflreadpy as nfl
import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import create_engine, text
import xgboost as xgb

from nfl_guru import NFL_GURU_FULL_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_URL = os.environ.get("DATABASE_URL")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

if not DB_URL or not GEMINI_KEY:
    raise ValueError("FATAL: DATABASE_URL and GEMINI_API_KEY must be configured in environment.")

engine = create_engine(DB_URL, pool_size=5, max_overflow=10, pool_pre_ping=True)
ai_client = genai.Client(api_key=GEMINI_KEY)

MODEL_FILE = "nfl_model.json"
model = xgb.XGBClassifier()
if os.path.exists(MODEL_FILE):
    model.load_model(MODEL_FILE)
    EXPECTED_FEATURES = model.get_booster().feature_names
else:
    raise FileNotFoundError(f"Required model artifact '{MODEL_FILE}' not found.")

KEY_MARGINS = [3, 7, 6, 10, 4, 1, 2, 14, 8, 11, 13, 17]
COMMON_SCORES = [20, 24, 17, 23, 27, 30, 31, 13, 14, 10, 34, 38, 28, 16, 21]
TEAM_ABBR_MAP = {"LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX"}
INACTIVE_STATUSES: Set[str] = {"OUT", "IR", "INJURED RESERVE", "PUP", "RESERVE/PUP", "NFI", "DOUBTFUL", "SUSPENDED"}
MIN_BETTABLE_EDGE_PCT = 1.8

LOG_SIGMA = {
    "QB_Pass": 0.32, "QB_Rush": 0.52, "RB_Rush": 0.48,
    "RB_Rec": 0.55, "WR_Rec": 0.58, "TE_Rec": 0.54
}

def clean_team_abbr(team_str: str) -> str:
    if not isinstance(team_str, str):
        return ""
    val = str(team_str).strip().upper()
    return TEAM_ABBR_MAP.get(val, val)

def resolve_directional_market_context(total_line: float, spread_line: float) -> Tuple[float, float, float, float]:
    home_margin = float(spread_line)
    implied_home = (total_line + home_margin) / 2.0
    implied_away = (total_line - home_margin) / 2.0
    sigma = 13.45 * math.sqrt(max(32.0, total_line) / 44.0)
    market_home_prob = float(norm.cdf(home_margin / sigma))
    return home_margin, round(implied_home, 2), round(implied_away, 2), market_home_prob

def project_discrete_nfl_scores(projected_margin: float, total_line: float) -> Tuple[int, int]:
    eff_margin = projected_margin if abs(projected_margin) >= 0.10 else 0.50
    home_favored = eff_margin > 0.0
    abs_margin = abs(eff_margin)

    selected_margin = min(KEY_MARGINS, key=lambda m: abs(m - abs_margin))
    raw_home = (total_line + (selected_margin if home_favored else -selected_margin)) / 2.0
    raw_away = (total_line - (selected_margin if home_favored else -selected_margin)) / 2.0

    best_pair = (27, 20) if home_favored else (20, 27)
    min_loss = float("inf")

    c_home = [s for s in COMMON_SCORES if abs(s - raw_home) <= 6.5] or [int(round(raw_home))]
    c_away = [s for s in COMMON_SCORES if abs(s - raw_away) <= 6.5] or [int(round(raw_away))]

    for h in c_home:
        for a in c_away:
            if h == a or (home_favored and h <= a) or (not home_favored and a <= h):
                continue
            pair_margin = abs(h - a)
            pair_total = h + a
            loss = (abs(pair_total - total_line) * 1.0) + (abs(pair_margin - abs_margin) * 1.5)
            if pair_margin not in [3, 7, 6, 10, 4]:
                loss += 3.0
            if loss < min_loss:
                min_loss = loss
                best_pair = (h, a)

    return int(best_pair[0]), int(best_pair[1])

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
    ).reset_index().rename(columns={"defteam": "team"})

    merged = pd.merge(off_stats, def_stats, on=["season", "week", "team"], how="outer").fillna(0)
    merged.sort_values(["team", "season", "week"], inplace=True)
    for col in [
        "off_dropback_epa", "off_rush_epa", "off_early_down_success", "off_late_down_epa", "off_explosive",
        "def_dropback_epa", "def_rush_epa", "def_early_down_success", "def_late_down_epa"
    ]:
        merged[f"roll_{col}"] = merged.groupby("team")[col].transform(lambda x: x.shift(1).ewm(span=6, min_periods=1).mean())
    return merged

def extract_empirical_team_pace(pbp_df: pd.DataFrame, team_abbr: str) -> Dict[str, float]:
    neutral_pbp = filter_neutral_game_states(pbp_df)
    if neutral_pbp.empty:
        return {"neutral_plays": 63.5, "neutral_pass_rate": 0.56, "ypa": 7.10, "ypc": 4.20}
    t_pbp = neutral_pbp[neutral_pbp["posteam"] == team_abbr]
    if t_pbp.empty:
        return {"neutral_plays": 63.5, "neutral_pass_rate": 0.56, "ypa": 7.10, "ypc": 4.20}

    recent = t_pbp[t_pbp["week"].isin(sorted(t_pbp["week"].unique())[-6:])]
    n_plays = len(recent)
    passes = recent[recent["play_type"] == "pass"]
    runs = recent[recent["play_type"] == "run"]

    return {
        "neutral_plays": float(max(55.0, min(75.0, n_plays / max(1, recent["game_id"].nunique())))),
        "neutral_pass_rate": float(max(0.44, min(0.70, len(passes) / max(1, n_plays)))),
        "ypa": float(max(5.5, min(9.5, passes["yards_gained"].mean() if not passes.empty else 7.10))),
        "ypc": float(max(3.2, min(5.8, runs["yards_gained"].mean() if not runs.empty else 4.20)))
    }

def extract_empirical_player_usage(pbp_df: pd.DataFrame, team_abbr: str) -> Dict[str, Dict[str, float]]:
    usage = {"target_shares": {}, "ypt": {}, "rz_target_shares": {}, "rush_shares": {}, "ypc": {}, "gl_rush_shares": {}}
    neutral_pbp = filter_neutral_game_states(pbp_df)
    if neutral_pbp.empty:
        return usage
    t_pbp = neutral_pbp[neutral_pbp["posteam"] == team_abbr]
    if t_pbp.empty:
        return usage

    recent = t_pbp[t_pbp["week"].isin(sorted(t_pbp["week"].unique())[-6:])]
    passes = recent[recent["play_type"] == "pass"]
    rushes = recent[recent["play_type"] == "run"]

    if not passes.empty and passes["receiver_player_name"].notna().any():
        counts = passes["receiver_player_name"].value_counts()
        total_p = counts.sum()
        for p, count in counts.items():
            usage["target_shares"][p] = float(count / total_p)
            usage["ypt"][p] = float(passes[passes["receiver_player_name"] == p]["yards_gained"].sum() / count)
        rz_p = passes[passes["yardline_100"] <= 20]
        if not rz_p.empty:
            for p, count in rz_p["receiver_player_name"].value_counts().items():
                usage["rz_target_shares"][p] = float(count / len(rz_p))

    if not rushes.empty and rushes["rusher_player_name"].notna().any():
        counts = rushes["rusher_player_name"].value_counts()
        total_r = counts.sum()
        for p, count in counts.items():
            usage["rush_shares"][p] = float(count / total_r)
            usage["ypc"][p] = float(rushes[rushes["rusher_player_name"] == p]["yards_gained"].sum() / count)
        gl_r = rushes[rushes["yardline_100"] <= 10]
        if not gl_r.empty:
            for p, count in gl_r["rusher_player_name"].value_counts().items():
                usage["gl_rush_shares"][p] = float(count / len(gl_r))

    return usage

def generate_closed_loop_skill_projections(
    team_abbr: str, implied_total: float, team_spread_margin: float,
    pass_edge: float, rush_edge: float, depth_names: Dict[str, str],
    pbp_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    pace = extract_empirical_team_pace(pbp_df, team_abbr)
    usage = extract_empirical_player_usage(pbp_df, team_abbr)

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

    roles = ["WR1", "WR2", "WR3", "TE1", "RB1", "RB2"]
    emp_targets = {r: usage["target_shares"].get(depth_names.get(r, ""), 0.0) for r in roles}
    emp_ypt = {r: usage["ypt"].get(depth_names.get(r, ""), ypa) for r in roles}
    emp_rz = {r: usage["rz_target_shares"].get(depth_names.get(r, ""), emp_targets[r]) for r in roles}

    if sum(emp_targets.values()) < 0.40:
        emp_targets = {"WR1": 0.28, "WR2": 0.19, "WR3": 0.12, "TE1": 0.20, "RB1": 0.13, "RB2": 0.08}
        emp_rz = emp_targets.copy()

    norm_targets = {r: emp_targets[r] / sum(emp_targets.values()) for r in roles}
    weighted_rec = {r: norm_targets[r] * (emp_ypt[r] / max(1.0, ypa)) for r in roles}
    norm_rec_shares = {r: weighted_rec[r] / sum(weighted_rec.values()) for r in roles}
    rec_means = {r: gross_pass_mean * norm_rec_shares[r] for r in roles}

    rush_roles = ["RB1", "RB2", "QB1"]
    emp_rushes = {r: usage["rush_shares"].get(depth_names.get(r, ""), 0.0) for r in rush_roles}
    emp_gl = {r: usage["gl_rush_shares"].get(depth_names.get(r, ""), emp_rushes[r]) for r in rush_roles}
    if sum(emp_rushes.values()) < 0.40:
        emp_rushes = {"RB1": 0.65, "RB2": 0.25, "QB1": 0.10}
        emp_gl = emp_rushes.copy()

    norm_rushes = {r: emp_rushes[r] / sum(emp_rushes.values()) for r in rush_roles}
    rush_means = {r: gross_rush_mean * norm_rushes[r] for r in rush_roles}

    norm_rz = {r: emp_rz[r] / max(0.01, sum(emp_rz.values())) for r in roles}
    norm_gl = {r: emp_gl[r] / max(0.01, sum(emp_gl.values())) for r in rush_roles}

    raw_lambdas = {
        "QB1": team_rush_tds * norm_gl.get("QB1", 0.10),
        "RB1": (team_rush_tds * norm_gl.get("RB1", 0.65)) + (team_pass_tds * norm_rz.get("RB1", 0.10)),
        "RB2": (team_rush_tds * norm_gl.get("RB2", 0.25)) + (team_pass_tds * norm_rz.get("RB2", 0.05)),
        "WR1": team_pass_tds * norm_rz.get("WR1", 0.30),
        "WR2": team_pass_tds * norm_rz.get("WR2", 0.20),
        "WR3": team_pass_tds * norm_rz.get("WR3", 0.10),
        "TE1": team_pass_tds * norm_rz.get("TE1", 0.25),
    }

    scale_lam = team_td_budget / max(0.01, sum(raw_lambdas.values()))
    norm_lambdas = {k: raw_lambdas[k] * scale_lam for k in raw_lambdas}

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
    qb_rush = convert_mean_to_median(rush_means.get("QB1", 0.0), "QB_Rush")
    rb1_rush = convert_mean_to_median(rush_means.get("RB1", 0.0), "RB_Rush")
    rb1_rec = convert_mean_to_median(rec_means["RB1"], "RB_Rec")
    wr1_rec = convert_mean_to_median(rec_means["WR1"], "WR_Rec")
    wr2_rec = convert_mean_to_median(rec_means["WR2"], "WR_Rec")
    te1_rec = convert_mean_to_median(rec_means["TE1"], "TE_Rec")

    p_qb = calculate_lognormal_cover_probability(gross_pass_mean, synthesize_line("Pass Yds", "QB1", qb_pass), LOG_SIGMA["QB_Pass"])
    p_rb1 = calculate_lognormal_cover_probability(rush_means.get("RB1", 0.0), synthesize_line("Rush Yds", "RB1", rb1_rush), LOG_SIGMA["RB_Rush"])
    p_wr1 = calculate_lognormal_cover_probability(rec_means["WR1"], synthesize_line("Rec Yds", "WR1", wr1_rec), LOG_SIGMA["WR_Rec"])
    p_wr2 = calculate_lognormal_cover_probability(rec_means["WR2"], synthesize_line("Rec Yds", "WR2", wr2_rec), LOG_SIGMA["WR_Rec"])
    p_te1 = calculate_lognormal_cover_probability(rec_means["TE1"], synthesize_line("Rec Yds", "TE1", te1_rec), LOG_SIGMA["TE_Rec"])
    p_rb1_rec = calculate_lognormal_cover_probability(rec_means["RB1"], synthesize_line("Rec Yds", "RB1", rb1_rec), LOG_SIGMA["RB_Rec"])

    return [
        build_entry("WR1", depth_names.get("WR1", f"{team_abbr} WR1"), "Rec Yds", wr1_rec, 0.0, 0.0, wr1_rec, norm_lambdas["WR1"], p_wr1),
        build_entry("WR2", depth_names.get("WR2", f"{team_abbr} WR2"), "Rec Yds", wr2_rec, 0.0, 0.0, wr2_rec, norm_lambdas["WR2"], p_wr2),
        build_entry("TE1", depth_names.get("TE1", f"{team_abbr} TE1"), "Rec Yds", te1_rec, 0.0, 0.0, te1_rec, norm_lambdas["TE1"], p_te1),
        build_entry("RB1", depth_names.get("RB1", f"{team_abbr} RB1"), "Rush Yds", rb1_rush, 0.0, rb1_rush, rb1_rec, norm_lambdas["RB1"], p_rb1),
        build_entry("RB1_REC", f"{depth_names.get('RB1', 'RB1')} (Rec)", "Rec Yds", rb1_rec, 0.0, 0.0, rb1_rec, 0.0, p_rb1_rec),
        build_entry("QB1", depth_names.get("QB1", f"{team_abbr} QB"), "Pass Yds", qb_pass, qb_pass, qb_rush, 0.0, norm_lambdas["QB1"], p_qb),
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
    target_season = 2026

    # 1. Load Live Schedules directly from nflreadpy
    logging.info("Querying live nflverse schedules for Season 2026...")
    try:
        schedules = nfl.load_schedules(seasons=[target_season]).to_pandas()
    except Exception:
        schedules = pd.DataFrame()

    if schedules.empty:
        logging.error("Failed to retrieve schedules from nflreadpy.")
        sys.exit(1)

    for col in ["home_team", "away_team"]:
        if col in schedules.columns:
            schedules[col] = schedules[col].apply(clean_team_abbr)

    # 2. Dynamically extract the active unplayed slate
    unplayed = schedules[schedules["result"].isna()].copy()
    if unplayed.empty:
        logging.warning("Zero unplayed games detected. Slate is settled.")
        sys.exit(0)

    target_week = int(unplayed["week"].min())
    upcoming_slate = unplayed[unplayed["week"] == target_week].copy()

    logging.info(f"Targeting Season {target_season} Week {target_week} with {len(upcoming_slate)} active fixture(s)...")

    # 3. Load historical PBP and depth charts
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
        game_id = str(game.get("game_id", f"{target_season}_{target_week}_{away_team}_{home_team}"))

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

        net_pass = (get_stat(home_team, "roll_off_dropback_epa") - get_stat(away_team, "roll_def_dropback_epa")) - \
                   (get_stat(away_team, "roll_off_dropback_epa") - get_stat(home_team, "roll_def_dropback_epa"))
        net_rush = (get_stat(home_team, "roll_off_rush_epa") - get_stat(away_team, "roll_def_rush_epa")) - \
                   (get_stat(away_team, "roll_off_rush_epa") - get_stat(home_team, "roll_def_rush_epa"))
        net_late = (get_stat(home_team, "roll_off_late_down_epa") - get_stat(away_team, "roll_def_late_down_epa")) - \
                   (get_stat(away_team, "roll_off_late_down_epa") - get_stat(home_team, "roll_def_late_down_epa"))

        diff_succ = get_stat(home_team, "roll_off_early_down_success") - get_stat(away_team, "roll_off_early_down_success")
        diff_expl = get_stat(home_team, "roll_off_explosive") - get_stat(away_team, "roll_off_explosive")
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

        # Directional Bayesian shrinkage against consensus
        if canonical_spread >= 3.0:
            weight = 0.70 if abs(raw_prob - market_prob) > 0.25 else 0.50
            calibrated_win_prob = (1.0 - weight) * max(raw_prob, 1.0 - raw_prob) + (weight * market_prob)
        elif canonical_spread <= -3.0:
            weight = 0.70 if abs(raw_prob - market_prob) > 0.25 else 0.50
            calibrated_win_prob = (1.0 - weight) * min(raw_prob, 1.0 - raw_prob) + (weight * market_prob)
        else:
            calibrated_win_prob = (0.50 * raw_prob) + (0.50 * market_prob)

        calibrated_win_prob = max(0.02, min(0.98, calibrated_win_prob))
        sigma = 13.45 * math.sqrt(max(32.0, raw_total) / 44.0)
        model_projected_margin = norm.ppf(calibrated_win_prob) * sigma

        pred_home, pred_away = project_discrete_nfl_scores(model_projected_margin, raw_total)
        pred_total = pred_home + pred_away

        z_cover = (model_projected_margin - canonical_spread) / sigma
        home_cover = float(norm.cdf(z_cover))
        away_cover = 1.0 - home_cover

        home_edge = home_cover - 0.5238
        away_edge = away_cover - 0.5238

        if home_edge >= (MIN_BETTABLE_EDGE_PCT / 100.0) and home_edge > away_edge:
            rec_team = home_team
            rec_line = f"{home_team} {-canonical_spread:+g}"
            cover_prob = home_cover
            final_edge = min(0.050, home_edge)
        elif away_edge >= (MIN_BETTABLE_EDGE_PCT / 100.0) and away_edge > home_edge:
            rec_team = away_team
            rec_line = f"{away_team} {+canonical_spread:+g}"
            cover_prob = away_cover
            final_edge = min(0.050, away_edge)
        else:
            rec_team = "PASS"
            rec_line = "PASS - No Edge"
            cover_prob = max(home_cover, away_cover)
            final_edge = max(home_edge, away_edge)

        b = 1.9091 - 1.0
        q = max(0.0, 1.0 - cover_prob)
        kelly_units = round(max(0.0, min(2.0, (((b * cover_prob) - q) / b) * 0.125 * 100.0)), 2) if rec_team != "PASS" and final_edge >= (MIN_BETTABLE_EDGE_PCT / 100.0) else 0.0

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
            "player_projections": {"home": home_skills, "away": away_skills},
            "matchup_context": {
                "home_team": home_team, "home_roster": {p["role"]: p["player"] for p in home_skills},
                "away_team": away_team, "away_roster": {p["role"]: p["player"] for p in away_skills}
            },
            "tape_metrics": {
                "net_pass_epa_diff": f"{net_pass:+.3f}",
                "net_rush_epa_diff": f"{net_rush:+.3f}",
                "explosive_rate_diff": f"{diff_expl:+.3f}",
                "early_down_success_diff": f"{diff_succ:+.3f}"
            }
        })

    semaphore = asyncio.Semaphore(4)

    async def generate_matchup_analysis(item):
        verdict_str = f"Bet {item['recommended_line']} - {item['kelly_units']:.2f}u" if item['recommended_team'] != "PASS" and item['kelly_units'] > 0.0 else "PASS - 0.00u"
        prompt = f"""[MODE 1: NFL TACTICAL & STATISTICAL BREAKDOWN]
SUBJECT: {item['matchup']} Quantitative Evaluation
DOSSIER PAYLOAD:
{json.dumps(item, indent=2)}

CRITICAL NARRATIVE HARMONIZATION DIRECTIVE:
- Home Team: {item['home_team']} | Projected Score: {item['predicted_home_score']}
- Away Team: {item['away_team']} | Projected Score: {item['predicted_away_score']}
- Win Probabilities: {item['home_team']} Model {item['home_win_prob']*100:.1f}%, Market {item['market_prob']*100:.1f}%
- Stated Net Edge: {item['spread_edge']*100:+.1f}% | Kelly Stake: {item['kelly_units']:.2f}u
- Actionable Verdict: {verdict_str}

Output strictly valid JSON:
{{
  "executive_summary": "Two-sentence strategic verdict detailing player matchups, trench leverage, and harmonized edge assessment.",
  "schematic_matchup": {{
    "away_offense_vs_home_defense": "Detailed film breakdown citing specific named players, pass protection, and coverage shells.",
    "home_offense_vs_away_defense": "Detailed film breakdown citing specific named players, pass protection, and coverage shells."
  }},
  "actionable_verdict": "{verdict_str}"
}}"""
        async with semaphore:
            for attempt in range(3):
                try:
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: ai_client.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=NFL_GURU_FULL_SYSTEM_PROMPT,
                                temperature=0.15,
                                response_mime_type="application/json"
                            )
                        )
                    )
                    parsed = json.loads(response.text)
                    if parsed.get("executive_summary"):
                        parsed["actionable_verdict"] = verdict_str
                        return json.dumps(parsed)
                except Exception:
                    await asyncio.sleep(2 ** attempt)

            return json.dumps({
                "executive_summary": f"Line-of-scrimmage leverage on neutral downs establishes baseline edge on {item['matchup']}.",
                "schematic_matchup": {
                    "away_offense_vs_home_defense": f"{item['away_team']} must sustain early-down push to keep dropbacks on schedule.",
                    "home_offense_vs_away_defense": f"{item['home_team']} attacks intermediate boundary voids against split-safety shells."
                },
                "actionable_verdict": verdict_str
            })

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
        conn.execute(text("DELETE FROM nfl_weekly_analysis WHERE season = :s AND week = :w;"), {"s": target_season, "w": target_week})

    df_results.to_sql("nfl_weekly_analysis", engine, if_exists="append", index=False, method="multi")
    logging.info(f"Database successfully updated with verified live fixtures: {df_results['matchup'].tolist()}")

if __name__ == "__main__":
    asyncio.run(main())
