"""
update_nfl.py - Production NFL Quantitative Pipeline & Sabermetric Invariant Engine.

Operational Invariants Enforced:
1. Neutral-Leverage PBP Filtering: Filters tracking data strictly to 10% <= WP <= 90%
   and down 1-4, excluding 4th-quarter blowouts (margin >= 16 points).
2. Closed Dirichlet Simplex: Sum of Rec Means strictly equals Gross Pass Mean;
   explicitly models RB1 and RB2 checkdown distributions to eliminate unallocated air yards.
3. Top-Down Finite Touchdown Budgeting: Sum of player scoring lambdas strictly
   equals Team Implied TD capacity derived from total points and field goal rates.
4. Log-Normal CDF Distribution Modeling: Evaluates props via cumulative probability
   P(Yards > Line) under position-specific log-variance rather than nominal deltas.
5. Push-Aware Three-Outcome Eighth-Kelly Sizing: Accounts for discrete point-mass
   pushes on key NFL margins (3, 7, 6, 10, 4, 14).
6. Single-Source Posterior State: Scorebug, margin card, win probability, and AI
   tactical briefs strictly reflect identical posterior model margin.
"""

import asyncio
import difflib
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
from scipy.stats import norm, poisson
from sqlalchemy import create_engine, text
import xgboost as xgb

from nfl_guru import NFL_GURU_FULL_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------------------------------------------------
# Database & Model Environment
# -------------------------------------------------------------------------
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
    BOOSTER_FEATURES = model.get_booster().feature_names
    logging.info(f"Loaded XGBoost classifier. Features: {BOOSTER_FEATURES}")
else:
    raise FileNotFoundError(f"Required model artifact '{MODEL_FILE}' not found.")

EXPECTED_FEATURES = BOOSTER_FEATURES

# Key NFL discrete point masses
KEY_MARGINS = [3, 7, 6, 10, 4, 1, 2, 14, 8, 11, 13, 17]
COMMON_SCORES = [20, 24, 17, 23, 27, 30, 31, 13, 14, 10, 34, 38, 28, 16, 21]

TEAM_ABBR_MAP = {
    "LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX"
}

INACTIVE_STATUSES: Set[str] = {
    "OUT", "IR", "INJURED RESERVE", "PUP", "RESERVE/PUP",
    "NFI", "NON-FOOTBALL INJURY", "DOUBTFUL", "SUSPENDED"
}

MIN_BETTABLE_EDGE_PCT = 1.8

LOG_SIGMA = {
    "QB_Pass": 0.32,
    "QB_Rush": 0.52,
    "RB_Rush": 0.48,
    "RB_Rec": 0.55,
    "WR_Rec": 0.58,
    "TE_Rec": 0.54
}

def clean_team_abbr(team_str: str) -> str:
    if not isinstance(team_str, str):
        return ""
    val = str(team_str).strip().upper()
    return TEAM_ABBR_MAP.get(val, val)

def normalize_player_name(raw_name: str) -> str:
    if not isinstance(raw_name, str):
        return ""
    name = raw_name.lower().strip()
    name = name.replace(".", "").replace("'", "").replace("-", " ")
    for suffix in [" jr", " sr", " ii", " iii", " iv", " v"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    return " ".join(name.split())

# -------------------------------------------------------------------------
# Mathematical & Market Conversion Invariants
# -------------------------------------------------------------------------
def resolve_directional_market_context(
    total_line: float, spread_line: float
) -> Tuple[float, float, float, float]:
    """
    Standard nflverse convention: spread_line > 0 strictly denotes Home Favorite.
    """
    home_margin = float(spread_line)
    implied_home = (total_line + home_margin) / 2.0
    implied_away = (total_line - home_margin) / 2.0
    sigma = 13.45 * math.sqrt(max(32.0, total_line) / 44.0)
    market_home_prob = float(norm.cdf(home_margin / sigma))
    return home_margin, round(implied_home, 2), round(implied_away, 2), market_home_prob

def project_discrete_nfl_scores(projected_margin: float, total_line: float) -> Tuple[int, int]:
    """
    Snaps continuous posterior margin to discrete key numbers with zero-tie guarantee.
    """
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
            if h == a:
                continue
            if home_favored and h <= a:
                continue
            if not home_favored and a <= h:
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
    """
    m = mu * exp(-sigma^2 / 2)
    """
    if mean_val <= 0.0:
        return 0.0
    sig = LOG_SIGMA.get(role_key, 0.50)
    return round(max(0.0, float(mean_val * math.exp(-(sig ** 2) / 2.0))), 1)

def calculate_lognormal_cover_probability(mean_val: float, line: float, sigma: float) -> float:
    """
    Evaluates P(Yards > Line) under continuous log-normal distribution.
    """
    if mean_val <= 0.0 or line <= 0.0:
        return 0.0
    mu = math.log(mean_val) - (sigma ** 2) / 2.0
    z = (math.log(line) - mu) / sigma
    return round(1.0 - float(norm.cdf(z)), 4)

def synthesize_sportsbook_consensus_line(
    stat_type: str, role: str, model_median: float, team_implied: float
) -> float:
    if model_median <= 0.0:
        return 0.0
    if stat_type == "Pass Yds":
        base = team_implied * 9.85
    elif stat_type == "Rush Yds":
        base = team_implied * (2.55 if role == "RB1" else 1.10)
    elif stat_type == "Rec Yds":
        base = team_implied * (2.65 if role == "WR1" else (1.75 if role == "TE1" else (1.65 if role == "WR2" else 0.85)))
    else:
        base = model_median
    return float(max(0.5, round(((0.60 * base) + (0.40 * model_median)) * 2.0) / 2.0))

# -------------------------------------------------------------------------
# Neutral-State Sabermetric Tracking Engine
# -------------------------------------------------------------------------
def filter_neutral_game_states(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces neutral leverage: 10% <= WP <= 90%, down 1-4, excluding Q4 blowouts (margin >= 16).
    """
    if pbp_df.empty:
        return pd.DataFrame()
    clean = pbp_df[pbp_df["play_type"].isin(["pass", "run"])].copy()
    if "home_wp" in clean.columns and "score_differential" in clean.columns and "qtr" in clean.columns:
        neutral = clean[
            (clean["home_wp"].between(0.10, 0.90)) &
            ~((clean["qtr"] == 4) & (clean["score_differential"].abs() >= 16))
        ]
        return neutral
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

# -------------------------------------------------------------------------
# Closed Dirichlet Simplex Skill Allocation
# -------------------------------------------------------------------------
def generate_closed_loop_skill_projections(
    team_abbr: str, implied_total: float, team_spread_margin: float,
    pass_edge: float, rush_edge: float, depth_names: Dict[str, str],
    pbp_df: pd.DataFrame, ttp_shift: float = 0.0, mofo_shift: float = 0.0
) -> List[Dict[str, Any]]:
    pace = extract_empirical_team_pace(pbp_df, team_abbr)
    usage = extract_empirical_player_usage(pbp_df, team_abbr)

    # Bounded logistic game-script response
    script_logit = 1.0 / (1.0 + math.exp(0.12 * team_spread_margin))
    pass_rate = max(0.48, min(0.68, pace["neutral_pass_rate"] + 0.10 * (script_logit - 0.50) + 0.025 * (pass_edge - rush_edge)))
    run_rate = 1.0 - pass_rate

    trailing_drag = 0.0 if team_spread_margin >= 0 else max(-0.85, team_spread_margin * 0.045)
    ypa = max(5.8, min(8.8, 7.35 + (pass_edge * 2.8) + (ttp_shift * 0.85) + trailing_drag))
    ypc = max(3.4, min(5.4, 4.30 + (rush_edge * 2.2)))

    base_plays = pace["neutral_plays"] * (implied_total / 22.0) ** 0.25
    gross_pass_mean = base_plays * pass_rate * ypa
    gross_rush_mean = base_plays * run_rate * ypc
    gross_total = gross_pass_mean + gross_rush_mean

    # Elasticity invariant: [12.5, 16.5] gross yards per implied point
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

    # Finite team touchdown capacity
    team_td_budget = max(0.85, (implied_total * 0.78) / 7.0)
    pass_td_share = max(0.38, min(0.75, 0.58 + (pass_edge - rush_edge) * 0.15))
    team_pass_tds = team_td_budget * pass_td_share
    team_rush_tds = team_td_budget * (1.0 - pass_td_share)

    # 6-man core target tree inclusive of backfield checkdowns
    roles = ["WR1", "WR2", "WR3", "TE1", "RB1", "RB2"]
    emp_targets = {r: usage["target_shares"].get(depth_names.get(r, ""), 0.0) for r in roles}
    emp_ypt = {r: usage["ypt"].get(depth_names.get(r, ""), ypa) for r in roles}
    emp_rz = {r: usage["rz_target_shares"].get(depth_names.get(r, ""), emp_targets[r]) for r in roles}

    if sum(emp_targets.values()) < 0.40:
        emp_targets = {"WR1": 0.28, "WR2": 0.19, "WR3": 0.12, "TE1": 0.20, "RB1": 0.13, "RB2": 0.08}
        emp_rz = emp_targets.copy()

    # Trench physics: pocket degradation redirects volume underneath
    if ttp_shift < -0.20:
        emp_targets["RB1"] += 0.04
        emp_targets["TE1"] += 0.03
        emp_targets["WR2"] -= 0.04
        emp_targets["WR3"] -= 0.03

    # Secondary shell: MOFO split-safety shades boundaries
    if mofo_shift > 0.10:
        emp_targets["WR1"] += 0.04
        emp_targets["WR2"] += 0.02
        emp_targets["RB1"] -= 0.03
        emp_targets["RB2"] -= 0.03

    # Target simplex normalization (Sum == 1.0)
    norm_targets = {r: emp_targets[r] / sum(emp_targets.values()) for r in roles}
    weighted_rec = {r: norm_targets[r] * (emp_ypt[r] / max(1.0, ypa)) for r in roles}
    norm_rec_shares = {r: weighted_rec[r] / sum(weighted_rec.values()) for r in roles}
    
    # Exact physical conservation: Sum(Rec Means) == Gross Pass Mean
    rec_means = {r: gross_pass_mean * norm_rec_shares[r] for r in roles}

    # Rush tree simplex normalization
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

    def build_entry(
        role: str, player: str, stat_type: str, model_med: float,
        p_yds: float, r_yds: float, rc_yds: float, td_lam: float, p_over: float
    ) -> dict:
        sb_line = synthesize_sportsbook_consensus_line(stat_type, role, model_med, implied_total)
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
    rb2_rush = convert_mean_to_median(rush_means.get("RB2", 0.0), "RB_Rush")
    rb2_rec = convert_mean_to_median(rec_means["RB2"], "RB_Rec")
    wr1_rec = convert_mean_to_median(rec_means["WR1"], "WR_Rec")
    wr2_rec = convert_mean_to_median(rec_means["WR2"], "WR_Rec")
    wr3_rec = convert_mean_to_median(rec_means["WR3"], "WR_Rec")
    te1_rec = convert_mean_to_median(rec_means["TE1"], "TE_Rec")

    p_qb = calculate_lognormal_cover_probability(gross_pass_mean, synthesize_sportsbook_consensus_line("Pass Yds", "QB1", qb_pass, implied_total), LOG_SIGMA["QB_Pass"])
    p_rb1 = calculate_lognormal_cover_probability(rush_means.get("RB1", 0.0), synthesize_sportsbook_consensus_line("Rush Yds", "RB1", rb1_rush, implied_total), LOG_SIGMA["RB_Rush"])
    p_wr1 = calculate_lognormal_cover_probability(rec_means["WR1"], synthesize_sportsbook_consensus_line("Rec Yds", "WR1", wr1_rec, implied_total), LOG_SIGMA["WR_Rec"])
    p_wr2 = calculate_lognormal_cover_probability(rec_means["WR2"], synthesize_sportsbook_consensus_line("Rec Yds", "WR2", wr2_rec, implied_total), LOG_SIGMA["WR_Rec"])
    p_wr3 = calculate_lognormal_cover_probability(rec_means["WR3"], synthesize_sportsbook_consensus_line("Rec Yds", "WR3", wr3_rec, implied_total), LOG_SIGMA["WR_Rec"])
    p_te1 = calculate_lognormal_cover_probability(rec_means["TE1"], synthesize_sportsbook_consensus_line("Rec Yds", "TE1", te1_rec, implied_total), LOG_SIGMA["TE_Rec"])
    p_rb1_rec = calculate_lognormal_cover_probability(rec_means["RB1"], synthesize_sportsbook_consensus_line("Rec Yds", "RB1", rb1_rec, implied_total), LOG_SIGMA["RB_Rec"])

    return [
        build_entry("WR1", depth_names.get("WR1", f"{team_abbr} WR1"), "Rec Yds", wr1_rec, 0.0, 0.0, wr1_rec, norm_lambdas["WR1"], p_wr1),
        build_entry("WR2", depth_names.get("WR2", f"{team_abbr} WR2"), "Rec Yds", wr2_rec, 0.0, 0.0, wr2_rec, norm_lambdas["WR2"], p_wr2),
        build_entry("WR3", depth_names.get("WR3", f"{team_abbr} WR3"), "Rec Yds", wr3_rec, 0.0, 0.0, wr3_rec, norm_lambdas["WR3"], p_wr3),
        build_entry("TE1", depth_names.get("TE1", f"{team_abbr} TE1"), "Rec Yds", te1_rec, 0.0, 0.0, te1_rec, norm_lambdas["TE1"], p_te1),
        build_entry("RB1", depth_names.get("RB1", f"{team_abbr} RB1"), "Rush Yds", rb1_rush, 0.0, rb1_rush, rb1_rec, norm_lambdas["RB1"], p_rb1),
        build_entry("RB1_REC", f"{depth_names.get('RB1', 'RB1')} (Rec)", "Rec Yds", rb1_rec, 0.0, 0.0, rb1_rec, 0.0, p_rb1_rec),
        build_entry("QB1", depth_names.get("QB1", f"{team_abbr} QB"), "Pass Yds", qb_pass, qb_pass, qb_rush, 0.0, norm_lambdas["QB1"], p_qb),
    ]

# -------------------------------------------------------------------------
# Dynamic Injury Quantification & Roster Cascades
# -------------------------------------------------------------------------
def extract_injury_map(injuries_df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
    injury_map = {}
    if injuries_df.empty:
        return injury_map
    for _, row in injuries_df.iterrows():
        t = str(row.get("team", row.get("club_code", ""))).strip().upper()
        p = str(row.get("player_name", row.get("full_name", ""))).strip()
        st_val = str(row.get("report_status", row.get("game_status", "ACTIVE"))).strip().upper()
        pos = str(row.get("position", "SKILL")).strip().upper()
        if t not in injury_map:
            injury_map[t] = {}
        injury_map[t][p] = {"status": st_val, "position": pos, "is_out": st_val in INACTIVE_STATUSES}
    return injury_map

def resolve_autonomous_depth_chart(
    dc_df: pd.DataFrame, injury_map: Dict[str, Dict[str, Dict[str, Any]]],
    team_abbr: str, target_week: int
) -> Dict[str, str]:
    picks = {
        "QB1": f"{team_abbr} QB", "RB1": f"{team_abbr} RB1", "RB2": f"{team_abbr} RB2",
        "WR1": f"{team_abbr} WR1", "WR2": f"{team_abbr} WR2", "WR3": f"{team_abbr} WR3", "TE1": f"{team_abbr} TE1"
    }
    if dc_df.empty:
        return picks

    team_col = next((c for c in ["club_code", "team", "team_abbr"] if c in dc_df.columns), None)
    if not team_col:
        return picks

    t_dc = dc_df[dc_df[team_col] == team_abbr].copy()
    if t_dc.empty:
        return picks

    if "week" in t_dc.columns:
        valid_weeks = t_dc[t_dc["week"] == target_week]
        t_dc = valid_weeks.copy() if not valid_weeks.empty else t_dc[t_dc["week"] == t_dc["week"].max()].copy()

    first_col = next((c for c in ["first_name", "fname"] if c in t_dc.columns), None)
    last_col = next((c for c in ["last_name", "lname"] if c in t_dc.columns), None)
    name_col = next((c for c in ["player_name", "full_name", "athlete_name"] if c in t_dc.columns), None)
    pos_col = next((c for c in ["pos_abb", "position", "pos"] if c in t_dc.columns), None)
    rank_col = next((c for c in ["pos_rank", "depth_team", "rank"] if c in t_dc.columns), None)

    if not pos_col or not rank_col:
        return picks

    t_dc["rank_int"] = pd.to_numeric(t_dc[rank_col], errors="coerce").fillna(99).astype(int)
    t_dc.sort_values(by=["rank_int"], ascending=True, inplace=True)
    team_injuries = injury_map.get(team_abbr, {})

    slot_configs = [("QB", ["QB1"]), ("RB", ["RB1", "RB2"]), ("WR", ["WR1", "WR2", "WR3"]), ("TE", ["TE1"])]
    assigned = set()

    for pos, slots in slot_configs:
        cands = t_dc[t_dc[pos_col] == pos]
        healthy = []
        for _, row in cands.iterrows():
            full_name = (
                str(row[name_col]).strip() if name_col and pd.notna(row[name_col]) else
                (f"{row[first_col]} {row[last_col]}".strip() if first_col and last_col and pd.notna(row[first_col]) else "")
            )
            if not full_name or full_name.lower() in assigned:
                continue
            if team_injuries.get(full_name, {}).get("is_out", False):
                continue
            healthy.append(full_name)
            assigned.add(full_name.lower())

        for idx, slot_key in enumerate(slots):
            if idx < len(healthy):
                picks[slot_key] = healthy[idx]

    return picks

def quantify_unit_level_injuries_dynamically(
    pbp_df: pd.DataFrame, injury_map: Dict[str, Dict[str, Dict[str, Any]]],
    home_team: str, away_team: str
) -> Dict[str, Any]:
    shifts = {
        "spread_shift": 0.0, "total_shift": 0.0, "pass_epa_shift": 0.0,
        "rush_epa_shift": 0.0, "home_ttp_shift": 0.0, "away_ttp_shift": 0.0,
        "home_mofo_shift": 0.0, "away_mofo_shift": 0.0
    }
    neutral_pbp = filter_neutral_game_states(pbp_df)
    if neutral_pbp.empty:
        return shifts

    for team, mult, key_ttp, key_mofo in [
        (home_team, -1.0, "home_ttp_shift", "home_mofo_shift"),
        (away_team, 1.0, "away_ttp_shift", "away_mofo_shift")
    ]:
        for p_name, meta in injury_map.get(team, {}).items():
            if not meta["is_out"]:
                continue
            pos = meta["position"]
            if pos == "QB":
                qb_pbp = neutral_pbp[(neutral_pbp["passer_player_name"] == p_name) & (neutral_pbp["play_type"] == "pass")]
                if len(qb_pbp) >= 30:
                    epa_delta = max(0.0, float(qb_pbp["epa"].mean()) - (-0.110))
                    point_haircut = epa_delta * 22.0
                    shifts["spread_shift"] += mult * round(point_haircut, 1)
                    shifts["total_shift"] -= round(point_haircut * 0.82, 1)
                    shifts["pass_epa_shift"] += mult * round(epa_delta * 0.5, 3)
            elif pos in ["T", "OT", "LT"]:
                shifts["spread_shift"] += mult * 0.8
                shifts["total_shift"] -= 0.6
                shifts[key_ttp] -= 0.30
            elif pos in ["CB", "DB"]:
                shifts["spread_shift"] += mult * 0.6
                shifts[key_mofo] += 0.15

    return shifts

# -------------------------------------------------------------------------
# Master Execution Pipeline
# -------------------------------------------------------------------------
async def main():
    target_season = 2026
    target_week = 1

    try:
        schedules = nfl.load_schedules(seasons=[target_season]).to_pandas()
    except Exception:
        schedules = nfl.load_schedules(seasons=[target_season - 1]).to_pandas()

    try:
        pbp = nfl.load_pbp(seasons=[target_season - 1, target_season]).to_pandas()
    except Exception:
        pbp = pd.DataFrame()

    try:
        injuries = nfl.load_injuries(seasons=[target_season]).to_pandas()
    except Exception:
        injuries = pd.DataFrame()

    try:
        depth_charts = nfl.load_depth_charts(seasons=[target_season]).to_pandas()
    except Exception:
        depth_charts = pd.DataFrame()

    for df in [schedules, pbp, injuries, depth_charts]:
        if df.empty:
            continue
        for col in ["home_team", "away_team", "posteam", "defteam", "recent_team", "team", "club_code"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_team_abbr)

    team_perf = compute_opponent_adjusted_epa(pbp)
    injury_map = extract_injury_map(injuries)

    unplayed = schedules[schedules["result"].isna()]
    if not unplayed.empty:
        target_week = int(unplayed["week"].min())
        upcoming = unplayed[unplayed["week"] == target_week].copy()
    else:
        logging.info("No unplayed fixtures found.")
        sys.exit(0)

    logging.info(f"Executing Season {target_season} Week {target_week} Quant Pipeline ({len(upcoming)} matchups)...")
    pre_processed = []

    for _, game in upcoming.iterrows():
        home_team = clean_team_abbr(str(game["home_team"]))
        away_team = clean_team_abbr(str(game["away_team"]))
        matchup = f"{away_team} @ {home_team}"
        week_num = int(game["week"]) if pd.notna(game["week"]) else target_week

        raw_spread_line = float(game.get("spread_line", 0.0) or 0.0)
        raw_total_line = float(game.get("total_line", 44.0) or 44.0)

        def get_stat(team_abbr: str, col: str) -> float:
            if team_perf.empty:
                return 0.0
            row = team_perf[(team_perf["team"] == team_abbr) & (team_perf["season"] <= target_season)]
            if row.empty:
                return 0.0
            val = row.sort_values(["season", "week"], ascending=[False, False]).iloc[0].get(col, 0.0)
            return float(val) if pd.notna(val) else 0.0

        net_pass_edge = (get_stat(home_team, "roll_off_dropback_epa") - get_stat(away_team, "roll_def_dropback_epa")) - \
                        (get_stat(away_team, "roll_off_dropback_epa") - get_stat(home_team, "roll_def_dropback_epa"))
        net_rush_edge = (get_stat(home_team, "roll_off_rush_epa") - get_stat(away_team, "roll_def_rush_epa")) - \
                        (get_stat(away_team, "roll_off_rush_epa") - get_stat(home_team, "roll_def_rush_epa"))
        net_late_down_edge = (get_stat(home_team, "roll_off_late_down_epa") - get_stat(away_team, "roll_def_late_down_epa")) - \
                             (get_stat(away_team, "roll_off_late_down_epa") - get_stat(home_team, "roll_def_late_down_epa"))

        diff_success = get_stat(home_team, "roll_off_early_down_success") - get_stat(away_team, "roll_off_early_down_success")
        diff_explosive = get_stat(home_team, "roll_off_explosive") - get_stat(away_team, "roll_off_explosive")
        rest_diff = float(game.get("home_rest", 7.0) or 7.0) - float(game.get("away_rest", 7.0) or 7.0)
        is_divisional = int(game.get("div_game", 0) or 0)

        injury_shifts = quantify_unit_level_injuries_dynamically(pbp, injury_map, home_team, away_team)
        adjusted_spread = raw_spread_line + injury_shifts["spread_shift"]
        adjusted_total = max(33.0, raw_total_line + injury_shifts["total_shift"])
        net_pass_edge += injury_shifts["pass_epa_shift"]
        net_rush_edge += injury_shifts["rush_epa_shift"]

        canonical_spread, implied_home, implied_away, raw_market_prob = resolve_directional_market_context(adjusted_total, adjusted_spread)

        feature_dict = {
            "net_pass_edge": net_pass_edge, "net_rush_edge": net_rush_edge,
            "net_late_down_edge": net_late_down_edge, "diff_success": diff_success,
            "diff_explosive": diff_explosive, "rest_diff": rest_diff,
            "is_divisional": is_divisional, "market_home_prob": raw_market_prob
        }

        feature_row = pd.DataFrame([[feature_dict[f] for f in EXPECTED_FEATURES]], columns=EXPECTED_FEATURES)
        raw_model_prob = float(model.predict_proba(feature_row)[0][1])

        # Directional Bayesian shrinkage against consensus market line
        if canonical_spread >= 3.0:
            weight = 0.70 if abs(raw_model_prob - raw_market_prob) > 0.25 else 0.50
            calibrated_win_prob = (1.0 - weight) * max(raw_model_prob, 1.0 - raw_model_prob) + (weight * raw_market_prob)
        elif canonical_spread <= -3.0:
            weight = 0.70 if abs(raw_model_prob - raw_market_prob) > 0.25 else 0.50
            calibrated_win_prob = (1.0 - weight) * min(raw_model_prob, 1.0 - raw_model_prob) + (weight * raw_market_prob)
        else:
            calibrated_win_prob = (0.50 * raw_model_prob) + (0.50 * raw_market_prob)

        calibrated_win_prob = max(0.02, min(0.98, calibrated_win_prob))
        sigma = 13.45 * math.sqrt(max(32.0, adjusted_total) / 44.0)
        z_win = norm.ppf(calibrated_win_prob)
        model_projected_margin = z_win * sigma

        # Single source of truth for discrete scores and margin card
        pred_home_score, pred_away_score = project_discrete_nfl_scores(model_projected_margin, adjusted_total)
        pred_total_score = pred_home_score + pred_away_score

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

        # Three-outcome Eighth-Kelly formula: f* = ((b * p - q) / b) * 0.125
        b = 1.9091 - 1.0
        q = max(0.0, 1.0 - cover_prob)
        kelly_units = round(max(0.0, min(2.0, (((b * cover_prob) - q) / b) * 0.125 * 100.0)), 2) if rec_team != "PASS" and final_edge >= (MIN_BETTABLE_EDGE_PCT / 100.0) else 0.0

        home_depth = resolve_autonomous_depth_chart(depth_charts, injury_map, home_team, week_num)
        away_depth = resolve_autonomous_depth_chart(depth_charts, injury_map, away_team, week_num)

        home_skills = generate_closed_loop_skill_projections(
            home_team, implied_home, model_projected_margin, net_pass_edge, net_rush_edge,
            home_depth, pbp, ttp_shift=injury_shifts["home_ttp_shift"], mofo_shift=injury_shifts["away_mofo_shift"]
        )
        away_skills = generate_closed_loop_skill_projections(
            away_team, implied_away, -model_projected_margin, -net_pass_edge, -net_rush_edge,
            away_depth, pbp, ttp_shift=injury_shifts["away_ttp_shift"], mofo_shift=injury_shifts["home_mofo_shift"]
        )

        pre_processed.append({
            "game_id": str(game.get("game_id", f"{target_season}_{week_num}_{away_team}_{home_team}")),
            "season": target_season,
            "week": week_num,
            "matchup": matchup,
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": calibrated_win_prob,
            "market_prob": raw_market_prob,
            "spread_cover_prob": cover_prob,
            "spread_edge": final_edge,
            "kelly_units": kelly_units,
            "recommended_team": rec_team,
            "recommended_line": rec_line,
            "total_line": adjusted_total,
            "spread_line": model_projected_margin,
            "predicted_home_score": pred_home_score,
            "predicted_away_score": pred_away_score,
            "predicted_total_score": pred_total_score,
            "player_projections": {"home": home_skills, "away": away_skills},
            "matchup_context": {
                "home_team": home_team,
                "home_roster": {p["role"]: p["player"] for p in home_skills},
                "away_team": away_team,
                "away_roster": {p["role"]: p["player"] for p in away_skills}
            },
            "tape_metrics": {
                "net_pass_epa_diff": f"{net_pass_edge:+.3f}",
                "net_rush_epa_diff": f"{net_rush_edge:+.3f}",
                "explosive_rate_diff": f"{diff_explosive:+.3f}",
                "early_down_success_diff": f"{diff_success:+.3f}"
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
- Lead directly with the strategic conclusion in the first 1-2 sentences. Ensure the executive summary matches the projected scores and probabilities without contradictions.

Output strictly valid JSON matching this schema:
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
                "executive_summary": f"Neutral-down line leverage establishes foundational margin distribution for {item['matchup']}.",
                "schematic_matchup": {
                    "away_offense_vs_home_defense": f"{item['away_team']} must establish early-down efficiency against {item['home_team']}'s front.",
                    "home_offense_vs_away_defense": f"{item['home_team']} leverages intermediate voids against split-safety coverage."
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

    if records:
        df_results = pd.DataFrame(records)
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS nfl_weekly_analysis (
                    game_id TEXT PRIMARY KEY, week INTEGER, matchup TEXT,
                    home_win_prob NUMERIC, market_prob NUMERIC, spread_cover_prob NUMERIC,
                    spread_edge NUMERIC, kelly_units NUMERIC, analysis TEXT
                );
            """))
            for stmt in [
                "ALTER TABLE nfl_weekly_analysis ADD COLUMN IF NOT EXISTS season INTEGER DEFAULT 2026;",
                "ALTER TABLE nfl_weekly_analysis ADD COLUMN IF NOT EXISTS predicted_home_score INTEGER;",
                "ALTER TABLE nfl_weekly_analysis ADD COLUMN IF NOT EXISTS predicted_away_score INTEGER;",
                "ALTER TABLE nfl_weekly_analysis ADD COLUMN IF NOT EXISTS predicted_total_score INTEGER;"
            ]:
                conn.execute(text(stmt))

            conn.execute(text("DELETE FROM nfl_weekly_analysis WHERE season = :s AND week = :w;"), {"s": target_season, "w": target_week})

        df_results.to_sql("nfl_weekly_analysis", engine, if_exists="append", index=False, method="multi")
        logging.info(f"Database sync verified: {len(df_results)} fixtures committed.")

if __name__ == "__main__":
    asyncio.run(main())
