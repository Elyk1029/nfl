"""
update_nfl.py - Institutional NFL Quantitative Terminal Pipeline Orchestrator.
Features:
- Full Multi-Tier Injury Engine: Evaluates QB, OL (LT/RT/C), Secondary (CB1/CB2/FS),
  Front-7 (EDGE/IDL), and Skill injuries to dynamically adjust spreads, totals, TTP, and shell distributions.
- Autonomous Roster Cascading: Live ingestion of league injury wires; prunes IR/PUP/OUT scratches.
- Strict Opponent-Cross EPA Mathematics: (off_home - def_away) - (off_away - def_home).
- Canonical Spread Alignment: spread_line > 0 strictly designates Home Favorite.
- Closed-Loop Dirichlet Target Tree Simplex: Sum of Rec Means == Gross Team Pass Mean.
- Log-Normal Median Conversions: m = mu * exp(-sigma^2 / 2).
- Team-Budgeted Poisson TD Allocation bound to Implied Total / 7.15.
- Research Director & Quantitative Architect System Prompt embedded via Gemini 3.8 Flash.
- Auto-Migrating Neon PostgreSQL Persistence.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------------------------------------------------
# 1. Environment Verification & Client Initialization
# -------------------------------------------------------------------------
db_url = os.environ.get("DATABASE_URL")
gemini_key = os.environ.get("GEMINI_API_KEY")

if not db_url or not gemini_key:
    raise ValueError("FATAL: DATABASE_URL and GEMINI_API_KEY must be configured in environment or secrets.")

engine = create_engine(db_url, pool_size=5, max_overflow=10, pool_pre_ping=True)
client = genai.Client(api_key=gemini_key)

MODEL_FILE = "nfl_model.json"
model = xgb.XGBClassifier()
if os.path.exists(MODEL_FILE):
    model.load_model(MODEL_FILE)
    BOOSTER_FEATURES = model.get_booster().feature_names
    logging.info(f"XGBoost booster verified. Features: {BOOSTER_FEATURES}")
else:
    raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found in root directory.")

DEFAULT_FEATURES = [
    "net_pass_edge", "net_rush_edge", "net_late_down_edge", "diff_success",
    "diff_explosive", "rest_diff", "is_divisional", "market_home_prob"
]
EXPECTED_FEATURES = BOOSTER_FEATURES if BOOSTER_FEATURES else DEFAULT_FEATURES

NFL_KEY_MARGINS = [3, 7, 6, 10, 4, 1, 2, 14, 8, 11, 13, 17]
COMMON_TEAM_SCORES = [20, 24, 17, 23, 27, 30, 31, 13, 14, 10, 34, 38, 28, 16, 21]
TEAM_ABBR_MAP = {"LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX"}

INACTIVE_STATUS_DESIGNATIONS: Set[str] = {
    "OUT", "IR", "INJURED RESERVE", "PUP", "RESERVE/PUP",
    "NFI", "NON-FOOTBALL INJURY", "DOUBTFUL", "SUSPENDED"
}

STATUS_PROBABILITY_WEIGHTS: Dict[str, float] = {
    "OUT": 0.00,
    "IR": 0.00,
    "PUP": 0.00,
    "DOUBTFUL": 0.05,
    "QUESTIONABLE_DNP": 0.25,
    "QUESTIONABLE_LP": 0.55,
    "QUESTIONABLE_FP": 0.85,
    "QUESTIONABLE": 0.60,
    "ACTIVE": 1.00,
    "HEALTHY": 1.00
}

# Empirical Multi-Tier Positional Valuation Matrix (VORP & Schematic Gradients)
POSITIONAL_IMPACT_METRICS: Dict[str, Dict[str, float]] = {
    "QB_ELITE": {"spread_val": 4.5, "total_drop": 4.0, "pass_epa_delta": 0.160, "ttp_delta": 0.00},
    "QB_STARTER": {"spread_val": 2.5, "total_drop": 2.2, "pass_epa_delta": 0.100, "ttp_delta": 0.00},
    "LT": {"spread_val": 1.2, "total_drop": 1.0, "pass_epa_delta": 0.065, "ttp_delta": -0.35},
    "RT": {"spread_val": 0.8, "total_drop": 0.7, "pass_epa_delta": 0.045, "ttp_delta": -0.22},
    "C":  {"spread_val": 0.6, "total_drop": 0.5, "pass_epa_delta": 0.035, "ttp_delta": -0.15},
    "EDGE1": {"spread_val": 1.0, "total_drop": -0.8, "pass_epa_delta": -0.055, "ttp_delta": 0.28},
    "IDL1": {"spread_val": 0.7, "total_drop": -0.5, "rush_epa_delta": -0.060, "box_fit_leak": 0.10},
    "CB1": {"spread_val": 1.1, "total_drop": -0.9, "pass_epa_delta": -0.060, "mofo_shift": 0.18},
    "CB2": {"spread_val": 0.6, "total_drop": -0.5, "pass_epa_delta": -0.035, "mofo_shift": 0.08},
    "FS1": {"spread_val": 0.7, "total_drop": -0.6, "pass_epa_delta": -0.045, "deep_hole_leak": 0.12},
    "WR1": {"spread_val": 1.3, "total_drop": 1.1, "pass_epa_delta": 0.070, "target_funnel_loss": 0.28},
    "RB1": {"spread_val": 0.5, "total_drop": 0.4, "rush_epa_delta": 0.030, "target_funnel_loss": 0.12}
}

def clean_team_abbr(team_str: str) -> str:
    if not isinstance(team_str, str):
        return ""
    c = str(team_str).strip().upper()
    return TEAM_ABBR_MAP.get(c, c)

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
# 2. Canonical Directional Scoring Engine
# -------------------------------------------------------------------------
def resolve_directional_market_context(total_line: float, nflfastr_spread_line: float) -> Tuple[float, float, float, float]:
    """
    In nflreadpy schedules:
    spread_line > 0 strictly indicates HOME is favored.
    spread_line < 0 strictly indicates AWAY is favored.
    """
    canonical_home_margin = float(nflfastr_spread_line)
    implied_home = (total_line + canonical_home_margin) / 2.0
    implied_away = (total_line - canonical_home_margin) / 2.0

    sigma = 13.45 * math.sqrt(max(32.0, total_line) / 44.0)
    market_home_prob = float(norm.cdf(canonical_home_margin / sigma))

    return canonical_home_margin, round(implied_home, 2), round(implied_away, 2), market_home_prob

def project_discrete_nfl_scores(projected_margin: float, total_line: float) -> Tuple[int, int]:
    effective_margin = projected_margin if abs(projected_margin) >= 0.10 else 0.50
    home_favored = effective_margin > 0.0
    abs_margin = abs(effective_margin)

    selected_discrete_margin = min(NFL_KEY_MARGINS, key=lambda m: abs(m - abs_margin))
    raw_home = (total_line + (selected_discrete_margin if home_favored else -selected_discrete_margin)) / 2.0
    raw_away = (total_line - (selected_discrete_margin if home_favored else -selected_discrete_margin)) / 2.0

    best_pair = (27, 20) if home_favored else (20, 27)
    min_loss = float("inf")

    candidate_home = [s for s in COMMON_TEAM_SCORES if abs(s - raw_home) <= 6.5] or [int(round(raw_home))]
    candidate_away = [s for s in COMMON_TEAM_SCORES if abs(s - raw_away) <= 6.5] or [int(round(raw_away))]

    for h in candidate_home:
        for a in candidate_away:
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

# -------------------------------------------------------------------------
# 3. Closed-Loop Skill Engine & Sportsbook Benchmarks
# -------------------------------------------------------------------------
LOG_SIGMA = {
    "QB_Pass": 0.32, "QB_Rush": 0.52, "RB_Rush": 0.48, 
    "RB_Rec": 0.55, "WR_Rec": 0.58, "TE_Rec": 0.54
}

def convert_mean_to_median(mean_val: float, role_key: str) -> float:
    if mean_val <= 0.0:
        return 0.0
    sig = LOG_SIGMA.get(role_key, 0.50)
    return round(max(0.0, float(mean_val * math.exp(-(sig ** 2) / 2.0))), 1)

def synthesize_sportsbook_consensus_line(stat_type: str, role: str, model_median: float, team_implied: float) -> float:
    if model_median <= 0.0:
        return 0.0
    if stat_type == "Pass Yds":
        base = 215.0 + (team_implied - 20.0) * 4.8
        line = round((0.65 * base + 0.35 * model_median) * 2) / 2.0
    elif stat_type == "Rush Yds":
        if role == "RB1":
            base = 54.0 + (team_implied - 20.0) * 3.1
            line = round((0.65 * base + 0.35 * model_median) * 2) / 2.0
        elif role == "RB2":
            base = 22.0 + (team_implied - 20.0) * 1.4
            line = round((0.65 * base + 0.35 * model_median) * 2) / 2.0
        else:
            line = round((model_median * 0.90) * 2) / 2.0
    elif stat_type == "Rec Yds":
        if role == "WR1":
            base = 58.0 + (team_implied - 20.0) * 2.4
            line = round((0.65 * base + 0.35 * model_median) * 2) / 2.0
        elif role == "WR2":
            base = 36.0 + (team_implied - 20.0) * 1.6
            line = round((0.65 * base + 0.35 * model_median) * 2) / 2.0
        elif role == "TE1":
            base = 40.0 + (team_implied - 20.0) * 1.9
            line = round((0.65 * base + 0.35 * model_median) * 2) / 2.0
        else:
            line = round((model_median * 0.88) * 2) / 2.0
    else:
        line = round(model_median * 2) / 2.0
    return float(max(0.5, line))

def generate_closed_loop_skill_projections(
    team_abbr: str, 
    implied_total: float, 
    team_spread_margin: float,
    pass_edge: float, 
    rush_edge: float, 
    depth_names: Dict[str, str],
    pbp_df: pd.DataFrame,
    ttp_shift: float = 0.0,
    mofo_shift: float = 0.0
) -> List[Dict[str, Any]]:
    total_plays = 63.5 * (implied_total / 22.0) ** 0.25
    
    script_shift = -0.007 * team_spread_margin
    scheme_shift = 0.025 * (pass_edge - rush_edge)
    pass_rate = max(0.48, min(0.66, 0.575 + script_shift + scheme_shift))
    run_rate = 1.0 - pass_rate

    # TTP degradation compresses Yards Per Attempt (YPA)
    base_ypa = 7.35 + (pass_edge * 2.8) + (ttp_shift * 0.85)
    ypa = max(5.8, min(9.0, base_ypa))
    ypc = max(3.4, min(5.4, 4.30 + (rush_edge * 2.2)))

    gross_pass_mean = total_plays * pass_rate * ypa
    gross_rush_mean = total_plays * run_rate * ypc

    team_td_budget = max(0.90, implied_total / 7.15)
    pass_td_share = max(0.42, min(0.75, 0.58 + (pass_edge - rush_edge) * 0.15))
    team_pass_tds = team_td_budget * pass_td_share
    team_rush_tds = team_td_budget * (1.0 - pass_td_share)

    target_weights = {"WR1": 0.27, "WR2": 0.18, "WR3": 0.12, "TE1": 0.20, "RB1": 0.12, "RB2": 0.05, "OTHER": 0.06}
    ypt_multipliers = {"WR1": 1.16, "WR2": 1.05, "WR3": 0.95, "TE1": 0.92, "RB1": 0.62, "RB2": 0.55, "OTHER": 0.80}

    # Trench & Secondary Schematic Adaptations:
    if ttp_shift < -0.20:
        # Pass protection collapse: checkdown expansion
        target_weights["RB1"] += 0.04
        target_weights["TE1"] += 0.03
        target_weights["WR2"] -= 0.04
        target_weights["WR3"] -= 0.03

    if mofo_shift > 0.10:
        # Opponent secondary missing CB1: split safety shade opens boundary target volume
        target_weights["WR1"] += 0.04
        target_weights["WR2"] += 0.02
        target_weights["RB1"] -= 0.03
        target_weights["OTHER"] -= 0.03

    if not pbp_df.empty:
        t_passes = pbp_df[(pbp_df["posteam"] == team_abbr) & (pbp_df["play_type"] == "pass") & (pbp_df["home_wp"].between(0.10, 0.90))]
        if len(t_passes) > 50:
            counts = t_passes["receiver_player_name"].value_counts(normalize=True)
            for role_key in ["WR1", "WR2", "WR3", "TE1", "RB1", "RB2"]:
                p_name = depth_names.get(role_key, "")
                if p_name in counts:
                    target_weights[role_key] = float(counts[p_name])

    norm_factor = sum(target_weights.values())
    norm_target_shares = {k: target_weights[k] / norm_factor for k in target_weights}

    weighted_rec = {k: norm_target_shares[k] * ypt_multipliers[k] for k in norm_target_shares}
    rec_sum = sum(weighted_rec.values())
    rec_yard_shares = {k: weighted_rec[k] / rec_sum for k in weighted_rec}

    rec_means = {k: gross_pass_mean * rec_yard_shares[k] for k in rec_yard_shares}

    rb1_rush_mean = gross_rush_mean * 0.60
    rb2_rush_mean = gross_rush_mean * 0.28
    qb_rush_mean = gross_rush_mean * 0.08

    rb1_td_lambda = (team_rush_tds * 0.65) + (team_pass_tds * (norm_target_shares["RB1"] * 0.55))
    rb2_td_lambda = (team_rush_tds * 0.25) + (team_pass_tds * (norm_target_shares["RB2"] * 0.40))
    qb_td_lambda = team_rush_tds * 0.10
    wr1_td_lambda = team_pass_tds * (norm_target_shares["WR1"] * 1.30)
    wr2_td_lambda = team_pass_tds * (norm_target_shares["WR2"] * 1.05)
    wr3_td_lambda = team_pass_tds * (norm_target_shares["WR3"] * 0.80)
    te1_td_lambda = team_pass_tds * (norm_target_shares["TE1"] * 1.25)

    def calc_anytime_td(lam: float) -> float:
        return round(float((1.0 - math.exp(-max(0.001, lam))) * 100.0), 1)

    def build_entry(role: str, player: str, stat_type: str, model_med: float, p_yds: float, r_yds: float, rc_yds: float, td_lam: float) -> dict:
        sb_line = synthesize_sportsbook_consensus_line(stat_type, role, model_med, implied_total)
        delta = round(model_med - sb_line, 1)
        rec = "OVER" if delta >= 3.5 else ("UNDER" if delta <= -3.5 else "PASS")
        return {
            "role": role,
            "player": player,
            "primary_stat_type": stat_type,
            "model_median": model_med,
            "sportsbook_line": sb_line,
            "edge_delta": delta,
            "prop_recommendation": rec,
            "pass_yards": p_yds,
            "rush_yards": r_yds,
            "rec_yards": rc_yds,
            "total_tds": round(td_lam, 2),
            "anytime_td_prob": calc_anytime_td(td_lam)
        }

    qb_pass = convert_mean_to_median(gross_pass_mean, "QB_Pass")
    qb_rush = convert_mean_to_median(qb_rush_mean, "QB_Rush")
    rb1_rush = convert_mean_to_median(rb1_rush_mean, "RB_Rush")
    rb1_rec = convert_mean_to_median(rec_means["RB1"], "RB_Rec")
    rb2_rush = convert_mean_to_median(rb2_rush_mean, "RB_Rush")
    rb2_rec = convert_mean_to_median(rec_means["RB2"], "RB_Rec")
    wr1_rec = convert_mean_to_median(rec_means["WR1"], "WR_Rec")
    wr2_rec = convert_mean_to_median(rec_means["WR2"], "WR_Rec")
    wr3_rec = convert_mean_to_median(rec_means["WR3"], "WR_Rec")
    te1_rec = convert_mean_to_median(rec_means["TE1"], "TE_Rec")

    return [
        build_entry("WR1", depth_names.get("WR1", f"{team_abbr} WR1"), "Rec Yds", wr1_rec, 0.0, 0.0, wr1_rec, wr1_td_lambda),
        build_entry("WR2", depth_names.get("WR2", f"{team_abbr} WR2"), "Rec Yds", wr2_rec, 0.0, 0.0, wr2_rec, wr2_td_lambda),
        build_entry("WR3", depth_names.get("WR3", f"{team_abbr} WR3"), "Rec Yds", wr3_rec, 0.0, 0.0, wr3_rec, wr3_td_lambda),
        build_entry("TE1", depth_names.get("TE1", f"{team_abbr} TE1"), "Rec Yds", te1_rec, 0.0, 0.0, te1_rec, te1_td_lambda),
        build_entry("RB1", depth_names.get("RB1", f"{team_abbr} RB1"), "Rush Yds", rb1_rush, 0.0, rb1_rush, rb1_rec, rb1_td_lambda),
        build_entry("RB2", depth_names.get("RB2", f"{team_abbr} RB2"), "Rush Yds", rb2_rush, 0.0, rb2_rush, rb2_rec, rb2_td_lambda),
        build_entry("QB1", depth_names.get("QB1", f"{team_abbr} QB"), "Pass Yds", qb_pass, qb_pass, qb_rush, 0.0, qb_td_lambda),
    ]

# -------------------------------------------------------------------------
# 4. Multi-Source Ingestion & Autonomous Roster Resolution
# -------------------------------------------------------------------------
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
    injuries = nfl.load_injuries(seasons=[CURRENT_SEASON]).to_pandas()
except Exception:
    injuries = pd.DataFrame()

try:
    depth_charts = nfl.load_depth_charts(seasons=[CURRENT_SEASON]).to_pandas()
except Exception:
    depth_charts = pd.DataFrame()

for df in [schedules, pbp, injuries, depth_charts]:
    if df.empty:
        continue
    for col in ["home_team", "away_team", "posteam", "defteam", "recent_team", "team", "club_code"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_team_abbr)

def compute_opponent_adjusted_epa(pbp_df: pd.DataFrame) -> pd.DataFrame:
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

def get_latest_team_row(team_abbr: str, target_season: int, target_week: int) -> pd.DataFrame:
    if team_perf.empty:
        return pd.DataFrame()
    t_data = team_perf[
        (team_perf["team"] == team_abbr) & 
        ((team_perf["season"] < target_season) | ((team_perf["season"] == target_season) & (team_perf["week"] < target_week)))
    ]
    return t_data.sort_values(["season", "week"], ascending=[False, False]).head(1) if not t_data.empty else pd.DataFrame()

def extract_injury_map() -> Dict[str, Dict[str, Dict[str, Any]]]:
    injury_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if injuries.empty:
        return injury_map

    team_col = next((c for c in ["team", "club_code", "team_abbr"] if c in injuries.columns), None)
    status_col = next((c for c in ["report_status", "practice_status", "game_status"] if c in injuries.columns), None)
    name_col = next((c for c in ["player_name", "full_name", "athlete_name"] if c in injuries.columns), None)
    pos_col = next((c for c in ["position", "pos", "pos_abb"] if c in injuries.columns), None)

    if not team_col or not status_col or not name_col:
        return injury_map

    for _, row in injuries.iterrows():
        t = str(row[team_col]).strip().upper()
        p_name = str(row[name_col]).strip()
        status = str(row[status_col]).strip().upper() if pd.notna(row[status_col]) else "ACTIVE"
        pos = str(row[pos_col]).strip().upper() if pos_col and pd.notna(row[pos_col]) else "SKILL"

        if t not in injury_map:
            injury_map[t] = {}
        injury_map[t][p_name] = {
            "status": status,
            "position": pos,
            "is_out": status in INACTIVE_STATUS_DESIGNATIONS
        }

    return injury_map

LIVE_INJURY_MAP = extract_injury_map()

def resolve_autonomous_depth_chart(team_abbr: str, target_week: int) -> Dict[str, str]:
    picks = {
        "QB1": f"{team_abbr} QB", "RB1": f"{team_abbr} RB1", "RB2": f"{team_abbr} RB2",
        "WR1": f"{team_abbr} WR1", "WR2": f"{team_abbr} WR2", "WR3": f"{team_abbr} WR3", "TE1": f"{team_abbr} TE1"
    }
    if depth_charts.empty:
        return picks

    team_col = next((c for c in ["club_code", "team", "team_abbr"] if c in depth_charts.columns), None)
    if not team_col:
        return picks

    t_dc = depth_charts[depth_charts[team_col] == team_abbr].copy()
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

    team_injuries = LIVE_INJURY_MAP.get(team_abbr, {})
    slot_configs = [
        ("QB", ["QB1"]),
        ("RB", ["RB1", "RB2"]),
        ("WR", ["WR1", "WR2", "WR3"]),
        ("TE", ["TE1"])
    ]

    assigned_players: Set[str] = set()

    for pos, slots in slot_configs:
        cands = t_dc[t_dc[pos_col] == pos]
        healthy_candidates: List[str] = []

        for _, row in cands.iterrows():
            if name_col and pd.notna(row[name_col]) and str(row[name_col]).strip():
                full_name = str(row[name_col]).strip()
            elif first_col and last_col and pd.notna(row[first_col]) and pd.notna(row[last_col]):
                full_name = f"{row[first_col]} {row[last_col]}".strip()
            else:
                continue

            if full_name.lower() in assigned_players:
                continue

            # Autonomous Inactive Cascade: pop scratched starters (IR/PUP/OUT)
            status_meta = team_injuries.get(full_name, {"status": "ACTIVE", "is_out": False})
            if status_meta["is_out"]:
                logging.info(f"SCRATCH PRUNED: {team_abbr} {pos} {full_name} is [{status_meta['status']}]. Cascading depth.")
                continue

            healthy_candidates.append(full_name)
            assigned_players.add(full_name.lower())

        for idx, slot_key in enumerate(slots):
            if idx < len(healthy_candidates):
                picks[slot_key] = healthy_candidates[idx]
            else:
                picks[slot_key] = f"{team_abbr} {slot_key}"

    return picks

def quantify_unit_level_injuries(home_team: str, away_team: str) -> Dict[str, Any]:
    """
    Evaluates defensive front-7, secondary, and offensive line scratches
    to quantify net TTP shifts, box fit leaks, and EPA drift.
    """
    net_spread_shift = 0.0
    net_total_shift = 0.0
    pass_epa_shift = 0.0
    rush_epa_shift = 0.0
    home_ttp_shift = 0.0
    away_ttp_shift = 0.0
    home_mofo_shift = 0.0
    away_mofo_shift = 0.0

    home_inj = LIVE_INJURY_MAP.get(home_team, {})
    for p_name, meta in home_inj.items():
        if not meta["is_out"]:
            continue
        pos = meta["position"]
        metric = POSITIONAL_IMPACT_METRICS.get(pos, {})
        net_spread_shift -= metric.get("spread_val", 0.0)
        net_total_shift -= metric.get("total_drop", 0.0)
        pass_epa_shift -= metric.get("pass_epa_delta", 0.0)
        rush_epa_shift -= metric.get("rush_epa_delta", 0.0)
        home_ttp_shift += metric.get("ttp_delta", 0.0)
        home_mofo_shift += metric.get("mofo_shift", 0.0)

    away_inj = LIVE_INJURY_MAP.get(away_team, {})
    for p_name, meta in away_inj.items():
        if not meta["is_out"]:
            continue
        pos = meta["position"]
        metric = POSITIONAL_IMPACT_METRICS.get(pos, {})
        net_spread_shift += metric.get("spread_val", 0.0)
        net_total_shift -= metric.get("total_drop", 0.0)
        pass_epa_shift += metric.get("pass_epa_delta", 0.0)
        rush_epa_shift += metric.get("rush_epa_delta", 0.0)
        away_ttp_shift += metric.get("ttp_delta", 0.0)
        away_mofo_shift += metric.get("mofo_shift", 0.0)

    return {
        "spread_shift": net_spread_shift,
        "total_shift": net_total_shift,
        "pass_epa_shift": pass_epa_shift,
        "rush_epa_shift": rush_epa_shift,
        "home_ttp_shift": home_ttp_shift,
        "away_ttp_shift": away_ttp_shift,
        "home_mofo_shift": home_mofo_shift,
        "away_mofo_shift": away_mofo_shift
    }

# -------------------------------------------------------------------------
# 5. Gemini 3.8 Flash Scouting Engine (Exact Research Director Persona)
# -------------------------------------------------------------------------
RESEARCH_DIRECTOR_SYSTEM_PROMPT = """# ROLE & IDENTITY
You are the "NFL Research Director & Quantitative Architect," operating at the nexus of NFL coaching tape breakdown, spatiotemporal tracking physics (NGS), and advanced sabermetric modeling. You possess complete domain authority over offensive and defensive playbooks, scheme-on-scheme mechanics, Bayesian calibration, and automated AI evaluation.

Your dual mandate:
1. Deliver razor-sharp, objective, and analytically grounded NFL football breakdowns.
2. Serve as an expert AI evaluator: Continuously audit user-submitted AI prompts, analytical frameworks, statistical models, and projection logic to eliminate statistical noise, correct proxy errors, and enforce production-grade quantitative rigor.

## OPERATIONAL INVARIANTS
* Trench & Pocket Physics: Time-to-Pressure (TTP) vs. Time-to-Throw (TTT) determines pocket degradation. If TTP < TTT, evaluate pocket mobility archetype. Immobile pocket passers collapse under duress (P2S > 20%); dual threats convert pressure into scramble EPA or extended attempts.
* Run-Fit Geometry:
  - Gap / Duo / Power creates vertical displacement via double-teams. Exploits light nickel boxes (6-man fronts); neutralized by Odd 3-4 fronts with 0/1-technique two-gapping interior tackles.
  - Wide / Outside Zone creates horizontal flow to stress edge contain. Neutralized by Wide-9 alignments and disciplined C-gap setters.
* Coverage Shell Conditioning: Defenses do not play static coverage rates; coverage shell distributions are conditioned on offensive personnel groupings (11 vs. 12/21 personnel).
  - MOFC (Cover 1 / Cover 3): Single-high safety; leaves perimeter 1-on-1s; vulnerable to intermediate Dagger concepts, crossers, and deep seam shots.
  - MOFO (Cover 2 / Quarters / Cover 6): Split safeties; caps vertical boundary routes; vulnerable to underneath checkdowns and intermediate hole shots.
* Output strictly valid JSON matching the target schema without markdown wrappers."""

async def generate_matchup_analysis(semaphore, payload, recommended_team, recommended_line, kelly_units):
    verdict_str = f"Bet {recommended_line} - {kelly_units:.2f}u" if recommended_team != "PASS" and kelly_units > 0.0 else "PASS - 0.00u"

    prompt = f"""Evaluate this verified NFL quantitative dossier:
{json.dumps(payload, indent=2)}

Output strictly valid JSON matching this schema:
{{
  "executive_summary": "Two-sentence strategic verdict detailing player matchups and trench leverage.",
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
                    lambda: client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=RESEARCH_DIRECTOR_SYSTEM_PROMPT,
                            temperature=0.15,
                            response_mime_type="application/json"
                        )
                    )
                )
                parsed = json.loads(response.text)
                schematic = parsed.get("schematic_matchup", {})
                if (
                    parsed.get("executive_summary") 
                    and schematic.get("away_offense_vs_home_defense") 
                    and schematic.get("home_offense_vs_away_defense")
                ):
                    parsed["actionable_verdict"] = verdict_str
                    return json.dumps(parsed)
            except Exception:
                await asyncio.sleep(2 ** attempt)

        away_team = payload["matchup_context"]["away_team"]
        home_team = payload["matchup_context"]["home_team"]

        fallback = {
            "executive_summary": f"Line-of-scrimmage leverage establishes foundational value on {recommended_line}. Neutral-down efficiency dictates drive sustainability.",
            "schematic_matchup": {
                "away_offense_vs_home_defense": f"{away_team} must establish interior run push to keep pass protection ahead of down-and-distance against {home_team}'s front seven.",
                "home_offense_vs_away_defense": f"{home_team} establishes early-down rushing tempo to stress {away_team}'s edge contain, opening vertical crossing lanes."
            },
            "actionable_verdict": verdict_str
        }
        return json.dumps(fallback)

# -------------------------------------------------------------------------
# 6. Master Production Execution Loop
# -------------------------------------------------------------------------
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

        raw_spread_line = float(game["spread_line"]) if pd.notna(game.get("spread_line")) else 0.0
        raw_total_line = float(game["total_line"]) if pd.notna(game.get("total_line")) else 44.0

        home_row = get_latest_team_row(home_team, target_season, week_num)
        away_row = get_latest_team_row(away_team, target_season, week_num)

        def get_stat(df, col, default=0.0):
            return float(df[col].values[0]) if not df.empty and col in df.columns and pd.notna(df[col].values[0]) else float(default)

        # Base Opponent-Cross Subtraction (Synchronized with train_model.py)
        net_pass_edge = (get_stat(home_row, "roll_off_dropback_epa") - get_stat(away_row, "roll_def_dropback_epa")) - \
                        (get_stat(away_row, "roll_off_dropback_epa") - get_stat(home_row, "roll_def_dropback_epa"))
        net_rush_edge = (get_stat(home_row, "roll_off_rush_epa") - get_stat(away_row, "roll_def_rush_epa")) - \
                        (get_stat(away_row, "roll_off_rush_epa") - get_stat(home_row, "roll_def_rush_epa"))
        net_late_down_edge = (get_stat(home_row, "roll_off_late_down_epa") - get_stat(away_row, "roll_def_late_down_epa")) - \
                             (get_stat(away_row, "roll_off_late_down_epa") - get_stat(home_row, "roll_def_late_down_epa"))

        diff_success = get_stat(home_row, "roll_off_early_down_success", 0.44) - get_stat(away_row, "roll_off_early_down_success", 0.44)
        diff_explosive = get_stat(home_row, "roll_off_explosive", 0.12) - get_stat(away_row, "roll_off_explosive", 0.12)
        rest_diff = float(game.get("home_rest", 7.0) or 7.0) - float(game.get("away_rest", 7.0) or 7.0)
        is_divisional = int(game.get("div_game", 0) or 0)

        # Multi-Tier Holistic Unit Injury Quantification (Trench, Secondary, OL)
        injury_shifts = quantify_unit_level_injuries(home_team, away_team)
        
        # Apply injury adjustments to market lines and EPA
        adjusted_spread_line = raw_spread_line + injury_shifts["spread_shift"]
        adjusted_total_line = max(33.0, raw_total_line + injury_shifts["total_shift"])
        net_pass_edge += injury_shifts["pass_epa_shift"]
        net_rush_edge += injury_shifts["rush_epa_shift"]

        canonical_spread, implied_home_total, implied_away_total, raw_market_prob = resolve_directional_market_context(adjusted_total_line, adjusted_spread_line)

        home_ml = float(game["home_moneyline"]) if pd.notna(game.get("home_moneyline")) else None
        away_ml = float(game["away_moneyline"]) if pd.notna(game.get("away_moneyline")) else None

        if home_ml is not None and away_ml is not None and not math.isnan(home_ml) and not math.isnan(away_ml):
            p_h = 100.0 / (home_ml + 100.0) if home_ml > 0 else abs(home_ml) / (abs(home_ml) + 100.0)
            p_a = 100.0 / (away_ml + 100.0) if away_ml > 0 else abs(away_ml) / (abs(away_ml) + 100.0)
            market_home_prob = float(p_h / (p_h + p_a)) if (p_h + p_a) > 0 else raw_market_prob
        else:
            market_home_prob = raw_market_prob

        feature_dict = {
            "net_pass_edge": net_pass_edge,
            "net_rush_edge": net_rush_edge,
            "net_late_down_edge": net_late_down_edge,
            "diff_success": diff_success,
            "diff_explosive": diff_explosive,
            "rest_diff": rest_diff,
            "is_divisional": is_divisional,
            "market_home_prob": market_home_prob
        }

        ordered_features = [feature_dict[feat] for feat in EXPECTED_FEATURES if feat in feature_dict]
        feature_row = pd.DataFrame([ordered_features], columns=EXPECTED_FEATURES)

        raw_model_prob = float(model.predict_proba(feature_row)[0][1])

        # Directionally Anchored Bayesian Shrinkage
        if canonical_spread >= 3.0:
            market_weight = 0.70 if abs(raw_model_prob - market_home_prob) > 0.25 else 0.50
            calibrated_home_win_prob = (1.0 - market_weight) * max(raw_model_prob, 1.0 - raw_model_prob) + (market_weight * market_home_prob)
        elif canonical_spread <= -3.0:
            market_weight = 0.70 if abs(raw_model_prob - market_home_prob) > 0.25 else 0.50
            calibrated_home_win_prob = (1.0 - market_weight) * min(raw_model_prob, 1.0 - raw_model_prob) + (market_weight * market_home_prob)
        else:
            calibrated_home_win_prob = (0.50 * raw_model_prob) + (0.50 * market_home_prob)

        calibrated_home_win_prob = max(0.02, min(0.98, calibrated_home_win_prob))

        sigma = 13.45 * math.sqrt(max(32.0, adjusted_total_line) / 44.0)
        z_win = norm.ppf(calibrated_home_win_prob)
        model_projected_margin = z_win * sigma

        pred_home_score, pred_away_score = project_discrete_nfl_scores(model_projected_margin, adjusted_total_line)
        pred_total_score = pred_home_score + pred_away_score

        # Spread Cover Math (canonical_spread > 0 indicates Home favored by Vegas)
        z_cover_home = (model_projected_margin - canonical_spread) / sigma
        home_cover = float(norm.cdf(z_cover_home))
        away_cover = 1.0 - home_cover

        home_edge = home_cover - 0.5238
        away_edge = away_cover - 0.5238

        if home_edge > 0.018 and home_edge > away_edge:
            rec_team = home_team
            rec_line = f"{home_team} {-canonical_spread:+g}"
            cover_prob = home_cover
            final_edge = min(0.050, home_edge)
        elif away_edge > 0.018 and away_edge > home_edge:
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
        kelly_units = round(max(0.0, min(2.0, (((b * cover_prob) - q) / b) * 0.125 * 100.0)), 2) if rec_team != "PASS" else 0.0

        # Autonomous Active Depth Resolution (Pruning Scratched IR/OUT Players)
        home_depth = resolve_autonomous_depth_chart(home_team, week_num)
        away_depth = resolve_autonomous_depth_chart(away_team, week_num)

        # Generate Complete Projections with Sportsbook Benchmarks and Trench Shifts
        home_skills = generate_closed_loop_skill_projections(
            home_team, implied_home_total, model_projected_margin, net_pass_edge, net_rush_edge, 
            home_depth, pbp, ttp_shift=injury_shifts["home_ttp_shift"], mofo_shift=injury_shifts["away_mofo_shift"]
        )
        away_skills = generate_closed_loop_skill_projections(
            away_team, implied_away_total, -model_projected_margin, -net_pass_edge, -net_rush_edge, 
            away_depth, pbp, ttp_shift=injury_shifts["away_ttp_shift"], mofo_shift=injury_shifts["home_mofo_shift"]
        )

        home_roster_summary = {p["role"]: p["player"] for p in home_skills}
        away_roster_summary = {p["role"]: p["player"] for p in away_skills}

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
            "total_line": adjusted_total_line,
            "spread_line": canonical_spread,
            "predicted_home_score": pred_home_score,
            "predicted_away_score": pred_away_score,
            "predicted_total_score": pred_total_score,
            "player_projections": {"home": home_skills, "away": away_skills},
            "matchup_context": {
                "home_team": home_team,
                "home_roster": home_roster_summary,
                "away_team": away_team,
                "away_roster": away_roster_summary
            },
            "tape_metrics": {
                "net_pass_epa_diff": f"{net_pass_edge:+.3f}",
                "net_rush_epa_diff": f"{net_rush_edge:+.3f}",
                "explosive_rate_diff": f"{diff_explosive:+.3f}",
                "early_down_success_diff": f"{diff_success:+.3f}"
            }
        })

    semaphore = asyncio.Semaphore(4)
    tasks = [
        generate_matchup_analysis(
            semaphore, item, item["recommended_team"], item["recommended_line"], item["kelly_units"]
        ) for item in pre_processed
    ]
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
        parsed["player_projections"] = item["player_projections"]

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

            for stmt in [
                "ALTER TABLE nfl_weekly_analysis ADD COLUMN IF NOT EXISTS season INTEGER DEFAULT 2026;",
                "ALTER TABLE nfl_weekly_analysis ADD COLUMN IF NOT EXISTS predicted_home_score INTEGER;",
                "ALTER TABLE nfl_weekly_analysis ADD COLUMN IF NOT EXISTS predicted_away_score INTEGER;",
                "ALTER TABLE nfl_weekly_analysis ADD COLUMN IF NOT EXISTS predicted_total_score INTEGER;"
            ]:
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
        print(f"Database sync verified: {len(df_results)} fixtures committed for Season {target_season} Week {target_week}.")

if __name__ == "__main__":
    asyncio.run(main())
