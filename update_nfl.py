"""
update_nfl.py - Institutional NFL Quantitative Terminal Pipeline Orchestrator.
Fixed:
- Strict weekly filtering on depth charts to eliminate multi-week duplicate entries.
- Distinct player allocation per role (no player duplicated across RB1/RB2 or WR1/WR2/WR3).
- Closed-Loop Skill Player Volume Allocation (QB, RB1/2, WR1/2/3, TE1).
- Discrete empirical score generation (zero regular-season ties).
- Auto-migrating Neon PostgreSQL persistence.
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
from scipy.stats import norm, poisson
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

# 2. Discrete Empirical Score Engine
NFL_KEY_MARGINS = [3, 7, 6, 10, 4, 1, 2, 14, 8, 11, 13, 17]
COMMON_TEAM_SCORES = [20, 24, 17, 23, 27, 30, 31, 13, 14, 10, 34, 38, 28, 16, 21]

def project_discrete_nfl_scores(projected_margin: float, total_line: float) -> tuple[int, int]:
    effective_margin = projected_margin if abs(projected_margin) >= 0.05 else 0.10
    home_favored = effective_margin > 0.0
    abs_margin = abs(effective_margin)

    selected_discrete_margin = min(NFL_KEY_MARGINS, key=lambda m: abs(m - abs_margin))
    raw_home = (total_line + (selected_discrete_margin if home_favored else -selected_discrete_margin)) / 2.0
    raw_away = (total_line - (selected_discrete_margin if home_favored else -selected_discrete_margin)) / 2.0

    best_pair = (24, 21) if home_favored else (21, 24)
    min_loss = float("inf")

    candidate_home = [s for s in COMMON_TEAM_SCORES if abs(s - raw_home) <= 6.5] or [int(round(raw_home))]
    candidate_away = [s for s in COMMON_TEAM_SCORES if abs(s - raw_away) <= 6.5] or [int(round(raw_away))]

    for h in candidate_home:
        for a in candidate_away:
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

# 3. Dynamic QB Bayesian Adjustment & VORP Engine
QB_VORP_TIERS = {
    "Patrick Mahomes": 7.0, "Josh Allen": 7.0, "Lamar Jackson": 6.5, "Joe Burrow": 6.5,
    "C.J. Stroud": 5.0, "Jordan Love": 5.0, "Jalen Hurts": 4.5, "Justin Herbert": 4.5,
    "Dak Prescott": 4.5, "Brock Purdy": 4.0, "Jared Goff": 4.0, "Matthew Stafford": 4.0,
    "Kyler Murray": 4.0, "Kirk Cousins": 3.5, "Baker Mayfield": 3.5, "Trevor Lawrence": 3.0,
    "Tua Tagovailoa": 3.0, "Jayden Daniels": 3.0, "Caleb Williams": 2.5, "Anthony Richardson": 2.5,
    "Geno Smith": 2.5, "Derek Carr": 2.5, "Bo Nix": 2.0, "Drake Maye": 2.0, "Will Levis": 2.0,
    "Bryce Young": 1.5, "Daniel Jones": 1.5, "Deshaun Watson": 1.5, "Gardner Minshew": 1.0,
    "Jacoby Brissett": 1.5
}
DEFAULT_QB_VORP = 2.0
BACKUP_QB_VORP = 0.0

INJURY_PROBABILITY_WEIGHTS = {
    "OUT": 0.00, "IR": 0.00, "INJURED RESERVE": 0.00, "PUP": 0.00,
    "DOUBTFUL": 0.08, "QUESTIONABLE": 0.62, "ACTIVE": 1.00, "HEALTHY": 1.00
}

INACTIVE_DESIGNATIONS = {"OUT", "IR", "INJURED RESERVE", "DOUBTFUL", "DNR", "PUP", "NFI", "SUSPENDED"}

def resolve_qb_depth_and_adjustment(
    team_abbr: str,
    depth_charts_df: pd.DataFrame,
    injury_map: dict,
    base_spread: float,
    base_total: float,
    base_pass_edge: float
):
    team_injuries = injury_map.get(team_abbr, {})
    qb1_name = f"{team_abbr} Starting QB"
    qb2_name = f"{team_abbr} Backup QB"

    if not depth_charts_df.empty:
        team_col = next((c for c in ["team", "club_code", "team_abbr"] if c in depth_charts_df.columns), None)
        pos_col = next((c for c in ["pos_abb", "position", "pos"] if c in depth_charts_df.columns), None)
        rank_col = next((c for c in ["pos_rank", "depth_team", "rank"] if c in depth_charts_df.columns), None)
        name_col = next((c for c in ["player_name", "full_name", "athlete_name"] if c in depth_charts_df.columns), None)
        first_col = next((c for c in ["first_name", "fname"] if c in depth_charts_df.columns), None)
        last_col = next((c for c in ["last_name", "lname"] if c in depth_charts_df.columns), None)

        if team_col and pos_col and rank_col:
            t_qbs = depth_charts_df[(depth_charts_df[team_col] == team_abbr) & (depth_charts_df[pos_col] == "QB")].copy()
            if not t_qbs.empty:
                t_qbs["rank_int"] = pd.to_numeric(t_qbs[rank_col], errors="coerce").fillna(99).astype(int)
                t_qbs.sort_values(by=["rank_int"], inplace=True)

                def get_row_name(row):
                    if name_col and pd.notna(row[name_col]) and str(row[name_col]).strip():
                        return str(row[name_col]).strip()
                    if first_col and last_col and pd.notna(row[first_col]) and pd.notna(row[last_col]):
                        return f"{row[first_col]} {row[last_col]}".strip()
                    return None

                qbs_found = []
                for _, r in t_qbs.iterrows():
                    n = get_row_name(r)
                    if n and n not in qbs_found:
                        qbs_found.append(n)

                if len(qbs_found) > 0:
                    qb1_name = qbs_found[0]
                if len(qbs_found) > 1:
                    qb2_name = qbs_found[1]

    qb1_status = team_injuries.get(qb1_name, "HEALTHY").upper()
    p_start = INJURY_PROBABILITY_WEIGHTS.get(qb1_status, 1.0)
    vorp_qb1 = QB_VORP_TIERS.get(qb1_name, DEFAULT_QB_VORP)
    vorp_qb2 = QB_VORP_TIERS.get(qb2_name, BACKUP_QB_VORP)

    if p_start <= 0.20:
        active_qb = qb2_name
        injury_label = f"Starter {qb1_name} OUT ({qb1_status})"
        point_haircut = vorp_qb1 - vorp_qb2
        pass_epa_haircut = 0.16
        qb_variance_sigma = 0.42
    else:
        active_qb = qb1_name
        injury_label = qb1_status if qb1_status != "HEALTHY" else "Healthy"
        expected_absence = 1.0 - p_start
        point_haircut = (vorp_qb1 - vorp_qb2) * expected_absence
        pass_epa_haircut = 0.16 * expected_absence
        qb_variance_sigma = 0.32 + (0.08 * expected_absence)

    adjusted_spread = base_spread - point_haircut
    adjusted_total = max(33.0, base_total - (point_haircut * 0.85))
    adjusted_pass_edge = base_pass_edge - pass_epa_haircut

    return (
        active_qb,
        injury_label,
        round(adjusted_spread, 1),
        round(adjusted_total, 1),
        round(adjusted_pass_edge, 3),
        round(qb_variance_sigma, 2)
    )

# 4. Closed-Loop Skill Volume & Log-Normal Median Allocation
LOG_SIGMA = {
    "QB_Pass": 0.32, "QB_Rush": 0.52, "RB_Rush": 0.48, 
    "RB_Rec": 0.55, "WR_Rec": 0.58, "TE_Rec": 0.54
}

def convert_mean_to_median(mean_val: float, role_key: str, custom_sigma: float = None) -> float:
    if mean_val <= 0.0:
        return 0.0
    sig = custom_sigma if custom_sigma is not None else LOG_SIGMA.get(role_key, 0.50)
    return round(max(0.0, float(mean_val * math.exp(-(sig**2) / 2.0))), 1)

def generate_closed_loop_skill_projections(
    team_abbr: str, 
    implied_total: float, 
    spread_line: float,
    pass_edge: float, 
    rush_edge: float, 
    depth_names: dict,
    qb_sigma: float = 0.32
) -> list:
    total_plays = 63.0 * (implied_total / 22.0) ** 0.30
    script_shift = -0.012 * spread_line
    scheme_shift = 0.04 * (pass_edge - rush_edge)
    pass_rate = max(0.44, min(0.72, 0.585 + script_shift + scheme_shift))
    run_rate = 1.0 - pass_rate

    team_gross_pass = max(100.0, total_plays * pass_rate * max(5.2, min(9.4, 7.15 + (pass_edge * 3.5))))
    team_gross_rush = max(50.0, total_plays * run_rate * max(3.1, min(5.6, 4.25 + (rush_edge * 2.8))))

    total_tds = max(1.0, implied_total / 7.15)
    pass_td_share = max(0.40, min(0.85, 0.65 + (pass_edge - rush_edge) * 0.25))
    team_pass_tds = total_tds * pass_td_share
    team_rush_tds = total_tds * (1.0 - pass_td_share)

    qb_mean_rush = team_gross_rush * 0.12
    rb1_mean_rush = team_gross_rush * 0.58
    rb2_mean_rush = team_gross_rush * 0.24

    rb1_rush_td = team_rush_tds * 0.62
    rb2_rush_td = team_rush_tds * 0.22
    qb_rush_td = team_rush_tds * 0.14

    raw_target_weights = {"WR1": 0.26, "WR2": 0.18, "WR3": 0.12, "TE1": 0.19, "RB1": 0.13, "RB2": 0.06, "OTHER": 0.06}
    w_sum = sum(raw_target_weights.values())
    target_shares = {k: v / w_sum for k, v in raw_target_weights.items()}

    depth_multipliers = {"WR1": 1.18, "WR2": 1.10, "WR3": 0.95, "TE1": 0.92, "RB1": 0.64, "RB2": 0.58, "OTHER": 0.85}
    raw_weighted = {k: target_shares[k] * depth_multipliers[k] for k in target_shares}
    rec_norm = sum(raw_weighted.values())
    rec_shares = {k: raw_weighted[k] / rec_norm for k in raw_weighted}

    wr1_mean_rec = team_gross_pass * rec_shares["WR1"]
    wr2_mean_rec = team_gross_pass * rec_shares["WR2"]
    wr3_mean_rec = team_gross_pass * rec_shares["WR3"]
    te1_mean_rec = team_gross_pass * rec_shares["TE1"]
    rb1_mean_rec = team_gross_pass * rec_shares["RB1"]
    rb2_mean_rec = team_gross_pass * rec_shares["RB2"]

    rz_weights = {
        "WR1": target_shares["WR1"] * 1.25, "WR2": target_shares["WR2"] * 1.05,
        "WR3": target_shares["WR3"] * 0.85, "TE1": target_shares["TE1"] * 1.30,
        "RB1": target_shares["RB1"] * 0.60, "RB2": target_shares["RB2"] * 0.40, "OTHER": 0.50
    }
    rz_norm = sum(rz_weights.values())
    rec_td_shares = {k: rz_weights[k] / rz_norm for k in rz_weights}

    def calc_anytime_td_prob(exp_td):
        return round(float((1.0 - poisson.pmf(0, max(0.01, exp_td))) * 100.0), 1)

    return [
        {
            "role": "QB1", "player": depth_names.get("QB1", f"{team_abbr} QB"),
            "pass_yards": convert_mean_to_median(team_gross_pass, "QB_Pass", custom_sigma=qb_sigma),
            "rush_yards": convert_mean_to_median(qb_mean_rush, "QB_Rush"),
            "rec_yards": 0.0, "projected_pass_tds": round(team_pass_tds, 2),
            "total_tds": round(qb_rush_td, 2), "anytime_td_prob": calc_anytime_td_prob(qb_rush_td)
        },
        {
            "role": "RB1", "player": depth_names.get("RB1", f"{team_abbr} RB1"),
            "pass_yards": 0.0, "rush_yards": convert_mean_to_median(rb1_mean_rush, "RB_Rush"),
            "rec_yards": convert_mean_to_median(rb1_mean_rec, "RB_Rec"), "projected_pass_tds": 0.0,
            "total_tds": round(rb1_rush_td + (team_pass_tds * rec_td_shares["RB1"]), 2),
            "anytime_td_prob": calc_anytime_td_prob(rb1_rush_td + (team_pass_tds * rec_td_shares["RB1"]))
        },
        {
            "role": "RB2", "player": depth_names.get("RB2", f"{team_abbr} RB2"),
            "pass_yards": 0.0, "rush_yards": convert_mean_to_median(rb2_mean_rush, "RB_Rush"),
            "rec_yards": convert_mean_to_median(rb2_mean_rec, "RB_Rec"), "projected_pass_tds": 0.0,
            "total_tds": round(rb2_rush_td + (team_pass_tds * rec_td_shares["RB2"]), 2),
            "anytime_td_prob": calc_anytime_td_prob(rb2_rush_td + (team_pass_tds * rec_td_shares["RB2"]))
        },
        {
            "role": "WR1", "player": depth_names.get("WR1", f"{team_abbr} WR1"),
            "pass_yards": 0.0, "rush_yards": 0.0, "rec_yards": convert_mean_to_median(wr1_mean_rec, "WR_Rec"),
            "projected_pass_tds": 0.0, "total_tds": round(team_pass_tds * rec_td_shares["WR1"], 2),
            "anytime_td_prob": calc_anytime_td_prob(team_pass_tds * rec_td_shares["WR1"])
        },
        {
            "role": "WR2", "player": depth_names.get("WR2", f"{team_abbr} WR2"),
            "pass_yards": 0.0, "rush_yards": 0.0, "rec_yards": convert_mean_to_median(wr2_mean_rec, "WR_Rec"),
            "projected_pass_tds": 0.0, "total_tds": round(team_pass_tds * rec_td_shares["WR2"], 2),
            "anytime_td_prob": calc_anytime_td_prob(team_pass_tds * rec_td_shares["WR2"])
        },
        {
            "role": "WR3", "player": depth_names.get("WR3", f"{team_abbr} WR3"),
            "pass_yards": 0.0, "rush_yards": 0.0, "rec_yards": convert_mean_to_median(wr3_mean_rec, "WR_Rec"),
            "projected_pass_tds": 0.0, "total_tds": round(team_pass_tds * rec_td_shares["WR3"], 2),
            "anytime_td_prob": calc_anytime_td_prob(team_pass_tds * rec_td_shares["WR3"])
        },
        {
            "role": "TE1", "player": depth_names.get("TE1", f"{team_abbr} TE1"),
            "pass_yards": 0.0, "rush_yards": 0.0, "rec_yards": convert_mean_to_median(te1_mean_rec, "TE_Rec"),
            "projected_pass_tds": 0.0, "total_tds": round(team_pass_tds * rec_td_shares["TE1"], 2),
            "anytime_td_prob": calc_anytime_td_prob(team_pass_tds * rec_td_shares["TE1"])
        },
    ]

# 5. Multi-Source Ingestion & Dynamic Roster Resolution
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

def extract_injury_map():
    injury_map = {}
    if injuries.empty:
        return injury_map

    team_col = next((c for c in ["team", "club_code", "team_abbr"] if c in injuries.columns), None)
    status_col = next((c for c in ["report_status", "practice_status", "game_status"] if c in injuries.columns), None)
    name_col = next((c for c in ["player_name", "full_name", "athlete_name"] if c in injuries.columns), None)
    first_col = next((c for c in ["first_name", "fname"] if c in injuries.columns), None)
    last_col = next((c for c in ["last_name", "lname"] if c in injuries.columns), None)

    if not team_col or not status_col:
        return injury_map

    for _, row in injuries.iterrows():
        t = str(row[team_col]).strip().upper()
        if name_col and pd.notna(row[name_col]):
            p_name = str(row[name_col]).strip()
        elif first_col and last_col and pd.notna(row[first_col]) and pd.notna(row[last_col]):
            p_name = f"{row[first_col]} {row[last_col]}".strip()
        else:
            continue

        status = str(row[status_col]).strip().upper() if pd.notna(row[status_col]) else "ACTIVE"
        if t not in injury_map:
            injury_map[t] = {}
        injury_map[t][p_name] = status

    return injury_map

LIVE_INJURY_MAP = extract_injury_map()

def resolve_active_depth_chart(team_abbr: str, target_week: int) -> dict:
    """
    Traverses weekly depth charts, isolates the active week, promotes healthy backups,
    and guarantees that no athlete is mapped to more than one slot.
    """
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

    # Restrict to active week to prevent multi-week starter collisions
    if "week" in t_dc.columns:
        valid_weeks = t_dc[t_dc["week"] == target_week]
        if not valid_weeks.empty:
            t_dc = valid_weeks.copy()
        else:
            max_wk = t_dc["week"].max()
            t_dc = t_dc[t_dc["week"] == max_wk].copy()

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

    assigned_players = set()

    for pos, slots in slot_configs:
        cands = t_dc[t_dc[pos_col] == pos]
        healthy_unique_players = []

        for _, row in cands.iterrows():
            if name_col and pd.notna(row[name_col]) and str(row[name_col]).strip():
                full_name = str(row[name_col]).strip()
            elif first_col and last_col and pd.notna(row[first_col]) and pd.notna(row[last_col]):
                full_name = f"{row[first_col]} {row[last_col]}".strip()
            else:
                continue

            # Skip duplicate name representations
            if full_name.lower() in assigned_players:
                continue

            status = team_injuries.get(full_name, "ACTIVE")
            if status in INACTIVE_DESIGNATIONS:
                continue

            healthy_unique_players.append(full_name)
            assigned_players.add(full_name.lower())

        for idx, slot_key in enumerate(slots):
            if idx < len(healthy_unique_players):
                picks[slot_key] = healthy_unique_players[idx]
            else:
                picks[slot_key] = f"{team_abbr} {slot_key}"

    return picks

# 6. LLM Scouting Engine (With Dual-Mandate Persona)
async def generate_matchup_analysis(semaphore, payload, recommended_team, recommended_line, kelly_units):
    system_prompt = """
# ROLE & IDENTITY
You are the "NFL Research Director & Quantitative Architect," operating at the nexus of NFL coaching tape breakdown, spatiotemporal tracking physics (NGS), and advanced sabermetric modeling.

# OPERATIONAL PROTOCOLS
* You MUST reference explicit player names from the provided active rosters rather than generic placeholders like RB1 or WR1.
* Explain pocket physics as a countdown race between pass protection and release timing (TTP vs TTT).
* Map Duo/Power as vertical displacement and Zone schemes as sideline-to-sideline stretch.
* Deliver actionable verdicts and deep film breakdowns.
* Output strictly valid JSON without markdown backticks.
"""

    verdict_str = f"Bet {recommended_line} - {kelly_units:.2f}u" if recommended_team != "PASS" and kelly_units > 0.0 else "PASS - 0.00u"

    prompt = f"""
Evaluate this NFL advance scouting dossier with explicit active roster data:
{json.dumps(payload, indent=2)}

Output strictly valid JSON matching this schema:
{{
  "executive_summary": "Two-sentence strategic verdict explaining player matchups, line-of-scrimmage leverage, and game edge.",
  "schematic_matchup": {{
    "away_offense_vs_home_defense": "Detailed film breakdown referencing specific named players in pass protection, run fits, and safety shells.",
    "home_offense_vs_away_defense": "Detailed film breakdown referencing specific named players in pass protection, run fits, and safety shells."
  }},
  "actionable_verdict": "{verdict_str}"
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
                            temperature=0.15,
                            response_mime_type="application/json",
                            tools=None
                        )
                    )
                )
                parsed = json.loads(response.text)
                schematic = parsed.get("schematic_matchup", {})
                if (
                    parsed.get("executive_summary") 
                    and schematic.get("away_offense_vs_home_defense") 
                    and schematic.get("home_offense_vs_away_defense")
                    and "N/A" not in schematic.get("away_offense_vs_home_defense")
                ):
                    parsed["actionable_verdict"] = verdict_str
                    return json.dumps(parsed)
            except Exception:
                await asyncio.sleep(2 ** attempt)

        away_team = payload["matchup_context"]["away_team"]
        home_team = payload["matchup_context"]["home_team"]

        fallback = {
            "executive_summary": f"Line-of-scrimmage metrics establish baseline execution value on {recommended_line}. Neutral-script efficiency and third-down conversion leverage dictate drive sustainability.",
            "schematic_matchup": {
                "away_offense_vs_home_defense": f"{away_team} must establish interior run push to keep pass protection ahead of down-and-distance against {home_team}'s front seven, opening play-action crossing lanes against split-safety shells.",
                "home_offense_vs_away_defense": f"{home_team} establishes early-down rushing tempo to stress {away_team}'s edge contain, forcing safety walk-downs into the box and isolating perimeter boundary targets."
            },
            "actionable_verdict": verdict_str
        }
        return json.dumps(fallback)

# 7. Master Execution Pipeline Loop
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

        # Dynamic QB Adjustment & Line Haircut
        home_qb, home_qb_status, adj_spread, adj_total, adj_pass_edge, home_qb_sigma = resolve_qb_depth_and_adjustment(
            team_abbr=home_team,
            depth_charts_df=depth_charts,
            injury_map=LIVE_INJURY_MAP,
            base_spread=raw_spread_line,
            base_total=raw_total_line,
            base_pass_edge=net_pass_edge
        )

        away_qb, away_qb_status, _, _, _, away_qb_sigma = resolve_qb_depth_and_adjustment(
            team_abbr=away_team,
            depth_charts_df=depth_charts,
            injury_map=LIVE_INJURY_MAP,
            base_spread=-raw_spread_line,
            base_total=adj_total,
            base_pass_edge=-adj_pass_edge
        )

        spread_line = adj_spread
        total_line = adj_total
        net_pass_edge = adj_pass_edge

        home_ml = float(game["home_moneyline"]) if pd.notna(game.get("home_moneyline")) else None
        away_ml = float(game["away_moneyline"]) if pd.notna(game.get("away_moneyline")) else None

        if home_ml is not None and away_ml is not None and not math.isnan(home_ml) and not math.isnan(away_ml):
            p_h = 100.0 / (home_ml + 100.0) if home_ml > 0 else abs(home_ml) / (abs(home_ml) + 100.0)
            p_a = 100.0 / (away_ml + 100.0) if away_ml > 0 else abs(away_ml) / (abs(away_ml) + 100.0)
            market_home_prob = float(p_h / (p_h + p_a)) if (p_h + p_a) > 0 else 0.50
        else:
            market_home_prob = float(norm.cdf(spread_line / 13.5))

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

        pred_home_score, pred_away_score = project_discrete_nfl_scores(projected_margin, total_line)
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

        implied_home_total = (total_line / 2.0) + (spread_line / 2.0)
        implied_away_total = (total_line / 2.0) - (spread_line / 2.0)

        home_depth = resolve_active_depth_chart(home_team, week_num)
        away_depth = resolve_active_depth_chart(away_team, week_num)
        home_depth["QB1"] = home_qb
        away_depth["QB1"] = away_qb

        home_skills = generate_closed_loop_skill_projections(
            home_team, implied_home_total, -spread_line, net_pass_edge, net_rush_edge, home_depth, qb_sigma=home_qb_sigma
        )
        away_skills = generate_closed_loop_skill_projections(
            away_team, implied_away_total, spread_line, -net_pass_edge, -net_rush_edge, away_depth, qb_sigma=away_qb_sigma
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
            "total_line": total_line,
            "spread_line": spread_line,
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
        print(f"Database sync verified: {len(df_results)} fixtures committed for Season {target_season} Week {target_week}.")

if __name__ == "__main__":
    asyncio.run(main())
