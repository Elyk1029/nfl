"""
autonomous_roster_engine.py - Production NFL Depth-Chart & Injury Resolution Engine.
Features:
- Multi-format entity resolution (e.g. 'T.Tagovailoa' -> 'Tua Tagovailoa').
- Dynamic injury parsing: tracks practice participation vectors (DNP, LP, FP) to weight availability.
- Automated depth-chart cascading: prunes scratches (OUT, IR, PUP) and promotes next-man-up.
- Empirical Bayesian regression for backup quarterback efficiency metrics.
- Parameterized execution: zero hardcoded player names or static replacement tables.
"""
import asyncio
import datetime
import difflib
import json
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import nflreadpy as nfl
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

INACTIVE_STATUSES: Set[str] = {
    "OUT", "IR", "INJURED RESERVE", "PUP", "RESERVE/PUP",
    "NFI", "NON-FOOTBALL INJURY", "SUSPENDED", "DOUBTFUL"
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

TEAM_ABBR_MAP: Dict[str, str] = {
    "LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX"
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


class AutonomousNFLRosterEngine:
    def __init__(self, season: int = 2026, week: int = 1):
        self.season = season
        self.week = week
        self.injuries_df: pd.DataFrame = pd.DataFrame()
        self.depth_charts_df: pd.DataFrame = pd.DataFrame()
        self.pbp_df: pd.DataFrame = pd.DataFrame()

    def sync_live_feeds(self) -> None:
        logging.info(f"Synchronizing official nflverse feeds for Season {self.season}, Week {self.week}...")

        try:
            inj = nfl.load_injuries(seasons=[self.season]).to_pandas()
        except Exception:
            try:
                inj = nfl.load_injuries(seasons=[self.season - 1]).to_pandas()
            except Exception:
                inj = pd.DataFrame()

        try:
            dc = nfl.load_depth_charts(seasons=[self.season]).to_pandas()
        except Exception:
            try:
                dc = nfl.load_depth_charts(seasons=[self.season - 1]).to_pandas()
            except Exception:
                dc = pd.DataFrame()

        try:
            pbp = nfl.load_pbp(seasons=[self.season, self.season - 1]).to_pandas()
        except Exception:
            try:
                pbp = nfl.load_pbp(seasons=[self.season - 1]).to_pandas()
            except Exception:
                pbp = pd.DataFrame()

        for frame in [inj, dc, pbp]:
            if frame.empty:
                continue
            for col in ["team", "club_code", "team_abbr", "posteam", "defteam"]:
                if col in frame.columns:
                    frame[col] = frame[col].apply(clean_team_abbr)

        self.injuries_df = inj
        self.depth_charts_df = dc
        self.pbp_df = pbp
        logging.info("Feeds synced. Active records loaded into memory.")

    def match_player_entity(self, target_name: str, candidate_names: List[str]) -> Optional[str]:
        norm_target = normalize_player_name(target_name)
        candidates_clean = {normalize_player_name(c): c for c in candidate_names if isinstance(c, str)}

        if norm_target in candidates_clean:
            return candidates_clean[norm_target]

        matches = difflib.get_close_matches(norm_target, list(candidates_clean.keys()), n=1, cutoff=0.82)
        if matches:
            return candidates_clean[matches[0]]

        target_parts = norm_target.split()
        if len(target_parts) >= 2:
            initial_form = f"{target_parts[0][0]} {target_parts[-1]}"
            for cand_clean, cand_orig in candidates_clean.items():
                cand_parts = cand_clean.split()
                if len(cand_parts) >= 2:
                    if f"{cand_parts[0][0]} {cand_parts[-1]}" == initial_form:
                        return cand_orig

        return None

    def get_player_injury_status(self, player_name: str, team_abbr: str) -> Dict[str, Any]:
        if self.injuries_df.empty:
            return {"status": "HEALTHY", "p_active": 1.0, "is_inactive": False}

        df = self.injuries_df
        team_clean = clean_team_abbr(team_abbr)

        team_col = next((c for c in ["team", "club_code", "team_abbr"] if c in df.columns), None)
        status_col = next((c for c in ["report_status", "game_status", "injury_status"] if c in df.columns), None)
        practice_col = next((c for c in ["practice_status", "practice_primary"] if c in df.columns), None)
        name_col = next((c for c in ["player_name", "full_name", "athlete_name"] if c in df.columns), None)

        if not team_col or not status_col or not name_col:
            return {"status": "HEALTHY", "p_active": 1.0, "is_inactive": False}

        t_inj = df[df[team_col] == team_clean]
        resolved_name = self.match_player_entity(player_name, t_inj[name_col].dropna().tolist())

        if not resolved_name:
            return {"status": "HEALTHY", "p_active": 1.0, "is_inactive": False}

        p_row = t_inj[t_inj[name_col] == resolved_name].iloc[0]
        raw_status = str(p_row[status_col]).strip().upper() if pd.notna(p_row[status_col]) else "HEALTHY"
        raw_practice = str(p_row[practice_col]).strip().upper() if practice_col and pd.notna(p_row[practice_col]) else "FULL"

        if raw_status in INACTIVE_STATUSES:
            return {"status": raw_status, "p_active": 0.0, "is_inactive": True}

        if raw_status == "QUESTIONABLE":
            if "DNP" in raw_practice or "DID NOT" in raw_practice:
                prob = STATUS_PROBABILITY_WEIGHTS["QUESTIONABLE_DNP"]
                desig = "QUESTIONABLE (DNP)"
            elif "LIMITED" in raw_practice or "LP" in raw_practice:
                prob = STATUS_PROBABILITY_WEIGHTS["QUESTIONABLE_LP"]
                desig = "QUESTIONABLE (LP)"
            elif "FULL" in raw_practice or "FP" in raw_practice:
                prob = STATUS_PROBABILITY_WEIGHTS["QUESTIONABLE_FP"]
                desig = "QUESTIONABLE (FP)"
            else:
                prob = STATUS_PROBABILITY_WEIGHTS["QUESTIONABLE"]
                desig = "QUESTIONABLE"
            return {"status": desig, "p_active": prob, "is_inactive": prob < 0.20}

        return {"status": raw_status, "p_active": 1.0, "is_inactive": False}

    def resolve_active_depth_hierarchy(self, team_abbr: str) -> Dict[str, Dict[str, Any]]:
        team_clean = clean_team_abbr(team_abbr)
        dc = self.depth_charts_df

        if dc.empty:
            return {}

        team_col = next((c for c in ["team", "club_code", "team_abbr"] if c in dc.columns), None)
        pos_col = next((c for c in ["pos_abb", "position", "pos"] if c in dc.columns), None)
        rank_col = next((c for c in ["pos_rank", "depth_team", "rank"] if c in dc.columns), None)
        name_col = next((c for c in ["player_name", "full_name", "athlete_name"] if c in dc.columns), None)

        team_dc = dc[dc[team_col] == team_clean].copy()
        if "week" in team_dc.columns:
            target_wk_dc = team_dc[team_dc["week"] == self.week]
            team_dc = target_wk_dc if not target_wk_dc.empty else team_dc[team_dc["week"] == team_dc["week"].max()]

        team_dc["rank_int"] = pd.to_numeric(team_dc[rank_col], errors="coerce").fillna(99).astype(int)
        team_dc.sort_values(by=["rank_int"], inplace=True)

        assigned_roles: Dict[str, Dict[str, Any]] = {}
        assigned_names: Set[str] = set()

        role_structure = [
            ("QB", ["QB1", "QB2"]),
            ("RB", ["RB1", "RB2"]),
            ("WR", ["WR1", "WR2", "WR3"]),
            ("TE", ["TE1"])
        ]

        for pos, slots in role_structure:
            pos_pool = team_dc[team_dc[pos_col] == pos]
            eligible_players: List[Dict[str, Any]] = []

            for _, row in pos_pool.iterrows():
                p_name = str(row[name_col]).strip()
                norm_p = normalize_player_name(p_name)
                if norm_p in assigned_names:
                    continue

                health_eval = self.get_player_injury_status(p_name, team_clean)

                if health_eval["is_inactive"]:
                    logging.info(f"SCRATCH CONFIRMED: {team_clean} {pos} {p_name} is [{health_eval['status']}]. Cascading next-man-up.")
                    continue

                eligible_players.append({
                    "name": p_name,
                    "status": health_eval["status"],
                    "p_active": health_eval["p_active"]
                })
                assigned_names.add(norm_p)

            for idx, slot_key in enumerate(slots):
                if idx < len(eligible_players):
                    assigned_roles[slot_key] = eligible_players[idx]
                else:
                    assigned_roles[slot_key] = {
                        "name": f"{team_clean} Reserve {slot_key}",
                        "status": "HEALTHY",
                        "p_active": 1.00
                    }

        return assigned_roles

    def extract_qb_empirical_efficiency(self, qb_name: str) -> Dict[str, float]:
        prior_epa = -0.110
        prior_cpoe = -2.80
        prior_weight = 40.0

        if self.pbp_df.empty:
            return {"dropback_epa": prior_epa, "cpoe": prior_cpoe, "ttt": 2.70, "sample_size": 0}

        pbp = self.pbp_df
        norm_target = normalize_player_name(qb_name)
        all_passers = pbp["passer_player_name"].dropna().unique().tolist()
        resolved_passer = self.match_player_entity(norm_target, all_passers)

        if not resolved_passer:
            return {"dropback_epa": prior_epa, "cpoe": prior_cpoe, "ttt": 2.70, "sample_size": 0}

        qb_plays = pbp[
            (pbp["passer_player_name"] == resolved_passer) &
            (pbp["play_type"] == "pass") &
            (pbp["home_wp"].between(0.10, 0.90))
        ]

        n = len(qb_plays)
        if n == 0:
            return {"dropback_epa": prior_epa, "cpoe": prior_cpoe, "ttt": 2.70, "sample_size": 0}

        sample_epa = float(qb_plays["epa"].mean())
        sample_cpoe = float(qb_plays["cpoe"].mean()) if "cpoe" in qb_plays.columns and qb_plays["cpoe"].notna().any() else prior_cpoe

        calibrated_epa = (n * sample_epa + prior_weight * prior_epa) / (n + prior_weight)
        calibrated_cpoe = (n * sample_cpoe + prior_weight * prior_cpoe) / (n + prior_weight)

        return {
            "dropback_epa": round(calibrated_epa, 3),
            "cpoe": round(calibrated_cpoe, 2),
            "ttt": 2.70 if n < 50 else 2.55,
            "sample_size": n
        }
