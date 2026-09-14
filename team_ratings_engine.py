"""
team_ratings_engine.py - Institutional NFL Q-OVR vs. Madden Discrepancy Engine.
Computes empirical 0-100 Quantitative Ratings (Q-OVR) from neutral-down PBP tracking,
ingests commercial Madden ratings, and constructs unit-level variance tables for all 32 teams.
"""
import logging
import math
import os
from typing import Any, Dict, List
import nflreadpy as nfl
import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

db_url = os.environ.get("DATABASE_URL")
if not db_url:
    raise ValueError("FATAL: DATABASE_URL must be configured.")

engine = create_engine(db_url, pool_size=5, max_overflow=10, pool_pre_ping=True)

TEAM_ABBR_MAP = {"LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX"}
def clean_team_abbr(t: str) -> str:
    return TEAM_ABBR_MAP.get(str(t).strip().upper(), str(t).strip().upper())

ALL_32_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
]

class QuantitativeRatingsPipeline:
    def __init__(self, season: int = 2026):
        self.season = season
        self.pbp_df = pd.DataFrame()
        self.madden_df = pd.DataFrame()

    def sync_data(self) -> None:
        try:
            pbp = nfl.load_pbp(seasons=[self.season, self.season - 1]).to_pandas()
        except Exception:
            pbp = pd.DataFrame()

        for col in ["posteam", "defteam", "home_team", "away_team"]:
            if col in pbp.columns:
                pbp[col] = pbp[col].apply(clean_team_abbr)
        self.pbp_df = pbp
        self.madden_df = self._fetch_ea_madden_ratings()

    def _fetch_ea_madden_ratings(self) -> pd.DataFrame:
        url = "https://ratings-api.ea.com/v2/entities/m25-ratings"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        players = []
        offset = 0

        team_name_map = {
            "49ers": "SF", "Bears": "CHI", "Bengals": "CIN", "Bills": "BUF",
            "Broncos": "DEN", "Browns": "CLE", "Buccaneers": "TB", "Cardinals": "ARI",
            "Chargers": "LAC", "Chiefs": "KC", "Colts": "IND", "Commanders": "WAS",
            "Cowboys": "DAL", "Dolphins": "MIA", "Eagles": "PHI", "Falcons": "ATL",
            "Giants": "NYG", "Jaguars": "JAX", "Jets": "NYJ", "Lions": "DET",
            "Packers": "GB", "Panthers": "CAR", "Patriots": "NE", "Raiders": "LV",
            "Rams": "LA", "Ravens": "BAL", "Saints": "NO", "Seahawks": "SEA",
            "Steelers": "PIT", "Texans": "HOU", "Titans": "TEN", "Vikings": "MIN"
        }

        for _ in range(12):
            try:
                resp = requests.get(url, params={"offset": offset, "limit": 250}, headers=headers, timeout=10)
                if resp.status_code != 200:
                    break
                data = resp.json()
                docs = data.get("docs", [])
                if not docs:
                    break

                for d in docs:
                    raw_team = str(d.get("team", ""))
                    matched_team = "FA"
                    for k, v in team_name_map.items():
                        if k.lower() in raw_team.lower():
                            matched_team = v
                            break

                    players.append({
                        "player_name": f"{d.get('firstName', '').strip()} {d.get('lastName', '').strip()}".strip(),
                        "team": matched_team,
                        "position": str(d.get("position", "")).strip().upper(),
                        "madden_ovr": int(d.get("overall_rating", d.get("ovr", 70))),
                        "pass_block": int(d.get("pass_block", d.get("passBlock", 65))),
                        "run_block": int(d.get("run_block", d.get("runBlock", 65))),
                        "power_moves": int(d.get("power_moves", d.get("powerMoves", 65))),
                        "finesse_moves": int(d.get("finesse_moves", d.get("finesseMoves", 65))),
                        "man_coverage": int(d.get("man_coverage", d.get("manCoverage", 65))),
                        "zone_coverage": int(d.get("zone_coverage", d.get("zoneCoverage", 65)))
                    })

                offset += len(docs)
                if offset >= data.get("total", 0):
                    break
            except Exception:
                break

        return pd.DataFrame(players)

    def calculate_q_ovr(self) -> pd.DataFrame:
        records = []
        pbp = self.pbp_df
        neutral = pbp[pbp["home_wp"].between(0.10, 0.90)].copy() if not pbp.empty and "home_wp" in pbp.columns else pd.DataFrame()

        for team in ALL_32_TEAMS:
            if not neutral.empty:
                t_p = neutral[(neutral["posteam"] == team) & (neutral["play_type"] == "pass")]
                t_r = neutral[(neutral["posteam"] == team) & (neutral["play_type"] == "run")]
                t_dp = neutral[(neutral["defteam"] == team) & (neutral["play_type"] == "pass")]
                t_dr = neutral[(neutral["defteam"] == team) & (neutral["play_type"] == "run")]

                off_drop_epa = float(t_p["epa"].mean()) if not t_p.empty else 0.00
                off_rush_epa = float(t_r["epa"].mean()) if not t_r.empty else -0.08
                def_drop_epa = float(t_dp["epa"].mean()) if not t_dp.empty else 0.00
                def_rush_epa = float(t_dr["epa"].mean()) if not t_dr.empty else -0.08
                off_succ = float(neutral[neutral["posteam"] == team]["success"].mean()) if not neutral.empty else 0.44
            else:
                off_drop_epa, off_rush_epa, def_drop_epa, def_rush_epa, off_succ = 0.0, -0.08, 0.0, -0.08, 0.44

            q_off = max(55.0, min(99.0, 75.0 + (off_drop_epa * 65.0) + (off_rush_epa * 35.0) + (off_succ - 0.44) * 80.0))
            q_def = max(55.0, min(99.0, 75.0 - (def_drop_epa * 65.0) - (def_rush_epa * 35.0)))
            q_pass_pro = max(55.0, min(99.0, 74.0 + (off_drop_epa * 50.0)))
            q_pass_rush = max(55.0, min(99.0, 74.0 - (def_drop_epa * 50.0)))
            q_sec = max(55.0, min(99.0, 75.0 - (def_drop_epa * 40.0)))
            q_overall = round(0.52 * q_off + 0.48 * q_def, 1)

            m_ovr, m_off, m_def, m_pass_pro, m_pass_rush, m_sec = 78.0, 78.0, 78.0, 75.0, 75.0, 75.0
            if not self.madden_df.empty:
                t_m = self.madden_df[self.madden_df["team"] == team]
                if not t_m.empty:
                    m_ovr = round(float(t_m.sort_values("madden_ovr", ascending=False).head(35)["madden_ovr"].mean()), 1)
                    ol = t_m[t_m["position"].isin(["LT", "LG", "C", "RG", "RT", "OL", "OT", "OG"])]
                    m_pass_pro = round(float(ol["pass_block"].mean()), 1) if not ol.empty else 75.0
                    dl = t_m[t_m["position"].isin(["RE", "LE", "DT", "EDGE", "DE"])]
                    m_pass_rush = round(float(((dl["power_moves"] + dl["finesse_moves"]) / 2.0).mean()), 1) if not dl.empty else 75.0
                    db = t_m[t_m["position"].isin(["CB", "FS", "SS", "S"])]
                    m_sec = round(float(((db["man_coverage"] + db["zone_coverage"]) / 2.0).mean()), 1) if not db.empty else 75.0
                    m_off = round(float(t_m[t_m["position"].isin(["QB", "WR", "TE", "RB", "HB", "LT", "LG", "C", "RG", "RT"])]["madden_ovr"].head(15).mean()), 1)
                    m_def = round(float(t_m[t_m["position"].isin(["RE", "LE", "DT", "EDGE", "DE", "MLB", "OLB", "CB", "FS", "SS"])]["madden_ovr"].head(15).mean()), 1)

            discrepancy = round(q_overall - m_ovr, 1)
            signal = "🔥 High Quant Upside" if discrepancy >= 3.5 else ("❄️ Fragile Composite" if discrepancy <= -3.5 else "⚖️ Market Efficient")

            records.append({
                "team": team, "model_q_ovr": q_overall, "madden_ovr": m_ovr, "discrepancy": discrepancy, "signal": signal,
                "model_offense": round(q_off, 1), "madden_offense": m_off, "model_defense": round(q_def, 1), "madden_defense": m_def,
                "model_pass_protection": round(q_pass_pro, 1), "madden_pass_protection": m_pass_pro,
                "model_pass_rush": round(q_pass_rush, 1), "madden_pass_rush": m_pass_rush,
                "model_secondary": round(q_sec, 1), "madden_secondary": m_sec,
                "net_dropback_epa": round(off_drop_epa, 3), "net_rush_epa": round(off_rush_epa, 3)
            })

        return pd.DataFrame(records).sort_values("model_q_ovr", ascending=False)

    def persist_to_database(self, df_ratings: pd.DataFrame) -> None:
        if df_ratings.empty:
            return
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS nfl_team_ratings_comparison (
                    team TEXT PRIMARY KEY, model_q_ovr NUMERIC, madden_ovr NUMERIC, discrepancy NUMERIC, signal TEXT,
                    model_offense NUMERIC, madden_offense NUMERIC, model_defense NUMERIC, madden_defense NUMERIC,
                    model_pass_protection NUMERIC, madden_pass_protection NUMERIC, model_pass_rush NUMERIC,
                    madden_pass_rush NUMERIC, model_secondary NUMERIC, madden_secondary NUMERIC,
                    net_dropback_epa NUMERIC, net_rush_epa NUMERIC, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.execute(text("TRUNCATE TABLE nfl_team_ratings_comparison;"))
        df_ratings.to_sql("nfl_team_ratings_comparison", engine, if_exists="append", index=False, method="multi")
        logging.info("Team ratings comparisons successfully committed to PostgreSQL.")

if __name__ == "__main__":
    pipeline = QuantitativeRatingsPipeline(season=2026)
    pipeline.sync_data()
    df_eval = pipeline.calculate_q_ovr()
    pipeline.persist_to_database(df_eval)
