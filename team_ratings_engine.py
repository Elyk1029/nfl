"""
team_ratings_engine.py - Excel-Integrated Database Overall & Roster Pipeline.
Parses Madden_27_Secondary_Weighted_Rankings.xlsx for Database Overalls, unit ratings,
and full player rosters to power the institutional terminal and AI research assistant.
"""

import logging
import os
import nflreadpy as nfl
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    raise ValueError("FATAL: DATABASE_URL must be configured in environment.")

engine = create_engine(DB_URL, pool_size=5, max_overflow=10, pool_pre_ping=True)

TEAM_ABBR_MAP = {
    "LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX"
}

SUMMARY_TEAM_TO_ABBR = {
    'Los Angeles Rams': 'LA', 'Baltimore Ravens': 'BAL', 'Detroit Lions': 'DET',
    'Buffalo Bills': 'BUF', 'New England Patriots': 'NE', 'Denver Broncos': 'DEN',
    'Philadelphia Eagles': 'PHI', 'Kansas City Chiefs': 'KC', 'San Francisco 49ers': 'SF',
    'Los Angeles Chargers': 'LAC', 'Dallas Cowboys': 'DAL', 'Cincinnati Bengals': 'CIN',
    'Tampa Bay Buccaneers': 'TB', 'Chicago Bears': 'CHI', 'Seattle Seahawks': 'SEA',
    'Houston Texans': 'HOU', 'Pittsburgh Steelers': 'PIT', 'Indianapolis Colts': 'IND',
    'Green Bay Packers': 'GB', 'Minnesota Vikings': 'MIN', 'Washington Commanders': 'WAS',
    'Las Vegas Raiders': 'LV', 'Atlanta Falcons': 'ATL', 'Jacksonville Jaguars': 'JAX',
    'NY Giants': 'NYG', 'Carolina Panthers': 'CAR', 'Arizona Cardinals': 'ARI',
    'New Orleans Saints': 'NO', 'Cleveland Browns': 'CLE', 'NY Jets': 'NYJ',
    'Tennessee Titans': 'TEN', 'Miami Dolphins': 'MIA'
}

ALL_32_TEAMS = list(SUMMARY_TEAM_TO_ABBR.values())

def clean_team_abbr(t: str) -> str:
    c = str(t).strip().upper()
    return TEAM_ABBR_MAP.get(c, c)

def init_ratings_schema(db_engine=engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS nfl_team_ratings_comparison (
        team TEXT PRIMARY KEY,
        model_q_ovr NUMERIC NOT NULL,
        madden_ovr NUMERIC NOT NULL,
        discrepancy NUMERIC NOT NULL,
        signal TEXT NOT NULL,
        model_offense NUMERIC NOT NULL,
        madden_offense NUMERIC NOT NULL,
        model_defense NUMERIC NOT NULL,
        madden_defense NUMERIC NOT NULL,
        model_pass_protection NUMERIC NOT NULL,
        madden_pass_protection NUMERIC NOT NULL,
        model_pass_rush NUMERIC NOT NULL,
        madden_pass_rush NUMERIC NOT NULL,
        model_secondary NUMERIC NOT NULL,
        madden_secondary NUMERIC NOT NULL,
        net_dropback_epa NUMERIC NOT NULL,
        net_rush_epa NUMERIC NOT NULL,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS nfl_team_rosters (
        player_id SERIAL PRIMARY KEY,
        team TEXT NOT NULL,
        player_name TEXT NOT NULL,
        position TEXT NOT NULL,
        overall_rating INTEGER NOT NULL,
        archetype TEXT,
        tier_status TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_nfl_team_ratings_q_ovr 
    ON nfl_team_ratings_comparison (model_q_ovr DESC);

    CREATE INDEX IF NOT EXISTS idx_nfl_rosters_team 
    ON nfl_team_rosters (team);
    """
    with db_engine.begin() as conn:
        conn.execute(text(ddl))

class QuantitativeRatingsPipeline:
    def __init__(self, season: int = 2026, excel_path: str = "Madden_27_Secondary_Weighted_Rankings.xlsx"):
        self.season = season
        self.excel_path = excel_path
        self.pbp_df = pd.DataFrame()
        self.summary_df = pd.DataFrame()
        self.roster_df = pd.DataFrame()
        init_ratings_schema(engine)

    def sync_data(self) -> None:
        logging.info("Syncing tracking data and parsing Secondary Weighted Rankings Excel file...")
        try:
            pbp = nfl.load_pbp(seasons=[self.season, self.season - 1]).to_pandas()
        except Exception:
            pbp = pd.DataFrame()

        for col in ["posteam", "defteam", "home_team", "away_team"]:
            if col in pbp.columns:
                pbp[col] = pbp[col].apply(clean_team_abbr)
        self.pbp_df = pbp

        if os.path.exists(self.excel_path):
            xls = pd.ExcelFile(self.excel_path)
            if 'Summary' in xls.sheet_names:
                self.summary_df = pd.read_excel(self.excel_path, sheet_name='Summary')

            team_rosters = []
            for sheet in xls.sheet_names:
                if sheet in ['Summary', 'All Teams Starters']:
                    continue
                df_team = pd.read_excel(self.excel_path, sheet_name=sheet)
                if 'firstName' in df_team.columns and 'lastName' in df_team.columns and 'overallRating' in df_team.columns:
                    abbr = SUMMARY_TEAM_TO_ABBR.get(sheet, clean_team_abbr(sheet))
                    for _, row in df_team.iterrows():
                        f_name = str(row.get('firstName', '')).strip()
                        l_name = str(row.get('lastName', '')).strip()
                        p_name = f"{f_name} {l_name}".strip()
                        pos = str(row.get('position/shortLabel', 'UNK')).strip().upper()
                        ovr = int(row.get('overallRating', 70))
                        arch = str(row.get('archetype/label', ''))
                        tier = str(row.get('Tier / Status', row.get('X-Factor', 'Starter')))
                        
                        team_rosters.append({
                            "team": abbr,
                            "player_name": p_name,
                            "position": pos,
                            "overall_rating": ovr,
                            "archetype": arch,
                            "tier_status": tier if tier != 'nan' else 'Starter'
                        })
            self.roster_df = pd.DataFrame(team_rosters)
        else:
            logging.warning(f"Excel file {self.excel_path} not found.")

    def calculate_q_ovr(self) -> pd.DataFrame:
        records = []
        pbp = self.pbp_df

        neutral = pd.DataFrame()
        if not pbp.empty and "home_wp" in pbp.columns:
            clean = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
            neutral = clean[
                (clean["home_wp"].between(0.10, 0.90)) &
                ~((clean.get("qtr", 1) == 4) & (clean.get("score_differential", 0).abs() >= 16))
            ].copy()

        # Build summary lookup
        summary_map = {}
        if not self.summary_df.empty:
            for _, r in self.summary_df.iterrows():
                t_name = str(r['Team']).strip()
                abbr = SUMMARY_TEAM_TO_ABBR.get(t_name, t_name)
                summary_map[abbr] = {
                    "ovr": float(r['Secondary-Boosted OVR']),
                    "off": float(r['Offense OVR']),
                    "def": float(r['Defense OVR'])
                }

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

            # Get Database Overall from Excel Summary
            t_sum = summary_map.get(team, {"ovr": 80.0, "off": 80.0, "def": 80.0})
            m_ovr, m_off, m_def = t_sum["ovr"], t_sum["off"], t_sum["def"]
            m_pass_pro = round(m_off - 2.0, 1)
            m_pass_rush = round(m_def - 1.0, 1)
            m_sec = round(m_def - 1.5, 1)

            discrepancy = round(q_overall - m_ovr, 1)

            if discrepancy >= 3.0:
                signal = "🔥 High Quant Upside (Database Undervalued)"
            elif discrepancy <= -3.0:
                signal = "❄️ Fragile Composite (Database Overrated)"
            else:
                signal = "⚖️ Market Efficient"

            records.append({
                "team": team,
                "model_q_ovr": q_overall,
                "madden_ovr": m_ovr, # Renamed in UI to Database Overall
                "discrepancy": discrepancy,
                "signal": signal,
                "model_offense": round(q_off, 1),
                "madden_offense": m_off,
                "model_defense": round(q_def, 1),
                "madden_defense": m_def,
                "model_pass_protection": round(q_pass_pro, 1),
                "madden_pass_protection": m_pass_pro,
                "model_pass_rush": round(q_pass_rush, 1),
                "madden_pass_rush": m_pass_rush,
                "model_secondary": round(q_sec, 1),
                "madden_secondary": m_sec,
                "net_dropback_epa": round(off_drop_epa, 3),
                "net_rush_epa": round(off_rush_epa, 3)
            })

        return pd.DataFrame(records).sort_values("model_q_ovr", ascending=False)

    def persist_to_database(self, df_ratings: pd.DataFrame) -> None:
        if df_ratings.empty:
            return
        init_ratings_schema(engine)
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE nfl_team_ratings_comparison RESTART IDENTITY CASCADE;"))
            conn.execute(text("TRUNCATE TABLE nfl_team_rosters RESTART IDENTITY CASCADE;"))
            
        df_ratings.to_sql("nfl_team_ratings_comparison", engine, if_exists="append", index=False, method="multi")
        if not self.roster_df.empty:
            self.roster_df.to_sql("nfl_team_rosters", engine, if_exists="append", index=False, method="multi")

if __name__ == "__main__":
    pipeline = QuantitativeRatingsPipeline(season=2026, excel_path="Madden_27_Secondary_Weighted_Rankings.xlsx")
    pipeline.sync_data()
    df_eval = pipeline.calculate_q_ovr()
    pipeline.persist_to_database(df_eval)
    print("Secondary Weighted Rankings and Roster Players successfully committed to Neon.")
