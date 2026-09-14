"""
evaluate_and_store_performance.py - Automated Post-Slate Model Evaluation & Residual Storage.
Computes Brier scores, Score MAE, and ATS hit rates against PostgreSQL predictions.
"""
import logging
import os
import sys
from typing import Any, Dict, List
import nflreadpy as nfl
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

db_url = os.environ.get("DATABASE_URL")
if not db_url:
    raise ValueError("FATAL: DATABASE_URL must be configured.")

engine = create_engine(db_url, pool_size=5, max_overflow=10, pool_pre_ping=True)
MIN_BETTABLE_EDGE_PCT = 1.8

TEAM_ABBR_MAP = {"LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX"}
def clean_team_abbr(t: str) -> str:
    return TEAM_ABBR_MAP.get(str(t).strip().upper(), str(t).strip().upper())

def evaluate_week_performance(season: int = 2026, week: int = 1) -> None:
    logging.info(f"Executing calibration audit for Season {season} Week {week}...")
    try:
        schedules = nfl.load_schedules(seasons=[season]).to_pandas()
    except Exception as e:
        logging.error(f"Failed to load schedules: {e}")
        sys.exit(1)

    for col in ["home_team", "away_team"]:
        if col in schedules.columns:
            schedules[col] = schedules[col].apply(clean_team_abbr)

    completed = schedules[(schedules["week"] == week) & schedules["home_score"].notna() & schedules["away_score"].notna()].copy()
    if completed.empty:
        logging.warning(f"No completed games for Season {season} Week {week}.")
        sys.exit(0)

    query = text("SELECT * FROM nfl_weekly_analysis WHERE season = :s AND week = :w;")
    with engine.connect() as conn:
        model_df = pd.read_sql(query, conn, params={"s": season, "w": week})

    if model_df.empty:
        logging.error("No model predictions found in nfl_weekly_analysis for target week.")
        sys.exit(1)

    audit_records: List[Dict[str, Any]] = []

    for _, actual in completed.iterrows():
        matchup_str = f"{actual['away_team']} @ {actual['home_team']}"
        pred_row = model_df[model_df["matchup"] == matchup_str]
        if pred_row.empty:
            continue
        pred = pred_row.iloc[0]

        actual_home_score = int(actual["home_score"])
        actual_away_score = int(actual["away_score"])
        actual_margin = actual_home_score - actual_away_score
        actual_total = actual_home_score + actual_away_score

        market_spread = float(actual.get("spread_line", 0.0) or 0.0)
        market_total = float(actual.get("total_line", 44.0) or 44.0)

        pred_home = int(pred["predicted_home_score"])
        pred_away = int(pred["predicted_away_score"])
        pred_margin = pred_home - pred_away
        pred_total = int(pred["predicted_total_score"])

        home_win_prob = float(pred["home_win_prob"])
        cover_prob = float(pred["spread_cover_prob"])
        spread_edge = float(pred["spread_edge"])
        kelly_units = float(pred["kelly_units"])

        actual_home_won = 1.0 if actual_margin > 0 else (0.5 if actual_margin == 0 else 0.0)
        pred_home_won = 1 if pred_margin > 0 else 0
        su_hit = 1 if actual_home_won == pred_home_won else 0
        brier_score = round((home_win_prob - actual_home_won) ** 2, 4)

        actual_home_covered = actual_margin > market_spread
        is_push = actual_margin == market_spread

        if is_push:
            ats_status, ats_hit = "PUSH", None
        elif spread_edge >= (MIN_BETTABLE_EDGE_PCT / 100.0) and kelly_units > 0.0:
            rec_home = cover_prob > 0.50
            bet_won = (rec_home and actual_home_covered) or (not rec_home and not actual_home_covered)
            ats_status, ats_hit = ("WIN", 1) if bet_won else ("LOSS", 0)
        else:
            ats_status, ats_hit = "NO_BET", None

        audit_records.append({
            "game_id": str(pred["game_id"]), "season": season, "week": week,
            "matchup": matchup_str, "market_spread": market_spread, "market_total": market_total,
            "actual_home_score": actual_home_score, "actual_away_score": actual_away_score,
            "actual_margin": actual_margin, "actual_total": actual_total,
            "predicted_home_score": pred_home, "predicted_away_score": pred_away,
            "predicted_margin": pred_margin, "predicted_total": pred_total,
            "home_win_prob": home_win_prob, "brier_score": brier_score, "su_hit": su_hit,
            "ats_status": ats_status, "ats_hit": ats_hit,
            "score_mae": round((abs(pred_home - actual_home_score) + abs(pred_away - actual_away_score)) / 2.0, 2),
            "margin_error": round(abs(pred_margin - actual_margin), 2),
            "total_error": round(abs(pred_total - actual_total), 2),
            "kelly_units": kelly_units, "spread_edge": spread_edge
        })

    if not audit_records:
        return

    audit_df = pd.DataFrame(audit_records)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS nfl_model_performance_ledger (
                game_id TEXT PRIMARY KEY, season INTEGER, week INTEGER, matchup TEXT,
                market_spread NUMERIC, market_total NUMERIC, actual_home_score INTEGER,
                actual_away_score INTEGER, actual_margin INTEGER, actual_total INTEGER,
                predicted_home_score INTEGER, predicted_away_score INTEGER, predicted_margin INTEGER,
                predicted_total INTEGER, home_win_prob NUMERIC, brier_score NUMERIC, su_hit INTEGER,
                ats_status TEXT, ats_hit INTEGER, score_mae NUMERIC, margin_error NUMERIC,
                total_error NUMERIC, kelly_units NUMERIC, spread_edge NUMERIC,
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("DELETE FROM nfl_model_performance_ledger WHERE season = :s AND week = :w;"), {"s": season, "w": week})

    audit_df.to_sql("nfl_model_performance_ledger", engine, if_exists="append", index=False, method="multi")
    logging.info(f"Audit completed: {len(audit_df)} games logged to PostgreSQL.")

if __name__ == "__main__":
    evaluate_week_performance(season=2026, week=1)
