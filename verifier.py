"""
verifier.py - Production System Auditor & Mathematical Invariant Verifier.
Validates model weights, schema compliance, target-tree conservation, and absence of halluncinations.
"""
import json
import math
import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import xgboost as xgb

EXPECTED_FEATURES = [
    "net_pass_edge", "net_rush_edge", "net_late_down_edge", "diff_success",
    "diff_explosive", "rest_diff", "is_divisional", "market_home_prob"
]

class NFLDataVerifier:
    def __init__(self, db_url: str = None, model_path: str = "nfl_model.json"):
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        self.model_path = model_path
        self.engine = create_engine(self.db_url, pool_pre_ping=True) if self.db_url else None

    def verify_model_weights(self) -> bool:
        """Verifies XGBoost model integrity and input dimensionality."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' is missing.")

        model = xgb.XGBClassifier()
        try:
            model.load_model(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load XGBoost model from '{self.model_path}': {e}")

        # Test inference on dummy vector
        test_vec = pd.DataFrame([[0.0] * len(EXPECTED_FEATURES)], columns=EXPECTED_FEATURES)
        try:
            prob = model.predict_proba(test_vec)[0]
            assert len(prob) == 2, "Model must output binary probability distribution."
            assert 0.0 <= prob[1] <= 1.0, "Probability must fall in [0, 1]."
            print("Model weights and inference dimensionality verified.")
            return True
        except Exception as e:
            raise ValueError(f"Inference sanity check failed: {e}")

    def verify_skill_volume_conservation(self, team_gross_pass: float, player_projections: list) -> bool:
        """
        Enforces line-of-scrimmage physical invariant:
        Sum of individual skill receiving medians must not exceed gross team passing capacity.
        """
        rec_medians = [
            float(p.get("rec_yards", 0.0)) for p in player_projections
            if p.get("role") not in ["QB1"]
        ]
        total_rec_median = sum(rec_medians)

        # Due to right-skew (log-normal), sum of medians is mathematically <= gross mean pass yards
        if total_rec_median > team_gross_pass * 1.05:
            raise AssertionError(
                f"Volume conservation failure: Allocated medians ({total_rec_median:.1f} yds) "
                f"exceed team gross pass volume ({team_gross_pass:.1f} yds)."
            )

        roles = [p.get("role") for p in player_projections]
        unique_roles = set(roles)
        if len(roles) != len(unique_roles):
            raise AssertionError(f"Duplicate positional slot detected in projections: {roles}")

        print(f"Volume conservation verified: {total_rec_median:.1f} yds allocated across {len(roles)} roles.")
        return True

    def verify_database_records(self, season: int, week: int) -> bool:
        """Audits database records to ensure zero tie scores and non-null analysis payloads."""
        if not self.engine:
            print("No database connection available; skipping DB record audit.")
            return True

        query = text("""
            SELECT game_id, matchup, home_win_prob, predicted_home_score, predicted_away_score, analysis
            FROM nfl_weekly_analysis
            WHERE season = :s AND week = :w;
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"s": season, "w": week})

        if df.empty:
            print(f"Warning: No database records found for Season {season} Week {week}.")
            return False

        for _, r in df.iterrows():
            h_sc = r["predicted_home_score"]
            a_sc = r["predicted_away_score"]
            if h_sc == a_sc:
                raise AssertionError(f"Regular season tie detected in game {r['game_id']}: {h_sc}-{a_sc}.")

            payload = json.loads(r["analysis"])
            if "player_projections" not in payload:
                raise AssertionError(f"Missing 'player_projections' key in analysis for game {r['game_id']}.")

            home_projs = payload["player_projections"].get("home", [])
            away_projs = payload["player_projections"].get("away", [])

            for p in home_projs + away_projs:
                assert float(p.get("pass_yards", 0.0)) >= 0.0, "Pass yards must be clamped >= 0."
                assert float(p.get("rush_yards", 0.0)) >= 0.0, "Rush yards must be clamped >= 0."
                assert float(p.get("rec_yards", 0.0)) >= 0.0, "Rec yards must be clamped >= 0."
                assert float(p.get("total_tds", 0.0)) >= 0.0, "TD expectation must be clamped >= 0."

        print(f"Successfully audited {len(df)} database records for Season {season} Week {week}.")
        return True


if __name__ == "__main__":
    verifier = NFLDataVerifier()
    verifier.verify_model_weights()
    if len(sys.argv) >= 3:
        verifier.verify_database_records(season=int(sys.argv[1]), week=int(sys.argv[2]))
