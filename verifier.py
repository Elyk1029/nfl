"""
verifier.py - Production Fail-Fast Data Verifier & Stop-Block Airlock.
Decouples macro game-edge verification from micro player-prop ingestion to eliminate false stop blocks.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NFLDataVerifier")


@dataclass(frozen=True)
class VerificationResult:
    is_valid: bool
    status: str
    error_code: Optional[str]
    audit_log: List[str]
    raw_payload: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "DATA_INTEGRITY_STATUS": self.status,
                "EXECUTION_STATE": "PROCEED" if self.is_valid else "HALTED_STOP_BLOCK",
                "ERROR_CODE": self.error_code,
                "AUDIT_LOG": self.audit_log,
            },
            indent=2,
        )


class NFLDataVerifier:

    @staticmethod
    def verify_team_target_tree(
        team_abbr: str,
        total_gross_pass_yds: float,
        team_projections: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        team_rec_yds = sum(
            float(p.get("projected_value", 0.0))
            for p in team_projections
            if "Rec" in p.get("prop_category", "")
        )

        if total_gross_pass_yds <= 0.0:
            return True, f"[{team_abbr}] Pass volume is 0 or unprojected."

        ratio = team_rec_yds / total_gross_pass_yds
        # Tolerate [70.0%, 130.0%] band to accommodate unscripted backup snaps and edge scheme variance
        if not (0.70 <= ratio <= 1.30):
            return (
                False,
                f"[{team_abbr}] Target Tree Alert: Receiving Yards ({team_rec_yds:.1f}) "
                f"is {ratio:.1%} of Gross Pass Volume ({total_gross_pass_yds:.1f}). Expected [70.0%, 130.0%].",
            )
        return True, f"[{team_abbr}] Target tree reconciled at {ratio:.1%} of passing volume."

    @staticmethod
    def verify_metric_bounds(tape_metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        violations = []
        bounds = {
            "early_down_success_diff": (-0.45, 0.45),
            "explosive_rate_diff": (-0.35, 0.35),
            "net_pass_epa_diff": (-0.90, 0.90),
            "net_rush_epa_diff": (-0.70, 0.70),
        }

        for metric, (low, high) in bounds.items():
            if metric in tape_metrics:
                try:
                    val = float(tape_metrics[metric])
                    if not (low <= val <= high):
                        violations.append(
                            f"Out-of-Bounds Metric: '{metric}' = {val}. Expected [{low}, {high}]."
                        )
                except (ValueError, TypeError):
                    violations.append(
                        f"Type Mismatch: '{metric}' must be a valid float. Found: {tape_metrics[metric]}."
                    )

        return len(violations) == 0, violations

    @staticmethod
    def verify_market_probability(market_prob: float) -> Tuple[bool, str]:
        if not (0.01 <= market_prob <= 0.99):
            return (
                False,
                f"Market probability '{market_prob:.3f}' violates sanity threshold [0.01, 0.99].",
            )
        return True, "Market probability clean."

    @classmethod
    def audit_slate_payload(
        cls, payload: Dict[str, Any], parsed_analysis: Dict[str, Any]
    ) -> VerificationResult:
        audit_trail = []
        fatal_violations = []

        # 1. Macro Market Verification (Fatal Gate)
        market_prob = float(payload.get("market_prob", 0.5))
        mkt_valid, mkt_msg = cls.verify_market_probability(market_prob)
        if not mkt_valid:
            fatal_violations.append(mkt_msg)
        else:
            audit_trail.append(mkt_msg)

        # 2. Metric Boundary Verification (Fatal Gate)
        tape_metrics = payload.get("tape_metrics", {})
        bounds_valid, bound_msgs = cls.verify_metric_bounds(tape_metrics)
        if not bounds_valid:
            fatal_violations.extend(bound_msgs)
        else:
            audit_trail.append("All tape-metric differentials verified within bounds.")

        # 3. Micro Prop Invariants (Non-Fatal Gate)
        projections = parsed_analysis.get("player_projections", [])
        if not projections or not isinstance(projections, list):
            audit_trail.append("Warning: Empty player projections array. Macro spread edge preserved.")
        else:
            distinct_teams = list(
                set(p.get("team", "").strip().upper() for p in projections if p.get("team"))
            )
            for t in distinct_teams:
                team_props = [p for p in projections if p.get("team", "").strip().upper() == t]
                team_pass_yds = sum(
                    float(p.get("projected_value", 0.0))
                    for p in team_props
                    if "QB" in p.get("role", "") and "Pass" in p.get("prop_category", "")
                )

                if team_pass_yds > 0.0:
                    tt_valid, tt_msg = cls.verify_team_target_tree(t, team_pass_yds, team_props)
                    audit_trail.append(tt_msg)

        # Stop-block triggers solely on corrupted macro inputs
        if fatal_violations:
            logger.error(f"[STOP BLOCK ENGAGED] {payload.get('matchup')} failed macro verification.")
            return VerificationResult(
                is_valid=False,
                status="FAILED",
                error_code="ERR_DATA_INTEGRITY_BREACH",
                audit_log=fatal_violations,
                raw_payload=payload,
            )

        return VerificationResult(
            is_valid=True,
            status="PASSED",
            error_code=None,
            audit_log=audit_trail,
            raw_payload=payload,
        )
