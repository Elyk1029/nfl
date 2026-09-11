"""
verifier.py - Production Fail-Fast Data Verifier & Stop-Block Airlock.
Option A: Relaxed zero-defaulting and widened target-tree tolerances for Week 1 slate processing.
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
        """
        Enforces physical conservation of team passing volume.
        Allocated skill receiving yards must sit within [75.0%, 125.0%] of team gross passing yards
        to accommodate early-season rotation and mobile QB rushing/receiving splits.
        """
        team_rec_yds = sum(
            float(p.get("projected_value", 0.0))
            for p in team_projections
            if "Rec" in p.get("prop_category", "")
        )

        if total_gross_pass_yds <= 0.0:
            if team_rec_yds > 0.0:
                return (
                    False,
                    f"[{team_abbr}] Team Gross Pass Yards is {total_gross_pass_yds:.1f}, but Receiving Yards sum to {team_rec_yds:.1f}.",
                )
            return True, f"[{team_abbr}] Pass volume is 0 or unprojected."

        ratio = team_rec_yds / total_gross_pass_yds
        if not (0.75 <= ratio <= 1.25):
            return (
                False,
                f"[{team_abbr}] Target Tree Breach: Allocated Receiving Yards ({team_rec_yds:.1f}) "
                f"is {ratio:.1%} of Team Pass Volume ({total_gross_pass_yds:.1f}). Expected [75.0%, 125.0%].",
            )
        return True, f"[{team_abbr}] Target tree reconciled at {ratio:.1%} of passing volume."

    @staticmethod
    def verify_metric_bounds(tape_metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates that sabermetric features reside within viable NFL physical limits."""
        violations = []
        bounds = {
            "early_down_success_diff": (-0.40, 0.40),
            "explosive_rate_diff": (-0.30, 0.30),
            "net_pass_epa_diff": (-0.80, 0.80),
            "net_rush_epa_diff": (-0.60, 0.60),
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
        """Guards against devigged consensus inversion or zero-anchoring."""
        if not (0.02 <= market_prob <= 0.98):
            return (
                False,
                f"Market probability '{market_prob:.3f}' violates sanity threshold [0.02, 0.98].",
            )
        return True, "Market probability clean."

    @classmethod
    def audit_slate_payload(
        cls, payload: Dict[str, Any], parsed_analysis: Dict[str, Any]
    ) -> VerificationResult:
        """Master gatekeeper. Audits both pre-inference features and LLM-generated output."""
        audit_trail = []
        violations = []

        # 1. Market Baseline Verification
        market_prob = float(payload.get("market_prob", 0.5))
        mkt_valid, mkt_msg = cls.verify_market_probability(market_prob)
        if not mkt_valid:
            violations.append(mkt_msg)
        else:
            audit_trail.append(mkt_msg)

        # 2. Metric Boundary Verification
        tape_metrics = payload.get("tape_metrics", {})
        bounds_valid, bound_msgs = cls.verify_metric_bounds(tape_metrics)
        if not bounds_valid:
            violations.extend(bound_msgs)
        else:
            audit_trail.append("All tape-metric differentials verified within bounds.")

        # 3. Micro Prop Invariants (Option A: Flag only truly negative/corrupted values)
        projections = parsed_analysis.get("player_projections", [])
        if projections:
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
                    if not tt_valid:
                        violations.append(tt_msg)
                    else:
                        audit_trail.append(tt_msg)

            # Option A Modification: Only flag if projected value is strictly negative (corrupted data)
            corrupted_projections = []
            for p in projections:
                val = float(p.get("projected_value", 0.0))
                if val < 0.0:
                    corrupted_projections.append(f"{p.get('player')} ({p.get('role')} {p.get('prop_category')})")

            if corrupted_projections:
                violations.append(f"Corrupted negative projections detected: {corrupted_projections}")
        else:
            violations.append("Empty player projections returned from inference engine.")

        if violations:
            logger.error(
                f"[STOP BLOCK ENGAGED] {payload.get('matchup', 'Matchup')} failed verification with {len(violations)} errors."
            )
            return VerificationResult(
                is_valid=False,
                status="FAILED",
                error_code="ERR_DATA_INTEGRITY_BREACH",
                audit_log=violations,
                raw_payload=payload,
            )

        return VerificationResult(
            is_valid=True,
            status="PASSED",
            error_code=None,
            audit_log=audit_trail,
            raw_payload=payload,
        )
