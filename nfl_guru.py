import math
from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import norm

# Standard position-specific log-standard deviations for NFL player props
LOG_SIGMA_FACTORS: Dict[str, float] = {
    "QB_Pass": 0.32,
    "RB_Rush": 0.48,
    "WR_Rec": 0.58,
    "TE_Rec": 0.54,
}

# Empirical NFL regular-season margin probabilities (historical frequency distribution)
KEY_MARGIN_PROBABILITIES: Dict[int, float] = {
    3: 0.148,
    7: 0.094,
    6: 0.059,
    10: 0.057,
    4: 0.052,
    14: 0.046,
    1: 0.038,
    2: 0.036
}


class NFLQuantitativeEngine:
    """
    Production-grade quantitative framework for nfl_guru.
    Enforces mathematical hygiene, discrete scoring distributions, and log-normal median conversions.
    """

    @staticmethod
    def calculate_lognormal_median(mean_projection: float, position_group: str) -> float:
        """
        Converts expected mean yardage into an estimated median (50th percentile)
        to align with sportsbook prop line construction: m = mu * exp(-sigma^2 / 2).
        """
        if mean_projection <= 0:
            return 0.0
        sigma = LOG_SIGMA_FACTORS.get(position_group, 0.50)
        median_projection = mean_projection * math.exp(-(sigma ** 2) / 2.0)
        return round(median_projection, 2)

    @staticmethod
    def evaluate_spread_edge(
        projected_margin: float,
        market_spread: float,
        standard_deviation: float = 13.45
    ) -> Tuple[float, float, float]:
        """
        Calculates home cover, away cover, and push probabilities using
        empirical discrete point mass adjustments around key NFL numbers.
        Spread is expressed from the home team perspective (e.g., -3.5).
        """
        # Continuous z-score baseline
        z = (projected_margin - (-market_spread)) / standard_deviation
        raw_home_cover = float(norm.cdf(z))
        
        # Check if spread falls on a discrete integer push number
        abs_spread = round(abs(market_spread))
        is_integer_spread = float(market_spread).is_integer()
        
        if is_integer_spread and abs_spread in KEY_MARGIN_PROBABILITIES:
            p_push = KEY_MARGIN_PROBABILITIES[abs_spread]
        else:
            p_push = 0.0

        # Adjust continuous CDF across discrete push mass
        p_home_cover = raw_home_cover * (1.0 - p_push)
        p_away_cover = (1.0 - raw_home_cover) * (1.0 - p_push)

        return round(p_home_cover, 4), round(p_away_cover, 4), round(p_push, 4)

    @staticmethod
    def calculate_eighth_kelly(
        win_prob: float,
        push_prob: float,
        decimal_odds: float = 1.9091  # Standard -110 American odds
    ) -> float:
        """
        Calculates conservative Eighth-Kelly stake sizing accounting for push equity.
        f* = (b * p - q) / b, where q = 1.0 - p - p_push.
        """
        b = decimal_odds - 1.0
        q = 1.0 - win_prob - push_prob
        edge = (b * win_prob) - q

        if edge <= 0:
            return 0.0

        full_kelly = edge / b
        eighth_kelly = full_kelly * 0.125
        # Cap intra-game exposure at 2.5 units
        return round(min(max(eighth_kelly * 100.0, 0.0), 2.5), 2)


class NFLTacticalDossierBuilder:
    """
    Constructs leak-proof, highly structured input payloads for AI scouting models.
    Strictly forbids open-ended generation of unverifiable tracking metrics.
    """

    @staticmethod
    def build_matchup_payload(
        matchup_name: str,
        home_offense_personnel: Dict[str, float],
        away_defense_coverage_vs_personnel: Dict[str, float],
        trench_metrics: Dict[str, float],
        projected_game_script: Dict[str, Any]
    ) -> str:
        dossier = {
            "matchup": matchup_name,
            "tactical_context": {
                "home_11_personnel_rate": home_offense_personnel.get("11_rate", 0.0),
                "home_12_personnel_rate": home_offense_personnel.get("12_rate", 0.0),
                "away_coverage_shell_vs_11": {
                    "MOFO_Quarters_Cover6": away_defense_coverage_vs_personnel.get("mofo_rate_vs_11", 0.0),
                    "MOFC_Cover1_Cover3": away_defense_coverage_vs_personnel.get("mofc_rate_vs_11", 0.0)
                },
                "trench_clock": {
                    "offensive_TTT": trench_metrics.get("time_to_throw", 0.0),
                    "defensive_TTP": trench_metrics.get("time_to_pressure", 0.0),
                    "protection_delta": round(
                        trench_metrics.get("time_to_pressure", 0.0) - trench_metrics.get("time_to_throw", 0.0), 2
                    )
                }
            },
            "market_and_model_projections": projected_game_script
        }

        prompt = (
            f"SYSTEM INSTRUCTION: You are an NFL Director of Research analyzing coaching film.\n"
            f"RULES:\n"
            f"1. You must ONLY cite the concrete tracking metrics and rates provided in the dossier below.\n"
            f"2. Never fabricate decimal-precision statistics not present in the payload.\n"
            f"3. Frame pocket integrity as the exact Delta: TTP ({trench_metrics.get('time_to_pressure')}s) vs "
            f"TTT ({trench_metrics.get('time_to_throw')}s).\n"
            f"4. If protection_delta < 0, evaluate pocket collapse and checkdown rates; do not praise downfield progression.\n\n"
            f"DATA DOSSIER:\n{dossier}\n\n"
            f"DELIVERABLE:\n"
            f"Provide an executive tactical breakdown detailing run-fit geometry and coverage shell leverage."
        )
        return prompt
