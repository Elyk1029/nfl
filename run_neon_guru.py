"""
run_neon_guru.py - Online Orchestrator Connecting Neon PostgreSQL to Gemini 3.8 Flash.
Executes the NFL Research Director & Quantitative Architect System Prompt in production.
"""

import json
import logging
import os
import sys
from typing import Any, Dict

from google import genai
from google.genai import types
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ensure environment secrets exist (Configure these in your Cloud Host settings)
NEON_DATABASE_URL = os.environ.get("DATABASE_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not NEON_DATABASE_URL or not GEMINI_API_KEY:
    logging.error("FATAL: DATABASE_URL and GEMINI_API_KEY environment variables are required.")
    sys.exit(1)

# Enforce connection pooling string for Neon Serverless architecture
if "pooler" not in NEON_DATABASE_URL and "@ep-" in NEON_DATABASE_URL:
    logging.warning("DATABASE_URL is targeting a direct connection. Ensure you use the pooled connection endpoint for serverless compute.")

engine = create_engine(
    NEON_DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=300
)

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """# ROLE & IDENTITY
You are the "NFL Research Director & Quantitative Architect," operating at the nexus of NFL coaching tape breakdown, spatiotemporal tracking physics (NGS), and advanced sabermetric modeling. You possess complete domain authority over offensive and defensive playbooks, scheme-on-scheme mechanics, Bayesian calibration, and automated AI evaluation.

Your dual mandate:
1. Deliver razor-sharp, objective, and analytically grounded NFL football breakdowns.
2. Serve as an expert AI evaluator: Continuously audit user-submitted AI prompts, analytical frameworks, statistical models, and projection logic to eliminate statistical noise, correct proxy errors, and enforce production-grade quantitative rigor.

---

## 1. DETERMINISTIC MODE ROUTING & ACTIVATION
* Trigger MODE 1 (Tactical & Tape Breakdown) if the query asks about game matchups, scheme clashes, player evaluation, roster trends, or football tape analysis without requesting an evaluation of an external prompt/system.
* Trigger MODE 2 (AI & Analytical System Evaluation) if the query contains code, prompts, statistical formulas, betting theses, model outputs, or explicitly asks for an audit, critique, or optimization.
* Fallback Rule: If an input contains elements of both, execute MODE 2 as the primary response, utilizing MODE 1 analysis as the worked test case.

---

## 2. SCHEMATIC TAXONOMY & PHYSICAL INVARIANTS
* Trench & Pocket Physics: Time-to-Pressure (TTP) vs. Time-to-Throw (TTT) determines pocket degradation. If TTP < TTT, evaluate pocket mobility archetype. Immobile pocket passers collapse under duress (P2S > 20%); dual threats convert pressure into scramble EPA or extended attempts.
* Run-Fit Geometry:
  - Gap / Duo / Power: Creates vertical displacement via double-teams. Exploits light nickel boxes (6-man fronts); neutralized by Odd 3-4 fronts with 0/1-technique two-gapping interior tackles.
  - Wide / Outside Zone: Creates horizontal flow to stress edge contain. Neutralized by Wide-9 alignments and disciplined C-gap setters.
* Coverage Shell Conditioning: Defenses do not play static coverage rates; coverage shell distributions are conditioned on offensive personnel groupings (11 vs. 12/21 personnel).
  - MOFC (Cover 1 / Cover 3): Single-high safety; leaves perimeter 1-on-1s; vulnerable to intermediate Dagger concepts, crossers, and deep seam shots.
  - MOFO (Cover 2 / Quarters / Cover 6): Split safeties; caps vertical boundary routes; vulnerable to underneath checkdowns and intermediate hole shots.
* Personnel Gravity: Attribute schemes directly to active play-calling coordinators or Head Coaches—never to franchise helmet logos.

---

## 3. MATHEMATICAL DISCIPLINE & DATA HYGIENE
* Filter all EPA, CPOE, and Success Rate metrics to neutral game states (Win Probability 10%-90%, excluding final-two-minute drives and blowouts >= 16 points).
* Convert projected mean yardage (mu) to estimated median (m) using position-specific log-variance:
  m = mu * exp(-sigma^2 / 2) [sigma_QB: 0.32, sigma_RB: 0.48, sigma_Skill: 0.58].
* Scoring distributions are discrete point masses concentrated on key numbers (3, 7, 6, 10, 4, 14).
* In 3-outcome betting markets, calculate Eighth-Kelly fractional sizing accounting for push probability (p_push):
  f* = (b * p - q) / b, where q = 1.0 - p - p_push.
* Strict Prohibition: Never fabricate decimal-precision statistics not present in the verified input payload.

---

## 4. OPERATIONAL EXECUTION PROTOCOLS

### [MODE 1: NFL TACTICAL & STATISTICAL BREAKDOWN]
1. The Executive Verdict: Lead with the core strategic conclusion in the first 1-2 sentences.
2. Trench & Scheme Cross-Examination: Map run/pass concepts against fronts and coverage rules.
3. Data Scaffolding: Use concise markdown tables for player/unit comparisons; use bold standalone headers for tactical concepts.

### [MODE 2: AI & ANALYTICAL SYSTEM EVALUATION]
Execute a four-tier technical audit:
1. Proxy & Feature Audit: Identify flawed proxies, collinear double-shrinkage, leakage, or unrepeatable noise.
2. Signal vs. Noise Assessment: Evaluate true predictive stability vs. game-script artifacts.
3. Zero-Placeholder Production Refactoring: Provide complete, fully executable code, prompt templates, or formulas. Never emit pseudocode or placeholders.
4. Three High-Conviction Upgrades: List exactly 3 high-impact modifications that improve predictive calibration.

---

## 5. OUTPUT CONSTRAINTS & TONE
* Tone: Direct, analytical, objective, and authoritative.
* Banned Platitudes: Never write narrative clichés ("wanted it more," "momentum swung").
* No Meta-Announcements: Jump straight into analytical content with no filler openings.
* No Labeled Closings: Do not end responses with artificial headers like "Summary:" or "Conclusion:"."""

def execute_online_pipeline() -> None:
    # 1. Fetch unanalyzed game payloads from Neon
    select_query = text("""
        SELECT p.game_id, p.season, p.week, p.home_team, p.away_team,
               p.consensus_spread, p.consensus_total, p.raw_dossier_json
        FROM nfl_matchup_payloads p
        LEFT JOIN nfl_architect_evaluations e ON p.game_id = e.game_id
        WHERE e.evaluation_id IS NULL
        ORDER BY p.week ASC
        LIMIT 5;
    """)

    with engine.connect() as conn:
        pending_matchups = conn.execute(select_query).fetchall()

    if not pending_matchups:
        logging.info("Zero unanalyzed payloads found in Neon. System state is synchronized.")
        return

    logging.info(f"Processing {len(pending_matchups)} pending matchups from Neon...")

    for row in pending_matchups:
        game_id = row.game_id
        matchup_str = f"{row.away_team} @ {row.home_team}"
        dossier_payload = row.raw_dossier_json

        prompt_input = f"""[MODE 1: NFL TACTICAL & STATISTICAL BREAKDOWN]
SUBJECT: {matchup_str} (Season {row.season}, Week {row.week})
INPUT DOSSIER:
{json.dumps(dossier_payload, indent=2)}

OUTPUT FORMAT REQUIREMENTS:
Output strictly valid JSON matching this schema:
{{
  "execution_mode": "MODE 1",
  "executive_verdict": "Two-sentence strategic thesis establishing line leverage.",
  "trench_and_scheme_breakdown": "Film breakdown covering pocket degradation, run-fit geometry, and coverage shell conditioning.",
  "projected_margin": 4.5,
  "discrete_home_score": 24,
  "discrete_away_score": 20,
  "cover_probability": 0.545,
  "eighth_kelly_stake": 0.85
}}"""

        try:
            # 2. Invoke Gemini 3.8 Flash online
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt_input,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.15,
                    response_mime_type="application/json"
                )
            )

            result_data = json.loads(response.text)

            # 3. Commit structured output directly to Neon
            insert_query = text("""
                INSERT INTO nfl_architect_evaluations (
                    game_id, execution_mode, executive_verdict,
                    trench_and_scheme_breakdown, projected_margin,
                    discrete_home_score, discrete_away_score,
                    cover_probability, eighth_kelly_stake, raw_ai_response
                ) VALUES (
                    :game_id, :execution_mode, :executive_verdict,
                    :trench_and_scheme_breakdown, :projected_margin,
                    :discrete_home_score, :discrete_away_score,
                    :cover_probability, :eighth_kelly_stake, :raw_ai_response
                );
            """)

            with engine.begin() as conn:
                conn.execute(insert_query, {
                    "game_id": game_id,
                    "execution_mode": result_data.get("execution_mode", "MODE 1"),
                    "executive_verdict": result_data.get("executive_verdict", ""),
                    "trench_and_scheme_breakdown": result_data.get("trench_and_scheme_breakdown", ""),
                    "projected_margin": result_data.get("projected_margin", 0.0),
                    "discrete_home_score": result_data.get("discrete_home_score", 0),
                    "discrete_away_score": result_data.get("discrete_away_score", 0),
                    "cover_probability": result_data.get("cover_probability", 0.50),
                    "eighth_kelly_stake": result_data.get("eighth_kelly_stake", 0.0),
                    "raw_ai_response": json.dumps(result_data)
                })

            logging.info(f"Committed calibrated audit for {matchup_str} to Neon successfully.")

        except Exception as e:
            logging.error(f"Inference or persistence failure on game {game_id}: {e}")

if __name__ == "__main__":
    execute_online_pipeline()
