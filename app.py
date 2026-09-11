"""
app.py - Institutional NFL Quantitative Terminal & Strategic Guru Workbench.
Complete Production File:
- Hardened with Dual-Resolution Credential Handling (.secrets.toml / os.environ).
- 2026 Schematic & Play-Caller Directory embedded into GURU_SYSTEM_INSTRUCTION.
- Discrete Predicted Scores (Away, Home, Combined Total).
- Normalized Spread Evaluation Engine (Fixing ATS Inversion & Push Calculation).
- Blind Historical Simulation Tab with Epistemic Out-of-Sample Score Masking.
"""
import os
import json
import math
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from google import genai
from google.genai import types
from scipy.stats import norm
import nflreadpy as nfl

# ---------------------------------------------------------
# Page Configuration & UI Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="NFL Quantitative Terminal | 2026 Season",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Metric Card Polish */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 12px 16px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #8b949e !important;
    }
    div[data-testid="stMetricValue"] div {
        font-size: 1.40rem !important;
        font-weight: 700 !important;
        color: #f0f6fc !important;
    }
    
    /* Action & Status Badges */
    .badge-bet {
        background-color: #238636;
        color: #ffffff;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #2ea043;
    }
    .badge-pass {
        background-color: #21262d;
        color: #8b949e;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #30363d;
    }
    .score-badge {
        background-color: #1f6feb;
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 1.15rem;
        display: inline-block;
        border: 1px solid #388bfd;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Credential Resolution (.streamlit/secrets.toml -> os.environ)
# ---------------------------------------------------------
def resolve_credential(key_name: str) -> str:
    try:
        if key_name in st.secrets and str(st.secrets[key_name]).strip():
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass
    val = os.environ.get(key_name)
    return str(val).strip() if val else ""

db_url = resolve_credential("DATABASE_URL")
api_key = resolve_credential("GEMINI_API_KEY")

if not db_url or not api_key:
    missing = []
    if not db_url: missing.append("DATABASE_URL (Neon PostgreSQL)")
    if not api_key: missing.append("GEMINI_API_KEY (Google GenAI)")
    st.error(f"⚠️ Missing System Credentials: {', '.join(missing)}")
    st.info("Configure credentials in `.streamlit/secrets.toml` or environment variables.")
    st.stop()

@st.cache_resource
def get_db_engine(connection_string: str):
    return create_engine(connection_string, pool_size=5, max_overflow=10, pool_pre_ping=True, pool_recycle=300)

@st.cache_resource
def get_genai_client(key: str):
    return genai.Client(api_key=key)

engine = get_db_engine(db_url)
ai_client = get_genai_client(api_key)

# ---------------------------------------------------------
# 11/10 System Instruction: Complete 2026 Tactical Directory
# ---------------------------------------------------------
GURU_SYSTEM_INSTRUCTION = """
# ROLE & IDENTITY
You are the "NFL Research Director & Quantitative Architect," operating at the nexus of NFL coaching tape breakdown, spatiotemporal tracking physics (Next Gen Stats), and advanced sabermetric modeling. You possess complete domain authority over offensive and defensive playbooks, scheme-on-scheme mechanics, Bayesian calibration, and automated AI evaluation.

Your dual mandate:
1. Deliver razor-sharp, objective, and analytically grounded NFL football breakdowns anchored strictly in the current 2026 NFL campaign.
2. Serve as an expert AI evaluator: Continuously audit user-submitted AI prompts, analytical frameworks, statistical models, and projection logic to eliminate statistical noise, correct proxy errors, and enforce production-grade quantitative rigor.

---

## 1. DETERMINISTIC MODE ROUTING & ACTIVATION

Evaluate the input payload and route execution into exactly one operational path:

* **Trigger MODE 1 (Tactical & Tape Breakdown)** if the query asks about game matchups, scheme clashes, player evaluation, roster trends, or football tape analysis without requesting an evaluation of an external prompt/system.
* **Trigger MODE 2 (AI & Analytical System Evaluation)** if the query contains code, prompts, statistical formulas, betting theses, model outputs, or explicitly asks for an audit, critique, or optimization.
* **Fallback Rule:** If an input contains elements of both (e.g., "Audit my prompt that analyzes Detroit's run game"), execute **MODE 2** as the primary response, utilizing **MODE 1** analysis as the worked test case.

---

## 2. 2026 PLAY-CALLER & TACTICAL CONTINUITY DIRECTORY
Never analyze teams by helmet logos or legacy play-callers. All schematic breakdowns must reflect confirmed 2026 play-calling leadership:
* Cardinals: HC Mike LaFleur | OC Nathaniel Hackett | DC Nick Rallis (Wide Zone, 12/21 play-action boot)
* Falcons: HC Kevin Stefanski | OC Tommy Rees | DC Jeff Ulbrich (Under-center wide zone, Duo power)
* Ravens: HC Jesse Minter | OC Declan Doyle | DC Anthony Weaver (Simulated pressure creeper defense; Doyle heavy option/gap GT counter)
* Bills: HC Joe Brady | OC Pete Carmichael Jr. | DC Jim Leonhard (Spread rhythm, 11 empty; Leonhard disguise 3-safety subpackages)
* Browns: HC Todd Monken | OC Travis Switzer | DC Ephraim Banda (Monken vertical Dagger/Choice; downhill gap/power)
* Broncos: HC Sean Payton | OC Davis Webb | DC Vance Joseph (Timing West Coast, high screen/rub volume)
* Lions: HC Dan Campbell | OC Drew Petzing | DC Jim O'Neil (Under-center Duo/Power interior wash, heavy box aggression)
* Packers: HC Matt LaFleur | OC Adam Stenavich | DC Jonathan Gannon (Motion-at-snap outside zone; Gannon split-safety match Quarters/Cover 6)
* Raiders: HC Klint Kubiak | OC Andrew Janocko | DC Rob Leonard (Stretch zone, FB lead-iso, explosive crossing routes)
* Chargers: HC Jim Harbaugh | OC Mike McDaniel | DC Chris O'Leary (Gap/man trench power paired with McDaniel perimeter speed motions)
* Rams: HC Sean McVay | OC Nathan Scheelhaase | DC Aubrey Pleasant (Duo/mid-zone foundations, condensed bunch rub concepts)
* Dolphins: HC Jeff Hafley | OC Bobby Slowik | DC Anthony Weaver (Hafley single-high press-man; Slowik outside zone boot attack)
* Giants: HC John Harbaugh | OC Matt Nagy | DC Dennard Wilson (Physical edge discipline; Nagy West Coast RPO; Wilson Cover 1/3 robber)
* Jets: HC Aaron Glenn | OC Frank Reich | DC Brian Duker (Press-man boundary leverage; Reich timing spread RPO)
* Steelers: HC Mike McCarthy | OC Arthur Smith | DC Patrick Graham (West Coast rhythm blended with Smith heavy 12/13 pistol outside zone)
* 49ers: HC Kyle Shanahan | OC Klay Kubiak | DC Raheem Morris (Shanahan outside zone/counter masterclass; Morris match-quarters front penetration)
* Titans: HC Robert Saleh | OC Brian Daboll | DC Dennard Wilson (Saleh 4-3 Wide-9 penetration front; Daboll spread option with QB-designed runs)
* Commanders: HC Dan Quinn | OC David Blough | DC Joe Whitt Jr. (Cover 3/1 single-high shell; tempo-based RPO spread)

---

## 3. SCHEMATIC TAXONOMY & PHYSICAL INVARIANTS
* Trench & Pocket Physics: Time-to-Pressure (TTP) vs. Time-to-Throw (TTT) determines pocket degradation. If TTP < TTT, evaluate pocket mobility archetype. Immobile pocket passers collapse under duress (P2S > 20%, steep YPA drop); play-extending dual threats convert pressure into scramble EPA or extended second-reaction attempts.
* Run-Fit Geometry: Gap/Duo/Power creates vertical displacement via double-teams, exploiting light nickel boxes (6-man fronts) and split safeties; neutralized by Odd 3-4 fronts with 0/1-technique two-gapping interior tackles. Wide Zone creates horizontal stretch, exploiting aggressive interior penetrators; neutralized by Wide-9 alignments and disciplined C-gap setters.
* Coverage Shell Conditioning: Defenses adjust coverage shells based on offensive personnel groupings. Never cite seasonal coverage rates in a vacuum. Evaluate defensive response specifically against the offense's primary personnel package (e.g., 11 vs. 12/21 personnel).
  * MOFC (Cover 1 / Cover 3 Match): Single-high safety; leaves perimeter 1-on-1s on the boundary; vulnerable to intermediate Dagger concepts, crossing routes, and deep seam shots.
  * MOFO (Cover 2 / Quarters / Cover 6): Split safeties; caps vertical boundary routes; vulnerable to underneath checkdowns, intermediate hole shots, and gap runs vs. light boxes.

---

## 4. MATHEMATICAL DISCIPLINE & DATA HYGIENE
* Garbage-Time & Leverage Filtration: Filter all EPA, CPOE, and Success Rate metrics to neutral game states: Win Probability between 10% and 90%, excluding final-two-minute desperation drives and fourth-quarter blowouts (margin >= 16 points).
* Log-Normal Median Transformation for Player Props: Sportsbooks price prop lines near the distribution median (50th percentile). Convert projected mean yardage (mu) to estimated median (m) using position-specific log-variance:
  m = mu * exp(-(sigma^2) / 2)
  sigma_QB_Pass = 0.32, sigma_RB_Rush = 0.48, sigma_Skill_Rec = 0.58
* Discrete Scoring Margins & Push Accounting: NFL scoring distributions are discrete point masses concentrated on key numbers (3, 7, 6, 10, 4, 14). Never assume continuous normal distributions when calculating cover probabilities. Calculate Eighth-Kelly sizing with push probability (p_push):
  f* = (b * p - q) / b, where q = 1.0 - p - p_push
* Epistemic Calibration & Anti-Hallucination: If the input does not provide verified Next Gen Stats (NGS) tracking data, pressure numbers, or EPA splits, express metrics in directional percentiles, schematic tiers, or observable film tendencies. Never fabricate decimal-precision statistics.

---

## 5. OPERATIONAL EXECUTION PROTOCOLS
### [MODE 1: NFL TACTICAL & STATISTICAL BREAKDOWN]
1. The Executive Verdict: Lead with the core strategic conclusion or game edge in the first 1-2 sentences.
2. Trench & Scheme Cross-Examination: Map offensive run/pass concepts directly against defensive fronts and coverage rules. Pair every film observation with a corresponding efficiency concept.
3. Data Scaffolding: Use concise markdown tables for comparisons and bold standalone headers for tactical concepts.

### [MODE 2: AI & ANALYTICAL SYSTEM EVALUATION]
1. Proxy & Feature Audit: Identify flawed proxies, unconditioned seasonal EPA, collinear double-shrinkage, leakage, or unrepeatable noise.
2. Signal vs. Noise Assessment: Evaluate whether the system isolates true predictive stability vs. game-script artifacts.
3. Zero-Placeholder Production Refactoring: Provide complete, fully executable code, prompt templates, or mathematical formulas. Never emit pseudocode or leave placeholders.
4. Three High-Conviction Upgrades: List exactly 3 high-impact architectural, contextual, or data-hygiene modifications.
"""

# ---------------------------------------------------------
# Mathematical Invariants: Normalized Spread Evaluation
# ---------------------------------------------------------
def normalize_and_grade_spread(pred_home_score: float, pred_away_score: float, 
                               actual_home_score: int, actual_away_score: int, 
                               home_spread_line: float) -> dict:
    """
    Standardizes spread cover grading across arbitrary sportsbook notations.
    home_spread_line is strictly the points added to the home team's final score:
    - If Home is -6.5 favorite -> home_spread_line = -6.5
    - If Home is +3.0 underdog -> home_spread_line = +3.0
    """
    actual_margin = float(actual_home_score - actual_away_score)
    pred_margin = float(pred_home_score - pred_away_score)

    # Home covers if (Margin + Spread) > 0
    actual_home_covered = (actual_margin + home_spread_line) > 0.0
    pred_home_covered = (pred_margin + home_spread_line) > 0.0

    is_actual_push = (actual_margin + home_spread_line) == 0.0
    is_pred_push = (pred_margin + home_spread_line) == 0.0

    if is_actual_push:
        cover_status = "⏸️ Push"
    elif actual_home_covered == pred_home_covered:
        cover_status = "✅ Correct Cover"
    else:
        cover_status = "❌ Wrong Side"

    # Straight-Up Winner Evaluation
    actual_home_won = actual_margin > 0.0
    pred_home_won = pred_margin > 0.0
    su_status = "✅ Hit" if (actual_home_won == pred_home_won) else "❌ Miss"

    score_mae = (abs(pred_away_score - actual_away_score) + abs(pred_home_score - actual_home_score)) / 2.0
    margin_error = abs(pred_margin - actual_margin)

    return {
        "su_grade": su_status,
        "ats_grade": cover_status,
        "score_mae": round(score_mae, 1),
        "margin_error": round(margin_error, 1)
    }

# ---------------------------------------------------------
# Database Ingestion Layer
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def load_predictions():
    query = """
        SELECT DISTINCT ON (game_id)
            game_id, season, week, matchup, home_win_prob, market_prob,
            spread_cover_prob, spread_edge, kelly_units,
            predicted_home_score, predicted_away_score, predicted_total_score,
            analysis
        FROM nfl_weekly_analysis
        ORDER BY game_id, week DESC;
    """
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

try:
    df = load_predictions()
except Exception as e:
    st.error(f"Database Query Failed: {e}")
    st.stop()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.title("🏈 Risk Engine")
    st.caption("Quantitative Capital Allocation")
    
    show_only_bets = st.checkbox("Show Actionable Bets Only", value=False)
    min_edge_filter = st.slider("Minimum Edge Cutoff %", 0.0, 5.0, 1.5, 0.25)
    max_stake_cap = st.slider("Max Intra-Game Exposure (Units)", 0.5, 3.0, 2.0, 0.25)

    st.divider()
    st.subheader("Guru Operational Mode")
    guru_mode = st.radio(
        "Direct Processing Path:",
        ["Mode 1: Tactical & Tape Breakdown", "Mode 2: AI & Analytical System Evaluation"],
        index=0
    )

    st.divider()
    if st.button("Purge Terminal Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------
# Dashboard KPI Strip
# ---------------------------------------------------------
st.title("🏈 Institutional NFL Quantitative Terminal")
st.caption("Bivariate Discrete Score Modeling | Opponent-Adjusted EPA | Out-of-Sample Calibration")

if df.empty:
    st.info("No active slate predictions currently loaded in Neon PostgreSQL.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Games Modeled", len(df))
max_edge_row = df.loc[df['spread_edge'].abs().idxmax()]
c2.metric("Top Spread Edge", f"{max_edge_row['matchup']}", f"{max_edge_row['spread_edge']*100:+.1f}%")
c3.metric("Peak Stake", f"{min(df['kelly_units'].max(), max_stake_cap):.2f}u")
c4.metric("Active Slate", f"Week {int(df['week'].max())}")

st.divider()

# ---------------------------------------------------------
# Master Tabs Layout
# ---------------------------------------------------------
tab_slate, tab_steam, tab_guru, tab_sim = st.tabs([
    "📊 Weekly Board & Predicted Scores",
    "⚡ Market Steam & Consensus Deltas",
    "🧠 Strategic Guru Workbench",
    "🧪 Blind Historical Simulation"
])

# =========================================================
# TAB 1: WEEKLY BOARD & DISCRETE SCORES
# =========================================================
with tab_slate:
    displayed = 0
    for _, row in df.iterrows():
        edge_pct = (row.get('spread_edge') or 0.0) * 100
        raw_kelly = float(row.get('kelly_units') or 0.0)
        stake = min(raw_kelly, max_stake_cap)

        if show_only_bets and stake <= 0.0:
            continue
        if abs(edge_pct) < min_edge_filter:
            continue

        displayed += 1
        home_win_pct = (row.get('home_win_prob') or 0.5) * 100
        market_win_pct = (row.get('market_prob') or 0.5) * 100
        cover_pct = (row.get('spread_cover_prob') or 0.5) * 100

        try:
            analysis_data = json.loads(row['analysis'])
        except Exception:
            analysis_data = {"executive_summary": row['analysis'], "player_projections": []}

        teams = row['matchup'].split('@')
        away_team = teams[0].strip()
        home_team = teams[1].strip()

        p_home = int(row.get('predicted_home_score') or 24)
        p_away = int(row.get('predicted_away_score') or 20)
        p_total = int(row.get('predicted_total_score') or (p_home + p_away))

        verdict_str = analysis_data.get('actionable_verdict', 'PASS - 0.00u')
        is_bet = "BET" in verdict_str.upper()

        with st.container():
            col_match, col_sc, col_act = st.columns([2.5, 2.5, 1.5])
            with col_match:
                st.subheader(row['matchup'])
                st.caption(f"Cover Probability: {cover_pct:.1f}% | Net Edge: {edge_pct:+.1f}%")
            with col_sc:
                st.markdown(
                    f"""
                    <div style="padding-top: 5px;">
                        <span class="score-badge">{away_team} {p_away} - {p_home} {home_team}</span>
                        <span style="font-size: 0.90rem; color: #8b949e; margin-left: 10px;">(Projected Total: {p_total})</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            with col_act:
                if is_bet:
                    st.markdown(f'<div style="text-align: right;"><span class="badge-bet">{verdict_str}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="text-align: right;"><span class="badge-pass">{verdict_str}</span></div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Model Win Prob", f"{home_win_pct:.1f}%")
            m2.metric("Market Consensus", f"{market_win_pct:.1f}%", f"{home_win_pct - market_win_pct:+.1f}% vs Book")
            m3.metric("Projected Margin", f"{home_team} {p_home - p_away:+d}")
            m4.metric("Eighth-Kelly Stake", f"{stake:.2f}u")

            with st.expander("Tactical Matchup Dossier & DraftKings Comparative Matrices", expanded=is_bet):
                st.markdown(f"**Tactical Brief:** {analysis_data.get('executive_summary', 'Analysis pending.')}")
                
                sub_scheme, sub_props = st.tabs(["🧠 Trench & Coverage Clash", "🎯 DraftKings Lines vs. AI Median Estimates"])
                with sub_scheme:
                    scheme = analysis_data.get('schematic_matchup', {})
                    sc1, sc2 = st.columns(2)
                    sc1.info(f"**{away_team} Offense vs. {home_team} Defense:**\n\n" + scheme.get('away_offense_vs_home_defense', 'N/A'))
                    sc2.info(f"**{home_team} Offense vs. {away_team} Defense:**\n\n" + scheme.get('home_offense_vs_away_defense', 'N/A'))

                with sub_props:
                    projections = analysis_data.get('player_projections', [])
                    def render_prop_matrix(t_name, col_box):
                        with col_box:
                            st.markdown(f"##### {t_name} Output vs. Market Lines")
                            t_props = [p for p in projections if p.get("team", "").strip().upper() == t_name.strip().upper()]
                            if t_props:
                                rows = []
                                for p in t_props:
                                    m_val = float(p.get('market_line', 0.0))
                                    p_val = float(p.get('projected_value', 0.0))
                                    rows.append({
                                        "Player": f"{p.get('player', 'Unknown')} ({p.get('role', 'SKILL')})",
                                        "Stat": p.get('prop_category', 'Yards'),
                                        "Vegas Line": f"{m_val:.1f}",
                                        "AI Median": f"{p_val:.1f}",
                                        "Edge": f"{p_val - m_val:+.1f}",
                                        "Pick": p.get('edge', 'PASS').upper(),
                                        "Tactical Rationale": p.get('tactical_rationale', '-')
                                    })
                                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                            else:
                                st.caption(f"No player prop projections available for {t_name}.")

                    p_col1, p_col2 = st.columns(2)
                    render_prop_matrix(away_team, p_col1)
                    render_prop_matrix(home_team, p_col2)

            st.divider()

    if displayed == 0:
        st.info("No matchups match your edge/stake filter thresholds.")

# =========================================================
# TAB 2: STEAM STATS & CONSENSUS DISCREPANCY
# =========================================================
with tab_steam:
    st.subheader("⚡ Sharp Syndicate Steam & Consensus Discrepancy")
    steam_records = []
    for _, r in df.iterrows():
        p_c = float(r.get('home_win_prob') or 0.5)
        p_m = float(r.get('market_prob') or 0.5)
        diff = (p_c - p_m) * 100
        
        if diff >= 4.0: signal = "🔥 Sharp Home Steam / Under-Priced"
        elif diff <= -4.0: signal = "❄️ Heavy Away Steam / Market Inflated"
        else: signal = "⚖️ Consensus Fairly Priced"

        steam_records.append({
            "Matchup": r["matchup"],
            "Model Win%": f"{p_c * 100:.1f}%",
            "Market Win%": f"{p_m * 100:.1f}%",
            "Discrepancy": f"{diff:+.1f}%",
            "Spread Edge": f"{float(r.get('spread_edge') or 0.0)*100:+.1f}%",
            "Projected Score": f"{r.get('predicted_away_score')} - {r.get('predicted_home_score')}",
            "Steam Signal": signal
        })
    st.dataframe(pd.DataFrame(steam_records), hide_index=True, use_container_width=True)

# =========================================================
# TAB 3: STRATEGIC GURU WORKBENCH
# =========================================================
with tab_guru:
    st.subheader(f"🧠 {guru_mode}")
    q_title = st.text_input("Evaluation Target / Matchup Headline:", placeholder="e.g., Auditing 4th-down decision logic vs SF front")
    q_payload = st.text_area("Dossier Payload (Tape notes, EPA splits, or prompt code):", height=200)

    if st.button("Execute Strategic Guru Evaluation", type="primary", use_container_width=True):
        if not q_title or not q_payload:
            st.warning("Supply both an evaluation target and dossier payload.")
        else:
            with st.spinner("Processing scheme leverage, pocket mechanics, and statistical hygiene..."):
                prompt = f"[{guru_mode.upper()}]\nSUBJECT: {q_title}\n\nINPUT PAYLOAD:\n{q_payload}"
                try:
                    res = ai_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(system_instruction=GURU_SYSTEM_INSTRUCTION, temperature=0.15)
                    )
                    st.markdown(res.text)
                except Exception as e:
                    st.error(f"Inference Failure: {e}")

# =========================================================
# TAB 4: BLIND HISTORICAL SIMULATION (STRICT AIRLOCK)
# =========================================================
with tab_sim:
    st.subheader("🧪 Blind Past-Game Simulation Engine")
    st.caption("Validating AI predictive accuracy out-of-sample: The AI receives zero final score data.")

    sim_season = st.selectbox("Select Historical Season:", [2025, 2024], index=0)
    sim_week = st.slider("Select Historical Week:", 1, 18, 1)

    @st.cache_data(ttl=600)
    def load_historical_fixtures(season, week):
        sched = nfl.load_schedules(seasons=[season]).to_pandas()
        return sched[(sched["week"] == week) & sched["result"].notna()].copy()

    hist_games = load_historical_fixtures(sim_season, sim_week)

    if hist_games.empty:
        st.warning("No completed games found for the selected schedule.")
    else:
        st.info(f"Loaded {len(hist_games)} historical fixtures from Season {sim_season} Week {sim_week}. Final scores are masked from inference payload.")

        if st.button("Execute Blind Out-of-Sample Simulation", type="primary", use_container_width=True):
            sim_results = []
            progress_bar = st.progress(0)

            for idx, (_, g) in enumerate(hist_games.iterrows()):
                home = str(g["home_team"])
                away = str(g["away_team"])
                matchup_label = f"{away} @ {home}"

                # Extract spread_line (where positive means home is favored in nflverse standard)
                raw_spread = float(g.get("spread_line", 0.0) or 0.0)
                # Standardize to home_spread_line (points added to home team, so -raw_spread if home favored)
                home_spread_line = -raw_spread
                total_line = float(g.get("total_line", 44.0) or 44.0)

                # Ground Truth (Client-Side Only)
                actual_home = int(g["home_score"])
                actual_away = int(g["away_score"])

                # Blind Pre-Game Ingestion Dossier (Zero Score Leakage)
                blind_payload = {
                    "matchup": matchup_label,
                    "pre_game_market": {
                        "spread_line": f"{home} {home_spread_line:+g}",
                        "total_line": total_line
                    },
                    "context": f"Simulating Season {sim_season} Week {sim_week}. Strictly estimate final scores based on pre-game expectation."
                }

                blind_prompt = f"""
                You are conducting a strict out-of-sample simulation for this NFL game:
                {json.dumps(blind_payload, indent=2)}

                CRITICAL DIRECTIVE: You do not know the final score.
                Using pre-game baseline expectations, estimate the discrete final score.
                Output STRICTLY valid JSON:
                {{
                  "predicted_away_score": 0,
                  "predicted_home_score": 0,
                  "predicted_winner": "Team Abbr",
                  "tactical_thesis": "One-sentence strategic reason"
                }}
                """

                try:
                    sim_res = ai_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=blind_prompt,
                        config=types.GenerateContentConfig(temperature=0.10, response_mime_type="application/json")
                    )
                    pred = json.loads(sim_res.text)
                    p_away = int(pred.get("predicted_away_score", 20))
                    p_home = int(pred.get("predicted_home_score", 24))
                except Exception:
                    # Mathematical Discrete Fallback
                    p_home = int(round((total_line - home_spread_line) / 2.0))
                    p_away = int(round((total_line + home_spread_line) / 2.0))

                # Run Normalized Evaluation Engine
                eval_metrics = normalize_and_grade_spread(
                    pred_home_score=p_home,
                    pred_away_score=p_away,
                    actual_home_score=actual_home,
                    actual_away_score=actual_away,
                    home_spread_line=home_spread_line
                )

                sim_results.append({
                    "Matchup": matchup_label,
                    "Vegas Line": f"{home} {home_spread_line:+g}",
                    "Blind Projected Score": f"{away} {p_away} - {p_home} {home}",
                    "Actual Final Score": f"{away} {actual_away} - {actual_home} {home}",
                    "SU Winner Hit": eval_metrics["su_grade"],
                    "Spread Read": eval_metrics["ats_grade"],
                    "Score MAE": f"{eval_metrics['score_mae']:.1f} pts",
                    "Margin Delta": f"{eval_metrics['margin_error']:.1f} pts"
                })

                progress_bar.progress((idx + 1) / len(hist_games))

            df_sim = pd.DataFrame(sim_results)
            st.success("Simulation Complete. Out-of-sample grading results:")
            st.dataframe(df_sim, hide_index=True, use_container_width=True)

            su_acc = (df_sim["SU Winner Hit"] == "✅ Hit").mean() * 100
            valid_covers = df_sim[df_sim["Spread Read"].isin(["✅ Correct Cover", "❌ Wrong Side"])]
            ats_acc = (valid_covers["Spread Read"] == "✅ Correct Cover").mean() * 100 if not valid_covers.empty else 0.0

            k1, k2 = st.columns(2)
            k1.metric("Blind Outright Win Accuracy (SU)", f"{su_acc:.1f}%")
            k2.metric("Blind Spread Cover Accuracy (ATS)", f"{ats_acc:.1f}%")
