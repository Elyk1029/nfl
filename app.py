"""
app.py - Institutional NFL Quantitative Terminal with Strategic Guru Workbench.
Hardened with Dual-Resolution Credential Handling & 2026 Directory Injections.
"""
import os
import json
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Institutional NFL Quantitative Terminal",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def resolve_credential(key_name: str) -> str:
    try:
        if key_name in st.secrets and str(st.secrets[key_name]).strip():
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass

    env_val = os.environ.get(key_name)
    if env_val and str(env_val).strip():
        return str(env_val).strip()

    return ""

db_url = resolve_credential("DATABASE_URL")
api_key = resolve_credential("GEMINI_API_KEY")

if not db_url or not api_key:
    missing = []
    if not db_url:
        missing.append("DATABASE_URL (Neon PostgreSQL)")
    if not api_key:
        missing.append("GEMINI_API_KEY (Google GenAI)")

    st.error(f"⚠️ Missing System Credentials: {', '.join(missing)}")
    st.markdown(
        """
        **Resolution Instructions:**
        Create `.streamlit/secrets.toml` in the root working directory:
        ```toml
        DATABASE_URL = "postgresql://<user>:<password>@<host>/<dbname>?sslmode=require"
        GEMINI_API_KEY = "AIzaSy..."
        ```
        """
    )
    st.stop()

@st.cache_resource
def get_db_engine(connection_string: str):
    return create_engine(connection_string, pool_size=5, max_overflow=10, pool_pre_ping=True, pool_recycle=300)

@st.cache_resource
def get_genai_client(key: str):
    return genai.Client(api_key=key)

engine = get_db_engine(db_url)
ai_client = get_genai_client(api_key)

# Injected 11/10 System Instruction with 2026 Scheme & Play-Caller Directory
GURU_SYSTEM_INSTRUCTION = """
# ROLE & IDENTITY
You are the "NFL Research Director & Quantitative Architect," operating at the nexus of NFL coaching tape breakdown, spatiotemporal tracking physics (Next Gen Stats), and advanced sabermetric modeling.

Your dual mandate:
1. Deliver razor-sharp, objective, and analytically grounded NFL football breakdowns anchored strictly in the current 2026 NFL campaign.
2. Serve as an expert AI evaluator: Continuously audit user-submitted AI prompts, analytical frameworks, statistical models, and projection logic to eliminate statistical noise, correct proxy errors, and enforce production-grade quantitative rigor.

---

## 1. 2026 PLAY-CALLER & TACTICAL CONTINUITY DIRECTORY
Never analyze teams by helmet logos or legacy 2024–2025 play-callers. All schematic breakdowns must reflect confirmed 2026 play-calling leadership and system architecture:
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

## 2. DETERMINISTIC MODE ROUTING & ACTIVATION
- Trigger MODE 1 (Tactical & Tape Breakdown) if the query asks about game matchups, scheme clashes, player evaluation, roster trends, or football tape analysis without requesting an evaluation of an external prompt/system.
- Trigger MODE 2 (AI & Analytical System Evaluation) if the query contains code, prompts, statistical formulas, betting theses, model outputs, or explicitly asks for an audit, critique, or optimization.
- Precedence Hierarchy: If an input contains elements of both, execute MODE 2 as the primary response, utilizing MODE 1 analysis as the worked test case.

---

## 3. SCHEMATIC TAXONOMY & PHYSICAL INVARIANTS
- Trench & Pocket Physics: Time-to-Pressure (TTP) vs. Time-to-Throw (TTT) determines pocket degradation. If TTP < TTT, evaluate pocket mobility archetype. Immobile pocket passers collapse under duress (P2S > 20%, steep YPA drop); play-extending dual threats convert pressure into scramble EPA or extended second-reaction attempts.
- Run-Fit Geometry: Gap/Duo/Power creates vertical displacement via double-teams, exploiting light nickel boxes (6-man fronts) and split safeties; neutralized by Odd 3-4 fronts with 0/1-technique two-gapping interior tackles. Wide Zone creates horizontal stretch, exploiting aggressive interior penetrators; neutralized by Wide-9 alignments and disciplined C-gap setters.
- Coverage Shell Conditioning: Defenses adjust coverage shells based on offensive personnel groupings. Never cite seasonal coverage rates in a vacuum. Evaluate defensive response specifically against the offense's primary personnel package (e.g., 11 vs. 12/21 personnel).
  - MOFC (Cover 1 / Cover 3 Match): Single-high safety; leaves perimeter 1-on-1s on the boundary; vulnerable to intermediate Dagger concepts, crossing routes, and deep seam shots.
  - MOFO (Cover 2 / Quarters / Cover 6): Split safeties; caps vertical boundary routes; vulnerable to underneath checkdowns, intermediate hole shots, and gap runs vs. light boxes.

---

## 4. MATHEMATICAL DISCIPLINE & DATA HYGIENE
- Garbage-Time & Leverage Filtration: Filter all EPA, CPOE, and Success Rate metrics to neutral game states: Win Probability between 10% and 90%, excluding final-two-minute desperation drives and fourth-quarter blowouts (margin >= 16 points).
- Log-Normal Median Transformation for Player Props: Sportsbooks price prop lines near the distribution median (50th percentile). Convert projected mean yardage (mu) to estimated median (m) using position-specific log-variance:
  m = mu * exp(-(sigma^2) / 2)
  sigma_QB_Pass = 0.32, sigma_RB_Rush = 0.48, sigma_Skill_Rec = 0.58
- Discrete Scoring Margins & Push Accounting: NFL scoring distributions are discrete point masses concentrated on key numbers (3, 7, 6, 10, 4, 14). Never assume continuous normal distributions when calculating cover probabilities. Calculate Eighth-Kelly sizing with push probability (p_push):
  f* = (b * p - q) / b, where q = 1.0 - p - p_push
- Epistemic Calibration & Anti-Hallucination: If the input does not provide verified Next Gen Stats (NGS) tracking data, pressure numbers, or EPA splits, express metrics in directional percentiles, schematic tiers, or observable film tendencies. Never fabricate decimal-precision statistics.

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

---

## 6. OUTPUT CONSTRAINTS & TONE
- Direct, analytical, objective, authoritative. No generic sports platitudes. Causation must be established via leverage, numbers, spacing, or probability.
- No conversational fluff or filler introductions. Jump directly into the structured data or verdict.
- Do not end responses with artificial headers like 'Summary:' or 'Conclusion:'.
"""

@st.cache_data(ttl=300)
def load_predictions():
    query = """
        SELECT DISTINCT ON (game_id)
            game_id,
            week,
            matchup,
            home_win_prob,
            market_prob,
            spread_cover_prob,
            spread_edge,
            kelly_units,
            (home_win_prob - market_prob) as ml_edge,
            analysis
        FROM nfl_weekly_analysis
        ORDER BY game_id, week DESC;
    """
    with engine.connect() as conn:
        data = pd.read_sql(query, conn)
    return data

try:
    df = load_predictions()
except Exception as e:
    st.error(f"Database Query Failed: {e}")
    st.stop()

with st.sidebar:
    st.title("🏈 Risk Engine")
    st.caption("Quantitative Execution Controls")

    min_spread_edge = st.slider("Minimum Edge Cutoff %", 0.0, 10.0, 1.5, 0.25)
    max_game_exposure = st.slider("Max Intra-Game Exposure (Units)", 1.0, 4.0, 2.5, 0.25)

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

st.title("🏈 Institutional NFL Quantitative Terminal")
st.caption("Opponent-Adjusted Ridge EPA | Log-Normal Median Calibration | Correlated Risk Management")

if df.empty:
    st.info("No active slate predictions currently loaded in Neon PostgreSQL.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Games Evaluated", len(df))
max_edge_row = df.loc[df['spread_edge'].abs().idxmax()]
c2.metric("Top Spread Edge", f"{max_edge_row['matchup']}", f"{max_edge_row['spread_edge']*100:+.1f}%")
c3.metric("Peak Recommended Stake", f"{df['kelly_units'].max():.2f}u")
c4.metric("Active Slate", f"Week {int(df['week'].max())}")

st.divider()

tab_slate, tab_steam, tab_guru = st.tabs([
    "📊 Weekly Board & Median Props",
    "⚡ Steam Stats & Line Delta",
    "🧠 Strategic Guru Workbench"
])

with tab_slate:
    for _, row in df.iterrows():
        spread_edge_pct = (row.get('spread_edge') or 0.0) * 100
        if abs(spread_edge_pct) < min_spread_edge:
            continue

        home_win_pct = (row.get('home_win_prob') or 0.5) * 100
        market_win_pct = (row.get('market_prob') or 0.5) * 100
        cover_pct = (row.get('spread_cover_prob') or 0.5) * 100
        kelly = float(row.get('kelly_units') or 0.0)

        allocated_stake = min(kelly, max_game_exposure)

        with st.container():
            cols = st.columns([2.5, 1.5, 1.5, 1.5, 1.5])
            cols[0].subheader(row['matchup'])
            cols[1].metric("Calibrated Model Win%", f"{home_win_pct:.1f}%")
            cols[2].metric("Devigged Consensus", f"{market_win_pct:.1f}%")
            cols[3].metric("Cover Probability", f"{cover_pct:.1f}%", f"{spread_edge_pct:+.1f}% Edge")
            cols[4].metric("Eighth-Kelly Stake", f"{allocated_stake:.2f}u")

            try:
                analysis_data = json.loads(row['analysis'])
            except Exception:
                analysis_data = {"executive_summary": row['analysis']}

            if "schematic_matchup" in analysis_data:
                verdict = analysis_data.get('actionable_verdict', 'PASS')
                if "PASS" in verdict.upper():
                    st.info(f"**Execution Verdict:** {verdict}")
                else:
                    st.success(f"**Execution Verdict:** {verdict}")

                with st.expander("Tactical Matchup Dossier & DraftKings Comparative Matrices"):
                    st.write(f"**Tactical Brief:** {analysis_data.get('executive_summary', '')}")
                    
                    tab_scheme, tab_props = st.tabs(["🧠 Trench & Coverage Clash", "🎯 DraftKings Lines vs. AI Median Estimates"])
                    with tab_scheme:
                        st.markdown("**Away Offense vs. Home Front & Shell**")
                        st.write(analysis_data['schematic_matchup'].get('away_offense_vs_home_defense', 'N/A'))
                        st.markdown("**Home Offense vs. Away Front & Shell**")
                        st.write(analysis_data['schematic_matchup'].get('home_offense_vs_away_defense', 'N/A'))
                    
                    with tab_props:
                        projections = analysis_data.get('player_projections', [])
                        teams = row['matchup'].split('@')
                        
                        def render_prop_matrix(team_name, col):
                            with col:
                                st.markdown(f"#### {team_name.strip()} Output vs. Market Lines")
                                team_props = [
                                    p for p in projections 
                                    if p.get("team", "").strip().upper() == team_name.strip().upper()
                                ] if isinstance(projections, list) else []

                                if team_props:
                                    rows = []
                                    for p in team_props:
                                        m_line = float(p.get('market_line', 0.0))
                                        proj = float(p.get('projected_value', 0.0))
                                        delta = proj - m_line
                                        action = p.get("edge", "PASS").upper()
                                        
                                        rows.append({
                                            "Role": p.get("role", "SKILL"),
                                            "Player": p.get("player", "Unknown"),
                                            "Category": p.get("prop_category", "Yards"),
                                            "Sportsbook Line": f"{m_line:.1f}",
                                            "AI Median": f"{proj:.1f}",
                                            "Market Edge": f"{delta:+.1f}",
                                            "Pick": action,
                                            "Tactical Rationale": p.get("tactical_rationale", "-")
                                        })
                                    
                                    st.dataframe(
                                        pd.DataFrame(rows),
                                        column_config={
                                            "Sportsbook Line": st.column_config.TextColumn("Consensus Line"),
                                            "AI Median": st.column_config.TextColumn("AI Median (50th%)"),
                                            "Market Edge": st.column_config.TextColumn("Edge (Delta)"),
                                            "Pick": st.column_config.TextColumn("Action"),
                                            "Tactical Rationale": st.column_config.TextColumn("Coaching Film Note", width="large")
                                        },
                                        hide_index=True,
                                        use_container_width=True
                                    )
                                else:
                                    st.caption("No structured player prop data available for this roster.")

                        c_away, c_home = st.columns(2)
                        render_prop_matrix(teams[0], c_away)
                        render_prop_matrix(teams[1], c_home)
            else:
                with st.expander("Raw Analysis Record"):
                    st.write(row['analysis'])

            st.divider()

with tab_steam:
    st.subheader("⚡ Sharp Syndicate Steam & Consensus Discrepancy")
    st.caption("Quantifying divergence between devigged market consensus and calibrated model power ratings.")

    steam_records = []
    for _, r in df.iterrows():
        p_cal = float(r.get('home_win_prob') or 0.5)
        p_mkt = float(r.get('market_prob') or 0.5)
        discrepancy = (p_cal - p_mkt) * 100
        edge = float(r.get('spread_edge') or 0.0) * 100
        units = float(r.get('kelly_units') or 0.0)

        if discrepancy >= 4.0:
            bias = "🔥 Sharp Home Steam / Market Under-Priced"
        elif discrepancy <= -4.0:
            bias = "❄️ Heavy Away Steam / Market Inflated"
        else:
            bias = "⚖️ Consensus Fairly Priced"

        steam_records.append({
            "Matchup": r["matchup"],
            "Model Win%": f"{p_cal * 100:.1f}%",
            "Devigged Market%": f"{p_mkt * 100:.1f}%",
            "Market Discrepancy": f"{discrepancy:+.1f}%",
            "Spread Cover Edge": f"{edge:+.1f}%",
            "Eighth-Kelly Stake": f"{units:.2f}u",
            "Steam Signal": bias
        })

    st.dataframe(pd.DataFrame(steam_records), hide_index=True, use_container_width=True)

with tab_guru:
    st.subheader(f"🧠 {guru_mode}")
    st.caption("Interactive coaching tape-to-metric analysis and production-grade AI prompt auditing.")

    query_title = st.text_input(
        "Evaluation Target / Matchup Headline:",
        placeholder="e.g., Auditing 4th-down decision logic or Stafford pocket degradation vs SF Odd Front"
    )
    raw_payload = st.text_area(
        "Dossier Payload (Tape Notes, EPA Splits, Prompt Code, or Model Arrays):",
        height=240,
        placeholder="Paste coaching tape notes, Pass Block Win Rates, EPA splits, or prompt templates for evaluation..."
    )

    if st.button("Execute Strategic Guru Evaluation", type="primary", use_container_width=True):
        if not query_title or not raw_payload:
            st.warning("Both an evaluation target and a dossier payload must be supplied.")
        else:
            with st.spinner("Processing scheme leverage, pocket mechanics, and statistical hygiene..."):
                augmented_prompt = f"[{guru_mode.upper()}]\nSUBJECT: {query_title}\n\nINPUT PAYLOAD:\n{raw_payload}"
                try:
                    res = ai_client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=augmented_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=GURU_SYSTEM_INSTRUCTION,
                            temperature=0.15
                        )
                    )
                    st.markdown(res.text)
                except Exception as e:
                    st.error(f"Inference Failure: {e}")
