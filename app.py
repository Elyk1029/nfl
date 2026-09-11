import os
import json
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Institutional NFL Quantitative Terminal",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Environment & Client Setup
# ---------------------------------------------------------
db_url = os.environ.get("DATABASE_URL")
api_key = os.environ.get("GEMINI_API_KEY")

if not db_url:
    st.error("DATABASE_URL environment variable is not configured.")
    st.stop()

if not api_key:
    st.error("GEMINI_API_KEY environment variable is not configured.")
    st.stop()

@st.cache_resource
def get_db_engine():
    return create_engine(
        db_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=300
    )

@st.cache_resource
def get_genai_client(key: str):
    return genai.Client(api_key=key)

engine = get_db_engine()
ai_client = get_genai_client(api_key)

# ---------------------------------------------------------
# Persona Definition (Guru & AI Evaluator)
# ---------------------------------------------------------
GURU_SYSTEM_INSTRUCTION = """
# ROLE & PERSONA
You are the "NFL Analytics Coordinator & Strategic Guru," operating at the intersection of advanced football sabermetrics and high-level coaching tape analysis. You possess elite-level fluency in both traditional football film study (schemes, coverages, run fits, route concepts) and modern predictive analytics (EPA/play, CPOE, Success Rate, DVOA, pressure rate vs. quick game, NGS tracking data).

Your dual mandate:
1. Deliver razor-sharp, objective, and analytically grounded NFL football analysis.
2. Act as an expert AI evaluator: Continuously review user-submitted AI prompts, analytical frameworks, model outputs, or predictive systems to pinpoint blind spots, eliminate statistical noise, and recommend improvements.

# CORE COMPETENCIES & KNOWLEDGE BASE
- Scheme & Tactical Fluency: Personnel groupings (11, 12, 21 personnel), pass-pro schemes, run-blocking schemes (Inside/Outside Zone, Duo, Power/Counter), route distribution vs. MOFO/MOFC (Middle of Field Open/Closed), coverage shells (Cover 1, 2-Man, Quarters, Palms, Cover 3 Match).
- Advanced Metrics & Modeling: EPA per play, Success Rate, CPOE, DVOA, Adjusted Net Yards Per Attempt (ANY/A), explosive play rate, win probability models, high-leverage 4th-down decision curves.
- Data Hygiene: Sample size discipline, regressing unstable metrics (turnover luck, fumble recovery rates, red zone TD% variance) toward the mean, distinguishing process from outcome.

# OPERATIONAL MODES

### MODE 1: NFL TACTICAL & STATISTICAL BREAKDOWN
- Lead with the verdict in the first 1-2 sentences.
- Contextualize Tape + Data: Never use raw stats without schematic context, and never make film claims without backing data.
- Structure cleanly: Use markdown tables for multi-player or multi-team statistical comparisons and bullet points for schematic keys.

### MODE 2: AI & ANALYTICAL SYSTEM EVALUATION
1. Audit & Blind Spot Detection: Pinpoint where the system relies on flawed proxies, surface-level box-score stats, or narrative-driven biases.
2. Signal vs. Noise Critique: Evaluate whether feature selection or prompt reasoning isolates true predictive stability vs. game-script variance.
3. Prompt & Logic Refactoring: Provide an upgraded, production-grade version of their prompt, feature list, or mathematical framework.
4. Actionable Edge Recommendations: Recommend 2-3 specific data points, contextual filters, or architectural adjustments.

# OUTPUT CONSTRAINTS & TONE
- Tone: Direct, analytical, objective, and authoritative. Sound like an NFL director of research speaking directly to an analytics engineer or offensive coordinator.
- Banned Habits: Avoid generic sports platitudes ("they wanted it more", "momentum shifted"). Always explain causation through leverage, numbers, spacing, or probability.
- Scaffolding: Favor structured bullet points, clear tables, and concise step-by-step logic over dense blocks of text.
"""

# ---------------------------------------------------------
# Database Ingestion from Neon PostgreSQL
# ---------------------------------------------------------
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
        df = pd.read_sql(query, conn)
    return df

df = load_predictions()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("Quantitative Risk Controls")
min_spread_edge = st.sidebar.slider("Minimum Edge Cutoff %", 0.0, 10.0, 1.5, 0.25)

st.sidebar.divider()
st.sidebar.header("Guru Workbench Mode")
guru_mode = st.sidebar.radio(
    "Select Operational Path:",
    ["Mode 1: Tactical & Tape Breakdown", "Mode 2: AI & Analytical System Evaluation"],
    index=0
)

if st.sidebar.button("Purge Terminal Cache", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------
# Global Summary Metrics
# ---------------------------------------------------------
st.title("🏈 Institutional NFL Quantitative Engine")
st.caption("Bivariate Skellam Modeling | DraftKings Prop Benchmarks | Eighth-Kelly Governance")

if df.empty:
    st.info("No prediction records currently available in Neon PostgreSQL.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Games Modeled", len(df))
max_edge_row = df.loc[df['spread_edge'].abs().idxmax()]
c2.metric("Top Model Edge", f"{max_edge_row['matchup']}", f"{max_edge_row['spread_edge']*100:+.1f}%")
c3.metric("Peak Recommended Stake", f"{df['kelly_units'].max():.2f}u")
c4.metric("Active Slate", f"Week {int(df['week'].max())}")

st.divider()

# ---------------------------------------------------------
# Master Terminal Navigation
# ---------------------------------------------------------
tab_slate, tab_steam, tab_guru = st.tabs([
    "📊 Weekly Board & Player Props",
    "⚡ Steam Stats & Market Delta",
    "🧠 Strategic Guru Workbench"
])

# =========================================================
# TAB 1: MATCHUP CARDS & DRAFTKINGS PROP MATRICES
# =========================================================
with tab_slate:
    for _, row in df.iterrows():
        spread_edge_pct = (row.get('spread_edge') or 0.0) * 100
        if abs(spread_edge_pct) < min_spread_edge:
            continue

        home_win_pct = (row.get('home_win_prob') or 0.5) * 100
        market_win_pct = (row.get('market_prob') or 0.5) * 100
        cover_pct = (row.get('spread_cover_prob') or 0.5) * 100
        kelly = row.get('kelly_units') or 0.0

        with st.container():
            # Header Row: Game & Macro Pricing
            cols = st.columns([2.5, 1.5, 1.5, 1.5, 1.5])
            cols[0].subheader(row['matchup'])
            cols[1].metric("AI Projected Win", f"{home_win_pct:.1f}%")
            cols[2].metric("DraftKings Implied Win", f"{market_win_pct:.1f}%")
            cols[3].metric("AI Cover Prob", f"{cover_pct:.1f}%", f"{spread_edge_pct:+.1f}% Edge")
            cols[4].metric("Eighth-Kelly Stake", f"{kelly:.2f}u")

            try:
                analysis_data = json.loads(row['analysis'])
            except Exception:
                analysis_data = {"executive_summary": row['analysis']}

            if "schematic_matchup" in analysis_data:
                verdict = analysis_data.get('actionable_verdict', 'PASS')
                if "PASS" in verdict.upper():
                    st.info(f"**Execution:** {verdict}")
                else:
                    st.success(f"**Execution:** {verdict}")

                with st.expander("Tactical Matchup Breakdown & DraftKings Prop Comparison"):
                    st.write(f"**Tactical Brief:** {analysis_data.get('executive_summary', '')}")
                    
                    tab_scheme, tab_props = st.tabs(["🧠 Trench & Coverage Clash", "🎯 DraftKings Lines vs. AI Projections"])
                    with tab_scheme:
                        st.markdown("**Away Offense vs. Home Front & Shell**")
                        st.write(analysis_data['schematic_matchup'].get('away_offense_vs_home_defense', 'N/A'))
                        st.markdown("**Home Offense vs. Away Front & Shell**")
                        st.write(analysis_data['schematic_matchup'].get('home_offense_vs_away_defense', 'N/A'))
                    
                    with tab_props:
                        projections = analysis_data.get('player_projections', [])
                        teams = row['matchup'].split('@')
                        
                        def render_comparative_prop_table(team_name, col):
                            with col:
                                st.markdown(f"#### {team_name.strip()} Output vs. Market Lines")
                                if isinstance(projections, list):
                                    team_props = [p for p in projections if p.get("team", "").upper() == team_name.strip().upper()]
                                else:
                                    team_props = []

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
                                            "Prop": p.get("prop_category", "Yards"),
                                            "DraftKings Line": f"{m_line:.1f}",
                                            "AI Estimate": f"{proj:.1f}",
                                            "Market Edge": f"{delta:+.1f}",
                                            "Pick": action,
                                            "Tactical Rationale": p.get("tactical_rationale", "-")
                                        })
                                    
                                    prop_df = pd.DataFrame(rows)
                                    # Render comparative table with visual highlighting
                                    st.dataframe(
                                        prop_df,
                                        column_config={
                                            "DraftKings Line": st.column_config.TextColumn("Sportsbook Line", help="DraftKings/Consensus Benchmark Line"),
                                            "AI Estimate": st.column_config.TextColumn("AI Projection", help="Model projected yardage"),
                                            "Market Edge": st.column_config.TextColumn("Edge (Delta)", help="AI Estimate minus Sportsbook Line"),
                                            "Pick": st.column_config.TextColumn("Execution", help="Actionable OVER / UNDER / PASS"),
                                            "Tactical Rationale": st.column_config.TextColumn("Coaching Film Note", width="large")
                                        },
                                        hide_index=True,
                                        use_container_width=True
                                    )
                                else:
                                    st.caption("No structured player prop data available for this squad.")

                        c_away, c_home = st.columns(2)
                        render_comparative_prop_table(teams[0], c_away)
                        render_comparative_prop_table(teams[1], c_home)
            else:
                with st.expander("Analysis Logs"):
                    st.write(row['analysis'])

            st.divider()

# =========================================================
# TAB 2: STEAM STATS & MARKET DELTA
# =========================================================
with tab_steam:
    st.subheader("⚡ Consensus Steam & Market Discrepancy Matrix")
    st.caption("Isolating sharp syndicate line movement vs. AI model power rating differentials.")

    steam_records = []
    for _, r in df.iterrows():
        p_cal = float(r.get('home_win_prob') or 0.5)
        p_mkt = float(r.get('market_prob') or 0.5)
        discrepancy = (p_cal - p_mkt) * 100
        edge = float(r.get('spread_edge') or 0.0) * 100
        units = float(r.get('kelly_units') or 0.0)

        if discrepancy >= 4.0:
            bias = "🔥 Sharp Home Steam / Under-priced"
        elif discrepancy <= -4.0:
            bias = "❄️ Heavy Away Steam / Public Inflated"
        else:
            bias = "⚖️ Consensus Fairly Priced"

        steam_records.append({
            "Matchup": r["matchup"],
            "Model Win%": f"{p_cal * 100:.1f}%",
            "DraftKings Implied%": f"{p_mkt * 100:.1f}%",
            "Market Discrepancy": f"{discrepancy:+.1f}%",
            "Spread Cover Edge": f"{edge:+.1f}%",
            "Eighth-Kelly Stake": f"{units:.2f}u",
            "Steam Signal": bias
        })

    st.dataframe(pd.DataFrame(steam_records), hide_index=True, use_container_width=True)

# =========================================================
# TAB 3: STRATEGIC GURU WORKBENCH
# =========================================================
with tab_guru:
    st.subheader(f"🧠 {guru_mode}")
    st.caption("Direct coaching tape-to-metric analysis and automated AI model/prompt auditing.")

    query_title = st.text_input(
        "Subject / Matchup / Framework Title:",
        placeholder="e.g. Auditing 4th-Down Aggressiveness Index or Evaluating SF Outside Zone vs Rams Odd Front"
    )
    raw_payload = st.text_area(
        "Input Payload (Tape Notes, EPA Splits, AI Prompt, Python Code, or Projections):",
        height=220,
        placeholder="Paste coaching tape observations, advanced metrics (EPA, CPOE, PBWR), or an AI system prompt/feature schema for review..."
    )

    if st.button("Execute Strategic Guru Evaluation", type="primary", use_container_width=True):
        if not query_title or not raw_payload:
            st.warning("Please supply both a subject title and an input payload.")
        else:
            with st.spinner("Analyzing scheme mechanics, coverage shells, and statistical rigor..."):
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
                    st.error(f"Inference Error: {e}")
