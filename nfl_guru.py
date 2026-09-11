import os
import json
import streamlit as st
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. Environment & Client Verification
# ---------------------------------------------------------
st.set_page_config(
    page_title="NFL Strategic Guru & AI Evaluator",
    page_icon="🏈",
    layout="wide"
)

DATABASE_URL = os.environ.get("DATABASE_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not DATABASE_URL:
    st.error("DATABASE_URL environment variable is not configured. Connect your Neon Postgres instance.")
    st.stop()

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY environment variable is not configured. Add your Google GenAI API key.")
    st.stop()

# Neon connection engine with auto-reconnect and pooling
@st.cache_resource
def get_neon_engine():
    return create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_recycle=300,
        pool_pre_ping=True
    )

engine = get_neon_engine()
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------
# 2. Database Schema Initialization
# ---------------------------------------------------------
def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS nfl_guru_audits (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                eval_mode TEXT NOT NULL,
                headline TEXT NOT NULL,
                input_payload TEXT NOT NULL,
                verdict TEXT NOT NULL,
                full_analysis TEXT NOT NULL
            );
        """))

init_db()

# ---------------------------------------------------------
# 3. Core Persona System Instructions
# ---------------------------------------------------------
GURU_SYSTEM_INSTRUCTION = """
# ROLE & PERSONA
You are the "NFL Analytics Coordinator & Strategic Guru," operating at the intersection of advanced football sabermetrics and high-level coaching tape analysis. You possess elite-level fluency in both traditional football film study (schemes, coverages, run fits, route concepts) and modern predictive analytics (EPA/play, CPOE, Success Rate, DVOA, pressure rate vs. quick game, NGS tracking data).

Your dual mandate:
1. Deliver razor-sharp, objective, and analytically grounded NFL football analysis.
2. Act as an expert AI evaluator: Continuously review user-submitted AI prompts, analytical frameworks, model outputs, or predictive systems to pinpoint blind spots, eliminate statistical noise, and recommend improvements.

# CORE COMPETENCIES & KNOWLEDGE BASE
- Scheme & Tactical Fluency: Personnel groupings (11, 12, 21 personnel), pass-pro schemes, run-blocking schemes (Inside/Outside Zone, Duo, Power/Counter), route distribution vs. MOFO/MOFC (Middle of Field Open/Closed), coverage shells (Cover 1, 2-Man, Quarters, Palms, Cover 3 Match).
- Advanced Metrics & Modeling: EPA per play, Success Rate, CPOE, Adjusted Net Yards Per Attempt (ANY/A), explosive play rate, win probability models, high-leverage 4th-down decision curves.
- Data Hygiene: Sample size discipline, regressing unstable metrics (turnover luck, fumble recovery rates, red zone TD% variance) toward the mean, distinguishing process from outcome.

# OPERATIONAL MODES

### MODE 1: NFL TACTICAL & STATISTICAL BREAKDOWN
- Lead with the verdict in the first 1-2 sentences.
- Contextualize Tape + Data: Never cite raw numbers without schematic root causes; never make tape claims without EPA, pressure rate, or success rate metrics.
- Structure cleanly: Use Markdown tables for comparative data and bullet points for strategic keys.

### MODE 2: AI & ANALYTICAL SYSTEM EVALUATION
1. Audit & Blind Spot Detection: Pinpoint reliance on flawed proxies, raw box-score counting stats, or narrative bias.
2. Signal vs. Noise Critique: Evaluate whether features isolate true predictive stability vs. game-script variance.
3. Prompt & Logic Refactoring: Deliver production-ready code, prompt revisions, or mathematical adjustments.
4. Actionable Edge Recommendations: Provide 2-3 specific data points or architectural upgrades.

# OUTPUT CONSTRAINTS & TONE
- Tone: Direct, analytical, objective, authoritative. Sound like an NFL director of research speaking directly to an analytics engineer or offensive coordinator.
- Banned Habits: No generic sports platitudes ("they wanted it more", "momentum shifted"). Explain causation via leverage, numbers, spacing, or probability.
- Scaffolding: Favor tables, bold inline callouts, and structured bullet points over walls of text.
"""

# ---------------------------------------------------------
# 4. LLM Generation & Neon Persistence Service
# ---------------------------------------------------------
def run_guru_evaluation(mode: str, topic_headline: str, user_payload: str) -> str:
    prompt = f"""
### EXECUTION REQUEST: {mode.upper()}
**TOPIC / HEADLINE:** {topic_headline}

**INPUT PAYLOAD / RAW DATA / PROMPT:**
{user_payload}

Analyze the input strictly according to your defined system instructions and operational modes.
"""
    response = ai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=GURU_SYSTEM_INSTRUCTION,
            temperature=0.15
        )
    )
    analysis_text = response.text

    # Extract first sentence as primary verdict
    first_line = analysis_text.strip().split("\n")[0].replace("#", "").strip()

    # Save execution to Neon PostgreSQL
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO nfl_guru_audits (eval_mode, headline, input_payload, verdict, full_analysis)
                VALUES (:eval_mode, :headline, :input_payload, :verdict, :full_analysis)
            """),
            {
                "eval_mode": mode,
                "headline": topic_headline,
                "input_payload": user_payload,
                "verdict": first_line[:255],
                "full_analysis": analysis_text
            }
        )
    return analysis_text

# ---------------------------------------------------------
# 5. Streamlit User Interface
# ---------------------------------------------------------
st.title("🏈 NFL Analytics Coordinator & Strategic Guru Terminal")
st.caption("Tape-Grounded Sabermetrics | Quantitative Prompt & Pipeline Auditing | Neon DB Sync")

# Tabs for Execution Modes & Database Logs
tab_mode1, tab_mode2, tab_history = st.tabs([
    "Mode 1: Tactical & Tape Breakdown",
    "Mode 2: AI & Pipeline Auditor",
    "🗄️ Neon Audit Archive"
])

# --- MODE 1: Tactical & Tape Breakdown ---
with tab_mode1:
    st.markdown("### Matchup, Scheme, and Player Evaluation")
    m1_headline = st.text_input("Matchup / Evaluation Subject", placeholder="e.g. SF 21-Personnel Outside Zone vs LA Odd-Front Quarters")
    m1_payload = st.text_area(
        "Tape Observations & Quantitative Baselines",
        height=200,
        placeholder="Paste personnel usage rates, EPA/play splits, pressure-to-sack ratios, or film tendencies..."
    )

    if st.button("Generate Strategic Breakdown", type="primary", use_container_width=True):
        if not m1_headline or not m1_payload:
            st.warning("Provide both an evaluation subject and data/tape context.")
        else:
            with st.spinner("Analyzing coverage shells, trench leverage, and EPA metrics..."):
                output = run_guru_evaluation("Mode 1: Tactical & Statistical Breakdown", m1_headline, m1_payload)
                st.markdown(output)

# --- MODE 2: AI & System Evaluation ---
with tab_mode2:
    st.markdown("### Prompt, Model Architecture & Thesis Audit")
    m2_headline = st.text_input("Model / Architecture Title", placeholder="e.g. Fourth-Down Decision Classifier or WR Prop XGBoost Framework")
    m2_payload = st.text_area(
        "Paste AI System Prompt, Python Feature Pipeline, or Math Framework",
        height=250,
        placeholder="Paste your prompt, feature vector list, betting heuristic, or raw LLM output here..."
    )

    if st.button("Audit Analytical Framework", type="primary", use_container_width=True):
        if not m2_headline or not m2_payload:
            st.warning("Provide both a framework title and prompt/code payload.")
        else:
            with st.spinner("Auditing for proxy errors, proxy leakage, and structural blind spots..."):
                output = run_guru_evaluation("Mode 2: AI & Analytical System Evaluation", m2_headline, m2_payload)
                st.markdown(output)

# --- TAB 3: Neon Audit Archive ---
with tab_history:
    st.markdown("### Historical Evaluations Synced to Neon")
    
    with engine.connect() as conn:
        archive_df = pd.read_sql(
            "SELECT id, created_at, eval_mode, headline, verdict, full_analysis FROM nfl_guru_audits ORDER BY created_at DESC LIMIT 50;",
            conn
        )

    if archive_df.empty:
        st.info("No audit evaluations recorded yet.")
    else:
        st.dataframe(
            archive_df[["id", "created_at", "eval_mode", "headline", "verdict"]],
            use_container_width=True,
            hide_index=True
        )

        selected_id = st.selectbox("Inspect Full Evaluation Record", archive_df["id"].tolist())
        selected_record = archive_df[archive_df["id"] == selected_id].iloc[0]
        
        with st.expander(f"Record #{selected_record['id']} - {selected_record['headline']}", expanded=True):
            st.caption(f"Executed on {selected_record['created_at']} | {selected_record['eval_mode']}")
            st.markdown(selected_record["full_analysis"])
