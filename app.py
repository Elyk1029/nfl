"""
app.py - Institutional NFL Quantitative Terminal & Strategic Guru Workbench.
Features:
- Predicted Discrete Game Scores (Team Totals & Combined Score).
- Tab 4: Blind Historical Simulation & AI Backtester (Scores strictly masked during inference).
"""
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from google import genai
from google.genai import types
from scipy.stats import norm
import nflreadpy as nfl

st.set_page_config(
    page_title="NFL Quantitative Terminal | 2026 Season",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #1a1e24;
        border: 1px solid #2d3748;
        padding: 12px 16px;
        border-radius: 8px;
    }
    .score-badge {
        background-color: #2b6cb0;
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 1.15rem;
        display: inline-block;
        border: 1px solid #4299e1;
    }
    .badge-bet {
        background-color: #276749;
        color: #c6f6d5;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-pass {
        background-color: #2d3748;
        color: #cbd5e0;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

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
    st.error("DATABASE_URL and GEMINI_API_KEY must be configured in secrets or environment.")
    st.stop()

@st.cache_resource
def get_db_engine(connection_string: str):
    return create_engine(connection_string, pool_size=5, max_overflow=10, pool_pre_ping=True, pool_recycle=300)

@st.cache_resource
def get_genai_client(key: str):
    return genai.Client(api_key=key)

engine = get_db_engine(db_url)
ai_client = get_genai_client(api_key)

GURU_SYSTEM_INSTRUCTION = """
# ROLE & IDENTITY
You are the NFL Research Director & Quantitative Architect.
Deliver objective, accessible game analyses and evaluate pre-game betting leverage.
Never fabricate stats. Adhere strictly to empirical football physics.
"""

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
    st.error(f"Database Query Error: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.title("🏈 Risk Engine")
    show_only_bets = st.checkbox("Show Actionable Bets Only", value=False)
    min_edge = st.slider("Minimum Edge Cutoff %", 0.0, 5.0, 1.5, 0.25)
    max_stake = st.slider("Max Intra-Game Exposure (Units)", 0.5, 3.0, 2.0, 0.25)
    st.divider()
    if st.button("Purge Terminal Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.title("🏈 Institutional NFL Quantitative Terminal")
st.caption("Discrete Score Projection | Opponent-Adjusted EPA | Blind Historical Calibration")

if df.empty:
    st.info("No active games currently loaded.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Games Modeled", len(df))
max_edge_row = df.loc[df['spread_edge'].abs().idxmax()]
c2.metric("Top Model Edge", f"{max_edge_row['matchup']}", f"{max_edge_row['spread_edge']*100:+.1f}%")
c3.metric("Peak Stake", f"{min(df['kelly_units'].max(), max_stake):.2f}u")
c4.metric("Active Slate", f"Week {int(df['week'].max())}")

st.divider()

tab_slate, tab_steam, tab_guru, tab_sim = st.tabs([
    "📊 Weekly Board & Predicted Scores",
    "⚡ Steam & Line Movement",
    "🧠 Strategic Guru Workbench",
    "🧪 Blind Historical Simulation"
])

# =========================================================
# TAB 1: WEEKLY BOARD & PREDICTED SCORES
# =========================================================
with tab_slate:
    for _, row in df.iterrows():
        edge_pct = (row.get('spread_edge') or 0.0) * 100
        stake = min(float(row.get('kelly_units') or 0.0), max_stake)

        if show_only_bets and stake <= 0.0:
            continue
        if abs(edge_pct) < min_edge:
            continue

        try:
            analysis_data = json.loads(row['analysis'])
        except Exception:
            analysis_data = {}

        teams = row['matchup'].split('@')
        away_team = teams[0].strip()
        home_team = teams[1].strip()

        p_home_score = int(row.get('predicted_home_score') or 24)
        p_away_score = int(row.get('predicted_away_score') or 20)
        p_total = int(row.get('predicted_total_score') or (p_home_score + p_away_score))

        verdict = analysis_data.get('actionable_verdict', 'PASS')
        is_bet = "BET" in verdict.upper()

        with st.container():
            col_matchup, col_scores, col_action = st.columns([2.5, 2.5, 1.5])
            with col_matchup:
                st.subheader(row['matchup'])
                st.caption(f"Cover Probability: {row['spread_cover_prob']*100:.1f}% | Net Edge: {edge_pct:+.1f}%")
            with col_scores:
                st.markdown(
                    f"""
                    <div style="padding-top: 5px;">
                        <span class="score-badge">{away_team} {p_away_score} - {p_home_score} {home_team}</span>
                        <span style="font-size: 0.9rem; color: #a0aec0; margin-left: 10px;">(Projected Total: {p_total})</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            with col_action:
                if is_bet:
                    st.markdown(f'<div style="text-align: right;"><span class="badge-bet">{verdict}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="text-align: right;"><span class="badge-pass">{verdict}</span></div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("AI Win Prob", f"{row['home_win_prob']*100:.1f}%")
            m2.metric("Market Consensus", f"{row['market_prob']*100:.1f}%")
            m3.metric("Predicted Margin", f"{home_team} {p_home_score - p_away_score:+d}")
            m4.metric("Eighth-Kelly Stake", f"{stake:.2f}u")

            with st.expander("Tactical Matchup Breakdown"):
                st.write(f"**Game Note:** {analysis_data.get('executive_summary', 'Tactical note pending.')}")
                scheme = analysis_data.get('schematic_matchup', {})
                sc1, sc2 = st.columns(2)
                sc1.info(f"**{away_team} Offense vs. {home_team} Defense:**\n\n" + scheme.get('away_offense_vs_home_defense', 'N/A'))
                sc2.info(f"**{home_team} Offense vs. {away_team} Defense:**\n\n" + scheme.get('home_offense_vs_away_defense', 'N/A'))

            st.divider()

# =========================================================
# TAB 2: STEAM STATS
# =========================================================
with tab_steam:
    st.subheader("⚡ Line Movement & Market Discrepancies")
    steam_records = []
    for _, r in df.iterrows():
        p_cal = float(r.get('home_win_prob') or 0.5)
        p_mkt = float(r.get('market_prob') or 0.5)
        discrepancy = (p_cal - p_mkt) * 100
        steam_records.append({
            "Matchup": r["matchup"],
            "Model Win%": f"{p_cal * 100:.1f}%",
            "Consensus Win%": f"{p_mkt * 100:.1f}%",
            "Discrepancy": f"{discrepancy:+.1f}%",
            "Spread Edge": f"{float(r.get('spread_edge') or 0.0)*100:+.1f}%",
            "Predicted Score": f"{r.get('predicted_away_score')} - {r.get('predicted_home_score')}"
        })
    st.dataframe(pd.DataFrame(steam_records), hide_index=True, use_container_width=True)

# =========================================================
# TAB 3: GURU WORKBENCH
# =========================================================
with tab_guru:
    st.subheader("🧠 Strategic Guru Interactive Workbench")
    q_title = st.text_input("Matchup / Query Headline:")
    q_payload = st.text_area("Input Dossier (Tape notes, EPA splits):", height=180)
    if st.button("Run Guru Evaluation", type="primary", use_container_width=True):
        if q_title and q_payload:
            with st.spinner("Generating film breakdown..."):
                res = ai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"SUBJECT: {q_title}\n\nDATA:\n{q_payload}",
                    config=types.GenerateContentConfig(system_instruction=GURU_SYSTEM_INSTRUCTION, temperature=0.15)
                )
                st.markdown(res.text)

# =========================================================
# TAB 4: BLIND HISTORICAL SIMULATION (STRICT AIRLOCK)
# =========================================================
with tab_sim:
    st.subheader("🧪 Blind Past-Game Simulation Engine")
    st.caption("Validating AI predictive accuracy out-of-sample: The AI receives zero final score data.")

    sim_season = st.selectbox("Select Historical Season to Simulate:", [2025, 2024], index=0)
    sim_week = st.slider("Select Historical Week:", 1, 18, 1)

    @st.cache_data(ttl=600)
    def load_historical_fixtures(season, week):
        sched = nfl.load_schedules(seasons=[season]).to_pandas()
        return sched[(sched["week"] == week) & sched["result"].notna()].copy()

    hist_games = load_historical_fixtures(sim_season, sim_week)

    if hist_games.empty:
        st.warning("No historical completed games found for this period.")
    else:
        st.info(f"Loaded {len(hist_games)} historical fixtures from Season {sim_season} Week {sim_week}. Final scores are masked from the inference payload.")

        if st.button("Execute Blind Out-of-Sample Simulation", type="primary"):
            sim_results = []
            prog_bar = st.progress(0)

            for idx, (_, g) in enumerate(hist_games.iterrows()):
                home = str(g["home_team"])
                away = str(g["away_team"])
                matchup_label = f"{away} @ {home}"
                spread = float(g.get("spread_line", 0.0) or 0.0)
                total = float(g.get("total_line", 44.0) or 44.0)

                # Ground Truth (Kept strictly on client side for grading)
                actual_home = int(g["home_score"])
                actual_away = int(g["away_score"])
                actual_margin = actual_home - actual_away
                actual_total = actual_home + actual_away

                # Pure Pre-Game Airlocked Dossier (Zero Score Leakage)
                blind_payload = {
                    "matchup": matchup_label,
                    "pre_game_market": {
                        "spread_line": spread,
                        "total_line": total
                    },
                    "context": f"Simulating Season {sim_season} Week {sim_week}. Predict discrete scores based strictly on pre-kickoff leverage."
                }

                blind_prompt = f"""
                You are conducting a strict out-of-sample simulation for this NFL game:
                {json.dumps(blind_payload, indent=2)}

                CRITICAL DIRECTIVE: You do not know the final score.
                Using historical baseline expectations, estimate the final score.
                Output STRICTLY valid JSON matching:
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
                    p_home = int(pred.get("predicted_home_score", 23))
                except Exception:
                    # Mathematical Fallback
                    p_home = int(round((total - spread) / 2.0))
                    p_away = int(round((total + spread) / 2.0))

                pred_margin = p_home - p_away
                pred_total = p_home + p_away

                # Grade Out-of-Sample Predictions
                actual_winner = home if actual_margin > 0 else away
                pred_winner = home if pred_margin > 0 else away
                winner_correct = (actual_winner == pred_winner)

                # Spread Cover Evaluation
                actual_home_cover = (actual_margin > spread)
                pred_home_cover = (pred_margin > spread)
                cover_correct = (actual_home_cover == pred_home_cover)

                score_error = abs(p_home - actual_home) + abs(p_away - actual_away)

                sim_results.append({
                    "Matchup": matchup_label,
                    "Vegas Spread": f"{spread:+g}",
                    "Blind AI Projected Score": f"{away} {p_away} - {p_home} {home}",
                    "Actual Final Score": f"{away} {actual_away} - {actual_home} {home}",
                    "SU Winner Hit": "✅ Hit" if winner_correct else "❌ Miss",
                    "Spread Read": "✅ Correct Cover" if cover_correct else "❌ Wrong Side",
                    "Score Error (MAE)": f"{score_error / 2.0:.1f} pts"
                })

                prog_bar.progress((idx + 1) / len(hist_games))

            df_sim_results = pd.DataFrame(sim_results)
            st.success("Simulation Complete. Out-of-sample grading results:")
            st.dataframe(df_sim_results, hide_index=True, use_container_width=True)

            su_acc = (df_sim_results["SU Winner Hit"] == "✅ Hit").mean() * 100
            ats_acc = (df_sim_results["Spread Read"] == "✅ Correct Cover").mean() * 100

            k1, k2 = st.columns(2)
            k1.metric("Blind Outright Win Accuracy", f"{su_acc:.1f}%")
            k2.metric("Blind Spread Cover Accuracy", f"{ats_acc:.1f}%")
