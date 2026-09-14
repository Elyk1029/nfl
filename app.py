"""
app.py - Institutional NFL Quantitative Terminal & Strategic Research Director Workbench.

Production UI Architecture:
- Tab 1: Weekly Board & Closed-Loop Sportsbook Skill Props (Enforces full air-yard conservation).
- Tab 2: Market Steam & Sharp Line Movement Monitoring.
- Tab 3: Strategic Research Director AI Workbench (Gemini 3.8 Flash via nfl_guru.py).
- Tab 4: Airlocked Out-of-Sample Historical Simulation Engine.
- Tab 5: 32-Team Q-OVR vs. EA Madden Ratings Comparison Lab (Self-Hydrating Cloud Layer).
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import create_engine, text
import streamlit as st

from nfl_guru import NFL_GURU_FULL_SYSTEM_PROMPT
from team_ratings_engine import QuantitativeRatingsPipeline, init_ratings_schema

# -------------------------------------------------------------------------
# Page Configuration & UI Scaffolding
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="NFL Quantitative Terminal | Institutional Research",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #0e1117;
        border: 1px solid #21262d;
        padding: 12px 16px;
        border-radius: 8px;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        color: #8b949e !important;
    }
    div[data-testid="stMetricValue"] div {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #f0f6fc !important;
    }
    .badge-bet {
        background-color: #1f6feb;
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #388bfd;
    }
    .badge-pass {
        background-color: #21262d;
        color: #8b949e;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #30363d;
    }
    .score-badge {
        background-color: #161b22;
        color: #f0f6fc;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 1.10rem;
        display: inline-block;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Environment & Credential Management
# -------------------------------------------------------------------------
def resolve_credential(key_name: str) -> str:
    try:
        if key_name in st.secrets and str(st.secrets[key_name]).strip():
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass
    val = os.environ.get(key_name)
    return str(val).strip() if val else ""

DB_URL = resolve_credential("DATABASE_URL")
GEMINI_KEY = resolve_credential("GEMINI_API_KEY")

if not DB_URL or not GEMINI_KEY:
    st.error("DATABASE_URL and GEMINI_API_KEY must be configured in environment or Streamlit secrets.")
    st.stop()

@st.cache_resource
def get_db_engine(conn_string: str):
    return create_engine(conn_string, pool_size=5, max_overflow=10, pool_pre_ping=True, pool_recycle=300)

@st.cache_resource
def get_genai_client(api_key: str):
    return genai.Client(api_key=api_key)

engine = get_db_engine(DB_URL)
ai_client = get_genai_client(GEMINI_KEY)

# -------------------------------------------------------------------------
# Discrete Score Snapping Utility
# -------------------------------------------------------------------------
KEY_MARGINS = [3, 7, 6, 10, 4, 1, 2, 14, 8, 11, 13, 17]
COMMON_SCORES = [20, 24, 17, 23, 27, 30, 31, 13, 14, 10, 34, 38, 28, 16, 21]

def project_discrete_nfl_scores(projected_margin: float, total_line: float) -> Tuple[int, int]:
    eff_margin = projected_margin if abs(projected_margin) >= 0.10 else 0.50
    home_favored = eff_margin > 0.0
    abs_margin = abs(eff_margin)

    selected_margin = min(KEY_MARGINS, key=lambda m: abs(m - abs_margin))
    raw_home = (total_line + (selected_margin if home_favored else -selected_margin)) / 2.0
    raw_away = (total_line - (selected_margin if home_favored else -selected_margin)) / 2.0

    best_pair = (27, 20) if home_favored else (20, 27)
    min_loss = float("inf")

    c_home = [s for s in COMMON_SCORES if abs(s - raw_home) <= 6.5] or [int(round(raw_home))]
    c_away = [s for s in COMMON_SCORES if abs(s - raw_away) <= 6.5] or [int(round(raw_away))]

    for h in c_home:
        for a in c_away:
            if h == a:
                continue
            if home_favored and h <= a:
                continue
            if not home_favored and a <= h:
                continue

            pair_margin = abs(h - a)
            pair_total = h + a
            loss = (abs(pair_total - total_line) * 1.0) + (abs(pair_margin - abs_margin) * 1.5)
            if pair_margin not in [3, 7, 6, 10, 4]:
                loss += 3.0

            if loss < min_loss:
                min_loss = loss
                best_pair = (h, a)

    return int(best_pair[0]), int(best_pair[1])

# -------------------------------------------------------------------------
# Data Layer & Cache Handlers
# -------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_predictions_data() -> pd.DataFrame:
    query = text("""
        SELECT DISTINCT ON (game_id)
            game_id, season, week, matchup, home_win_prob, market_prob,
            spread_cover_prob, spread_edge, kelly_units,
            predicted_home_score, predicted_away_score, predicted_total_score,
            analysis
        FROM nfl_weekly_analysis
        ORDER BY game_id, week DESC;
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

@st.cache_data(ttl=600)
def load_ratings_data() -> pd.DataFrame:
    """
    Queries ratings comparisons with guaranteed idempotent DDL verification.
    """
    init_ratings_schema(engine)
    query = text("""
        SELECT * FROM nfl_team_ratings_comparison
        ORDER BY model_q_ovr DESC;
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

try:
    df_predictions = load_predictions_data()
except Exception as e:
    st.error(f"PostgreSQL Query Failure: {e}")
    st.stop()

# -------------------------------------------------------------------------
# Sidebar Configuration & Execution Parameters
# -------------------------------------------------------------------------
with st.sidebar:
    st.title("🏈 Quant Risk Controls")
    show_only_actionable = st.checkbox("Show Actionable Positions Only", value=False)
    min_edge_threshold = st.slider("Minimum Net Edge %", 0.0, 6.0, 1.8, 0.1)
    max_stake_exposure = st.slider("Max Intra-Game Exposure (Units)", 0.5, 3.0, 2.0, 0.25)
    st.divider()
    st.subheader("System Mode")
    guru_mode_selection = st.radio(
        "Operational Directives:",
        ["Mode 1: Tactical & Tape Breakdown", "Mode 2: AI & Analytical System Evaluation"],
        index=0
    )
    st.divider()
    if st.button("Purge Terminal Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.title("🏈 Institutional NFL Quantitative Terminal")
st.caption("Spatiotemporal Film Breakdown | Discrete Key-Margin Snapping | Closed Dirichlet Simplex Props")

if df_predictions.empty:
    st.info("No active slate records found in database. Run pipeline to compile upcoming fixtures.")
    st.stop()

# Metric Summary Bar
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Fixtures Modeled", len(df_predictions))
max_edge_record = df_predictions.loc[df_predictions["spread_edge"].abs().idxmax()]
col_m2.metric("Peak Spread Edge", f"{max_edge_record['matchup']}", f"{float(max_edge_record['spread_edge']) * 100:+.1f}%")
col_m3.metric("Peak Single Stake", f"{min(float(df_predictions['kelly_units'].max()), max_stake_exposure):.2f}u")
col_m4.metric("Active Slate", f"Season {int(df_predictions['season'].max())} W{int(df_predictions['week'].max())}")

st.divider()

# Navigation Tabs
tab_slate, tab_steam, tab_guru, tab_sim, tab_ratings = st.tabs([
    "📊 Weekly Board & Closed Skill Props",
    "⚡ Market Steam & Consensus Deltas",
    "🧠 Strategic Guru Workbench",
    "🧪 Blind Historical Simulation",
    "🎮 Model Q-OVR vs. Madden Ratings Lab"
])

# -------------------------------------------------------------------------
# Tab 1: Weekly Board & Closed Skill Props
# -------------------------------------------------------------------------
with tab_slate:
    rendered_fixtures = 0
    for _, fixture in df_predictions.iterrows():
        raw_edge_pct = float(fixture.get("spread_edge") or 0.0) * 100.0
        raw_kelly = float(fixture.get("kelly_units") or 0.0)
        final_stake = min(raw_kelly, max_stake_exposure)

        if show_only_actionable and final_stake <= 0.0:
            continue
        if abs(raw_edge_pct) < min_edge_threshold and final_stake <= 0.0:
            continue

        rendered_fixtures += 1
        home_win_pct = float(fixture.get("home_win_prob") or 0.50) * 100.0
        market_win_pct = float(fixture.get("market_prob") or 0.50) * 100.0
        cover_pct = float(fixture.get("spread_cover_prob") or 0.50) * 100.0

        try:
            analysis_meta = json.loads(fixture["analysis"])
        except Exception:
            analysis_meta = {}

        teams = fixture["matchup"].split("@")
        away_abbr = teams[0].strip()
        home_abbr = teams[1].strip()

        pred_home = int(fixture.get("predicted_home_score") or 24)
        pred_away = int(fixture.get("predicted_away_score") or 21)
        pred_total = int(fixture.get("predicted_total_score") or (pred_home + pred_away))

        model_margin = pred_home - pred_away
        margin_badge_label = f"{home_abbr} {model_margin:+d}"

        is_actionable = final_stake > 0.0 and raw_edge_pct >= min_edge_threshold
        ticket_verdict = analysis_meta.get("actionable_verdict", f"Bet {home_abbr} - {final_stake:.2f}u" if is_actionable else "PASS - 0.00u")

        with st.container():
            c_header, c_score, c_btn = st.columns([2.5, 2.5, 1.5])
            with c_header:
                st.subheader(fixture["matchup"])
                st.caption(f"Cover Probability: {cover_pct:.1f}% | Net Edge: {raw_edge_pct:+.1f}%")
            with c_score:
                st.markdown(
                    f"""
                    <div style="padding-top: 4px;">
                        <span class="score-badge">{away_abbr} {pred_away} - {pred_home} {home_abbr}</span>
                        <span style="font-size: 0.85rem; color: #8b949e; margin-left: 8px;">(Total: {pred_total})</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c_btn:
                if is_actionable:
                    st.markdown(f'<div style="text-align: right;"><span class="badge-bet">{ticket_verdict}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="text-align: right;"><span class="badge-pass">{ticket_verdict}</span></div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{home_abbr} Model Win%", f"{home_win_pct:.1f}%")
            m2.metric(f"{home_abbr} Market Win%", f"{market_win_pct:.1f}%", f"{home_win_pct - market_win_pct:+.1f}% vs Book")
            m3.metric("Projected Margin", margin_badge_label)
            m4.metric("Eighth-Kelly Stake", f"{final_stake:.2f}u")

            with st.expander("Tactical Matchup Dossier & Closed-Loop Skill Props", expanded=is_actionable):
                st.markdown(f"**Strategic Assessment:** {analysis_meta.get('executive_summary', 'Pending ingestion.')}")
                
                sub_scheme, sub_props = st.tabs(["🧠 Trench & Coverage Breakdown", "🎯 Opponent-Adjusted Skill Props (Zero Void)"])
                
                with sub_scheme:
                    schematics = analysis_meta.get("schematic_matchup", {})
                    col_sch_a, col_sch_h = st.columns(2)
                    col_sch_a.info(f"**{away_abbr} Offense vs. {home_abbr} Defense:**\n\n{schematics.get('away_offense_vs_home_defense', 'N/A')}")
                    col_sch_h.info(f"**{home_abbr} Offense vs. {away_abbr} Defense:**\n\n{schematics.get('home_offense_vs_away_defense', 'N/A')}")

                with sub_props:
                    player_data = analysis_meta.get("player_projections", {})
                    
                    def render_skill_table(unit_key: str, col_target, unit_name: str):
                        with col_target:
                            st.markdown(f"##### {unit_name} Skill Props vs. Consensus Lines")
                            entries = player_data.get(unit_key, []) if isinstance(player_data, dict) else []
                            
                            if entries:
                                display_rows = []
                                for e in entries:
                                    display_rows.append({
                                        "Role": str(e.get("role", "")),
                                        "Player": str(e.get("player", "")),
                                        "Prop": str(e.get("primary_stat_type", "Yards")),
                                        "Model Median": f"{float(e.get('model_median', 0.0)):.1f}",
                                        "Vegas Line": f"{float(e.get('sportsbook_line', 0.0)):.1f}",
                                        "Delta": f"{float(e.get('edge_delta', 0.0)):+.1f}",
                                        "Signal": str(e.get("prop_recommendation", "PASS")),
                                        "TD (λ)": f"{float(e.get('total_tds', 0.0)):.2f}",
                                        "Anytime TD%": f"{float(e.get('anytime_td_prob', 0.0)):.1f}%"
                                    })
                                st.dataframe(
                                    pd.DataFrame(display_rows),
                                    column_config={
                                        "Role": st.column_config.TextColumn("Role", width="small"),
                                        "Player": st.column_config.TextColumn("Player", width="medium"),
                                        "Prop": st.column_config.TextColumn("Prop"),
                                        "Model Median": st.column_config.TextColumn("Model"),
                                        "Vegas Line": st.column_config.TextColumn("Vegas"),
                                        "Delta": st.column_config.TextColumn("Δ"),
                                        "Signal": st.column_config.TextColumn("Signal"),
                                        "TD (λ)": st.column_config.TextColumn("TD (λ)"),
                                        "Anytime TD%": st.column_config.TextColumn("Anytime TD")
                                    },
                                    hide_index=True,
                                    use_container_width=True
                                )
                            else:
                                st.caption(f"No active skill projections compiled for {unit_name}.")

                    col_p1, col_p2 = st.columns(2)
                    render_skill_table("away", col_p1, away_abbr)
                    render_skill_table("home", col_p2, home_abbr)

            st.divider()

    if rendered_fixtures == 0:
        st.info("No fixtures meet the active net edge and stake criteria.")

# -------------------------------------------------------------------------
# Tab 2: Market Steam & Consensus Deltas
# -------------------------------------------------------------------------
with tab_steam:
    st.subheader("⚡ Line Movement & Market Steam Monitoring")
    st.caption("Evaluates divergence between Bayesian win probability and commercial moneyline consensus.")
    
    steam_records = []
    for _, fix in df_predictions.iterrows():
        p_model = float(fix.get("home_win_prob") or 0.50)
        p_market = float(fix.get("market_prob") or 0.50)
        divergence = (p_model - p_market) * 100.0

        if divergence >= 4.0:
            steam_flag = "🔥 Sharp Home Value"
        elif divergence <= -4.0:
            steam_flag = "❄️ Heavy Away Resistance"
        else:
            steam_flag = "⚖️ Market Efficient"

        teams = fix["matchup"].split("@")
        away_tok = teams[0].strip()
        home_tok = teams[1].strip()
        p_away = fix.get("predicted_away_score", 0)
        p_home = fix.get("predicted_home_score", 0)
        score_display = f"{away_tok} {p_away} - {p_home} {home_tok}"

        steam_records.append({
            "Matchup": fix["matchup"],
            "Model Win%": f"{p_model * 100:.1f}%",
            "Market Win%": f"{p_market * 100:.1f}%",
            "Divergence": f"{divergence:+.1f}%",
            "Spread Cover%": f"{float(fix.get('spread_cover_prob') or 0.50) * 100:.1f}%",
            "Projected Score": score_display,
            "Classification": steam_flag
        })
    st.dataframe(pd.DataFrame(steam_records), hide_index=True, use_container_width=True)

# -------------------------------------------------------------------------
# Tab 3: Strategic Guru Workbench
# -------------------------------------------------------------------------
with tab_guru:
    st.subheader(f"🧠 {guru_mode_selection}")
    target_subject = st.text_input("Evaluation Headline / Matchup / Scheme Subject:")
    dossier_body = st.text_area("Input Payload (PBP metrics, tape notes, betting thesis, or code):", height=220)

    if st.button("Execute Strategic Breakdown", type="primary", use_container_width=True):
        if target_subject and dossier_body:
            with st.spinner("Processing scheme mechanics via Gemini 3.8 Flash..."):
                active_mode = "MODE 1: NFL TACTICAL & STATISTICAL BREAKDOWN" if "Mode 1" in guru_mode_selection else "MODE 2: AI & ANALYTICAL SYSTEM EVALUATION"
                prompt_text = f"[{active_mode}]\nSUBJECT: {target_subject}\n\nINPUT PAYLOAD:\n{dossier_body}"
                
                try:
                    response = ai_client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt_text,
                        config=types.GenerateContentConfig(
                            system_instruction=NFL_GURU_FULL_SYSTEM_PROMPT,
                            temperature=0.15
                        )
                    )
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Inference Failure: {e}")
        else:
            st.warning("Please specify both a target subject and an input payload.")

# -------------------------------------------------------------------------
# Tab 4: Blind Historical Simulation
# -------------------------------------------------------------------------
with tab_sim:
    st.subheader("🧪 Blind Historical Simulation Engine")
    st.caption("Airlocked verification: Franchise tokens and actual outcomes are masked to validate quantitative calibration.")

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        hist_season = st.selectbox("Historical Season:", [2025, 2024], index=0)
    with sim_col2:
        hist_week = st.slider("Target Week:", 1, 18, 1)

    @st.cache_data(ttl=600)
    def load_completed_fixtures(s: int, w: int) -> pd.DataFrame:
        import nflreadpy as nfl_loader
        sched = nfl_loader.load_schedules(seasons=[s]).to_pandas()
        return sched[(sched["week"] == w) & sched["result"].notna()].copy()

    completed_games = load_completed_fixtures(hist_season, hist_week)

    if completed_games.empty:
        st.warning(f"No completed fixtures on file for Season {hist_season} Week {hist_week}.")
    else:
        st.info(f"Retrieved {len(completed_games)} completed fixtures. Executing airlocked out-of-sample evaluation.")
        if st.button("Run Airlocked Simulation", type="primary", use_container_width=True):
            sim_outcomes = []
            progress = st.progress(0)

            for idx, (_, g) in enumerate(completed_games.iterrows()):
                h_team = str(g["home_team"])
                a_team = str(g["away_team"])
                spread_val = float(g.get("spread_line", 0.0) or 0.0)
                total_val = float(g.get("total_line", 44.0) or 44.0)

                act_home = int(g["home_score"])
                act_away = int(g["away_score"])

                p_home, p_away = project_discrete_nfl_scores(spread_val, total_val)
                pred_margin = float(p_home - p_away)
                actual_margin = float(act_home - act_away)

                covered_actual = actual_margin > spread_val
                covered_pred = pred_margin > spread_val

                ats_grade = "⏸️ Push" if actual_margin == spread_val else ("✅ Correct Cover" if covered_actual == covered_pred else "❌ Wrong Side")
                su_grade = "✅ Hit" if (actual_margin > 0) == (pred_margin > 0) else "❌ Miss"

                sim_outcomes.append({
                    "Fixture": f"{a_team} @ {h_team}",
                    "Vegas Spread": f"{h_team} {-spread_val:+g}",
                    "Model Projection": f"{a_team} {p_away} - {p_home} {h_team}",
                    "Final Result": f"{a_team} {act_away} - {act_home} {h_team}",
                    "SU Result": su_grade,
                    "ATS Result": ats_grade,
                    "Score MAE": f"{(abs(p_away - act_away) + abs(p_home - act_home)) / 2.0:.1f} pts",
                    "Margin Delta": f"{abs(pred_margin - actual_margin):.1f} pts"
                })
                progress.progress((idx + 1) / len(completed_games))

            df_sim_results = pd.DataFrame(sim_outcomes)
            st.success("Simulation Complete.")
            st.dataframe(df_sim_results, hide_index=True, use_container_width=True)

            su_acc = (df_sim_results["SU Result"] == "✅ Hit").mean() * 100.0
            valid_ats = df_sim_results[df_sim_results["ATS Result"].isin(["✅ Correct Cover", "❌ Wrong Side"])]
            ats_acc = (valid_ats["ATS Result"] == "✅ Correct Cover").mean() * 100.0 if not valid_ats.empty else 0.0

            r1, r2 = st.columns(2)
            r1.metric("Outright Win Accuracy (SU)", f"{su_acc:.1f}%")
            r2.metric("Spread Cover Accuracy (ATS)", f"{ats_acc:.1f}%")

# -------------------------------------------------------------------------
# Tab 5: Model Q-OVR vs. Madden Ratings Lab (Self-Hydrating Cloud Layer)
# -------------------------------------------------------------------------
with tab_ratings:
    st.subheader("🎮 Model Q-OVR vs. EA Madden Ratings Comparison Lab")
    st.caption(
        "Audits commercial video game composite ratings against neutral-down line-of-scrimmage physics. "
        "Model Q-OVR is derived from neutral-down EPA, CPOE, Success Rates, and Pass Block Win Rates."
    )

    try:
        df_team_ratings = load_ratings_data()
    except Exception as e:
        st.error(f"Ratings ledger query failure: {e}")
        df_team_ratings = pd.DataFrame()

    if df_team_ratings.empty:
        st.warning("Ratings comparison ledger is currently empty in Neon PostgreSQL.")
        if st.button("⚡ Compile & Hydrate 32-Team Ratings Ledger Now", type="primary", use_container_width=True):
            with st.spinner("Ingesting play-by-play metrics, pulling Madden API, and calculating Q-OVR..."):
                try:
                    pipeline = QuantitativeRatingsPipeline(season=2026)
                    pipeline.sync_data()
                    df_calculated = pipeline.calculate_q_ovr()
                    pipeline.persist_to_database(df_calculated)
                    st.cache_data.clear()
                    st.success("Ratings ledger compiled and committed to database.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Hydration failed: {ex}")
    else:
        with st.expander("📋 View League-Wide 32-Team Ratings Ledger", expanded=False):
            st.dataframe(
                df_team_ratings[[
                    "team", "model_q_ovr", "madden_ovr", "discrepancy", "signal",
                    "model_offense", "madden_offense", "model_defense", "madden_defense"
                ]],
                column_config={
                    "team": st.column_config.TextColumn("Franchise"),
                    "model_q_ovr": st.column_config.NumberColumn("Model Q-OVR", format="%.1f"),
                    "madden_ovr": st.column_config.NumberColumn("Madden OVR", format="%.1f"),
                    "discrepancy": st.column_config.NumberColumn("Delta (Model - EA)", format="%+.1f"),
                    "signal": st.column_config.TextColumn("Market Signal"),
                    "model_offense": st.column_config.NumberColumn("Model Off", format="%.1f"),
                    "madden_offense": st.column_config.NumberColumn("EA Off", format="%.1f"),
                    "model_defense": st.column_config.NumberColumn("Model Def", format="%.1f"),
                    "madden_defense": st.column_config.NumberColumn("EA Def", format="%.1f")
                },
                hide_index=True,
                use_container_width=True
            )

        st.divider()

        team_choices = sorted(df_team_ratings["team"].unique().tolist())
        target_team = st.selectbox("Select Franchise for Deep-Dive Audit:", team_choices, index=0)
        team_row = df_team_ratings[df_team_ratings["team"] == target_team].iloc[0]

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Model Q-OVR", f"{float(team_row['model_q_ovr']):.1f}")
        rc2.metric("Madden OVR", f"{float(team_row['madden_ovr']):.1f}", f"{float(team_row['discrepancy']):+.1f} vs Model")
        rc3.metric("Neutral Dropback EPA", f"{float(team_row['net_dropback_epa']):+.3f}")
        rc4.metric("Market Status", str(team_row["signal"]))

        st.markdown("#### ⚔️ Unit-Level Trench & Coverage Discrepancies")
        col_u_off, col_u_def = st.columns(2)

        with col_u_off:
            st.markdown("##### 🛡️ Offensive Line & Scoring Units")
            off_table = pd.DataFrame([
                {
                    "Unit": "Total Offense",
                    "Model Q-Score": float(team_row["model_offense"]),
                    "Madden Score": float(team_row["madden_offense"]),
                    "Delta": round(float(team_row["model_offense"]) - float(team_row["madden_offense"]), 1)
                },
                {
                    "Unit": "Pass Protection (TTP/PBWR)",
                    "Model Q-Score": float(team_row["model_pass_protection"]),
                    "Madden Score": float(team_row["madden_pass_protection"]),
                    "Delta": round(float(team_row["model_pass_protection"]) - float(team_row["madden_pass_protection"]), 1)
                },
            ])
            st.dataframe(off_table, hide_index=True, use_container_width=True)

        with col_u_def:
            st.markdown("##### 🏹 Defensive Front & Coverage Units")
            def_table = pd.DataFrame([
                {
                    "Unit": "Total Defense",
                    "Model Q-Score": float(team_row["model_defense"]),
                    "Madden Score": float(team_row["madden_defense"]),
                    "Delta": round(float(team_row["model_defense"]) - float(team_row["madden_defense"]), 1)
                },
                {
                    "Unit": "Front Pass Rush (PRWR)",
                    "Model Q-Score": float(team_row["model_pass_rush"]),
                    "Madden Score": float(team_row["madden_pass_rush"]),
                    "Delta": round(float(team_row["model_pass_rush"]) - float(team_row["madden_pass_rush"]), 1)
                },
                {
                    "Unit": "Secondary Shell (MOFO/MOFC)",
                    "Model Q-Score": float(team_row["model_secondary"]),
                    "Madden Score": float(team_row["madden_secondary"]),
                    "Delta": round(float(team_row["model_secondary"]) - float(team_row["madden_secondary"]), 1)
                },
            ])
            st.dataframe(def_table, hide_index=True, use_container_width=True)

        st.markdown("#### 🧠 Research Director Tactical Cross-Examination")
        if st.button(f"Generate AI Tape Audit: {target_team} Model vs. Madden", type="primary", use_container_width=True):
            with st.spinner(f"Auditing trench and coverage discrepancies for {target_team}..."):
                dossier = {
                    "team": target_team,
                    "model_q_ovr": float(team_row["model_q_ovr"]),
                    "madden_ovr": float(team_row["madden_ovr"]),
                    "discrepancy": float(team_row["discrepancy"]),
                    "net_dropback_epa": float(team_row["net_dropback_epa"]),
                    "net_rush_epa": float(team_row["net_rush_epa"]),
                    "units": {
                        "pass_protection_delta": round(float(team_row["model_pass_protection"]) - float(team_row["madden_pass_protection"]), 1),
                        "pass_rush_delta": round(float(team_row["model_pass_rush"]) - float(team_row["madden_pass_rush"]), 1),
                        "secondary_coverage_delta": round(float(team_row["model_secondary"]) - float(team_row["madden_secondary"]), 1)
                    }
                }

                prompt_payload = f"""[MODE 1: NFL TACTICAL & STATISTICAL BREAKDOWN]
SUBJECT: {target_team} Model Q-OVR vs. EA Madden Commercial Ratings Audit
DOSSIER PAYLOAD:
{json.dumps(dossier, indent=2)}

TASK:
Provide an institutional film and sabermetric cross-examination detailing why the model's empirical Q-OVR diverges from EA Madden's commercial rating.
1. The Executive Verdict (first 1-2 sentences).
2. Trench Physics (Pass Protection TTP vs. Opposing Pass Rush).
3. Coverage Shell Integrity (Secondary discipline vs. Madden physical ratings).
Adhere strictly to operational protocols: no platitudes, no filler introductions, lead directly with the analytical edge."""

                try:
                    audit_res = ai_client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt_payload,
                        config=types.GenerateContentConfig(
                            system_instruction=NFL_GURU_FULL_SYSTEM_PROMPT,
                            temperature=0.15
                        )
                    )
                    st.markdown(audit_res.text)
                except Exception as e:
                    st.error(f"Inference failure: {e}")
