"""
app.py - Institutional NFL Quantitative Terminal & Strategic Research Director Workbench.

Production UI Architecture:
- Dynamic Temporal Slate Resolution: Automatically locks to active week post-Monday Night Football.
- Tab 1: Weekly Board & Closed-Loop Sportsbook Skill Props (Dirichlet Simplex Conservation).
- Tab 2: Market Steam & Sharp Line Movement Monitoring.
- Tab 3: Strategic Research Director AI Workbench (Gemini 2.5 Flash via nfl_guru.py).
- Tab 4: Airlocked Out-of-Sample Historical Simulation Engine (Discrete Poisson Convolution).
- Tab 5: Model Q-OVR vs. Database Ratings & Roster Lab (Secondary-Weighted Ratings & Rosters).
"""

from datetime import datetime, timezone
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
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
# Discrete Score Snapping & Simulation Engine
# -------------------------------------------------------------------------
KEY_MARGIN_LOG_PRIORS: Dict[int, float] = {
    3: 0.85, 7: 0.65, 6: 0.45, 10: 0.40, 4: 0.30, 14: 0.25, 1: 0.15, 2: 0.15
}

def generate_team_score_pmf(implied_points: float, rz_td_rate: float = 0.55, max_score: int = 58) -> np.ndarray:
    pmf = np.zeros(max_score + 1, dtype=np.float64)
    if implied_points <= 2.0:
        pmf[0] = 0.60
        pmf[2] = 0.10
        pmf[3] = 0.30
        return pmf

    ev_per_score = (rz_td_rate * 6.95) + ((1.0 - rz_td_rate) * 3.0)
    lambda_scores = max(0.6, implied_points / max(2.0, ev_per_score))

    p_td7 = rz_td_rate * 0.975
    p_td6 = rz_td_rate * 0.015
    p_td8 = rz_td_rate * 0.010
    p_fg3 = max(0.04, 1.0 - rz_td_rate - 0.005)
    p_safety2 = 0.005

    single_drive = np.zeros(9, dtype=np.float64)
    single_drive[2] = p_safety2
    single_drive[3] = p_fg3
    single_drive[6] = p_td6
    single_drive[7] = p_td7
    single_drive[8] = p_td8

    drive_pmf = np.zeros(max_score + 1, dtype=np.float64)
    drive_pmf[0] = 1.0

    for n_drives in range(11):
        prob_n = poisson.pmf(n_drives, lambda_scores)
        if prob_n >= 1e-6:
            pmf += prob_n * drive_pmf
        drive_pmf = np.convolve(drive_pmf, single_drive)[:max_score + 1]

    total_mass = np.sum(pmf)
    if total_mass > 0:
        pmf /= total_mass
    pmf[1] = 0.0
    return pmf

def project_dynamic_nfl_scores(
    projected_margin: float,
    total_line: float,
    home_rz_td_rate: float = 0.58,
    away_rz_td_rate: float = 0.52
) -> Tuple[int, int]:
    eff_margin = float(projected_margin)
    implied_home = max(6.0, (total_line + eff_margin) / 2.0)
    implied_away = max(6.0, (total_line - eff_margin) / 2.0)

    home_pmf = generate_team_score_pmf(implied_home, rz_td_rate=home_rz_td_rate)
    away_pmf = generate_team_score_pmf(implied_away, rz_td_rate=away_rz_td_rate)

    joint_matrix = np.outer(home_pmf, away_pmf)
    np.fill_diagonal(joint_matrix, joint_matrix.diagonal() * 0.05)

    home_favored = eff_margin > 0.10
    away_favored = eff_margin < -0.10
    abs_margin = abs(eff_margin)

    best_pair = (int(round(implied_home)), int(round(implied_away)))
    best_utility = -1e9

    for h in range(len(home_pmf)):
        for a in range(len(away_pmf)):
            prob = joint_matrix[h, a]
            if prob < 1e-5:
                continue

            if home_favored and h <= a:
                continue
            if away_favored and a <= h:
                continue

            score_margin = abs(h - a)
            score_total = h + a

            margin_err = abs(score_margin - abs_margin)
            total_err = abs(score_total - total_line)
            key_log_bonus = KEY_MARGIN_LOG_PRIORS.get(score_margin, 0.0)

            utility = math.log(prob) - (margin_err * 0.22) - (total_err * 0.08) + key_log_bonus

            if utility > best_utility:
                best_utility = utility
                best_pair = (int(h), int(a))

    return best_pair[0], best_pair[1]

# -------------------------------------------------------------------------
# Data Layer & Cache Handlers
# -------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_predictions_data(selected_week: Optional[int] = None) -> pd.DataFrame:
    if selected_week is not None:
        query = text("""
            SELECT DISTINCT ON (game_id)
                game_id, season, week, matchup, home_win_prob, market_prob,
                spread_cover_prob, spread_edge, kelly_units,
                predicted_home_score, predicted_away_score, predicted_total_score,
                analysis
            FROM nfl_weekly_analysis
            WHERE week = :target_week
            ORDER BY game_id;
        """)
        params = {"target_week": selected_week}
    else:
        query = text("""
            WITH max_slate AS (
                SELECT MAX(season) AS max_s, MAX(week) AS max_w
                FROM nfl_weekly_analysis
            )
            SELECT DISTINCT ON (a.game_id)
                a.game_id, a.season, a.week, a.matchup, a.home_win_prob, a.market_prob,
                a.spread_cover_prob, a.spread_edge, a.kelly_units,
                a.predicted_home_score, a.predicted_away_score, a.predicted_total_score,
                a.analysis
            FROM nfl_weekly_analysis a
            JOIN max_slate m ON a.season = m.max_s AND a.week = m.max_w
            ORDER BY a.game_id;
        """)
        params = {}

    with engine.connect() as conn:
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        return pd.read_sql(query, conn, params=params)

@st.cache_data(ttl=600)
def load_ratings_data() -> pd.DataFrame:
    init_ratings_schema(engine)
    query = text("""
        SELECT * FROM nfl_team_ratings_comparison
        ORDER BY model_q_ovr DESC;
    """)
    with engine.connect() as conn:
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        return pd.read_sql(query, conn)

@st.cache_data(ttl=600)
def load_rosters_data() -> pd.DataFrame:
    with engine.connect() as conn:
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        return pd.read_sql(text("SELECT * FROM nfl_team_rosters ORDER BY overall_rating DESC;"), conn)

@st.cache_data(ttl=300)
def get_available_weeks() -> List[int]:
    query = text("SELECT DISTINCT week FROM nfl_weekly_analysis ORDER BY week DESC;")
    try:
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            df = pd.read_sql(query, conn)
            return df["week"].tolist() if not df.empty else [1]
    except Exception:
        return [1]

available_weeks = get_available_weeks()

# -------------------------------------------------------------------------
# Sidebar Configuration & Execution Parameters
# -------------------------------------------------------------------------
with st.sidebar:
    st.title("🏈 Quant Risk Controls")
    current_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    st.caption(f"Engine Clock: {current_utc_str}")
    
    week_selection = st.selectbox(
        "Active Slate Filter:",
        options=["Auto-Detect (Latest Week)"] + [f"Week {w}" for w in available_weeks],
        index=0
    )
    selected_week_int = None if week_selection.startswith("Auto") else int(week_selection.split()[1])

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

try:
    df_predictions = load_predictions_data(selected_week=selected_week_int)
except Exception as e:
    st.error(f"PostgreSQL Query Failure: {e}")
    st.stop()

st.title("🏈 Institutional NFL Quantitative Terminal")
st.caption("Spatiotemporal Film Breakdown | Discrete Key-Margin Snapping | Closed Dirichlet Simplex Props")

if df_predictions.empty:
    st.info("No active slate records found for the selected week. Run `update_nfl.py` to ingest upcoming fixtures.")
    st.stop()

active_season = int(df_predictions['season'].max())
active_week = int(df_predictions['week'].max())

# Metric Summary Bar
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Fixtures Modeled", len(df_predictions))
max_edge_record = df_predictions.loc[df_predictions["spread_edge"].abs().idxmax()]
col_m2.metric("Peak Spread Edge", f"{max_edge_record['matchup']}", f"{float(max_edge_record['spread_edge']) * 100:+.1f}%")
col_m3.metric("Peak Single Stake", f"{min(float(df_predictions['kelly_units'].max()), max_stake_exposure):.2f}u")
col_m4.metric("Active Slate", f"Season {active_season} Week {active_week}")

st.divider()

# Navigation Tabs
tab_slate, tab_steam, tab_guru, tab_sim, tab_ratings = st.tabs([
    "📊 Weekly Board & Closed Skill Props",
    "⚡ Market Steam & Consensus Deltas",
    "🧠 Strategic Guru Workbench",
    "🧪 Blind Historical Simulation",
    "🎮 Model Q-OVR vs. Database Ratings Lab"
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
            with st.spinner("Processing scheme mechanics via Gemini 2.5 Flash..."):
                active_mode = "MODE 1: NFL TACTICAL & STATISTICAL BREAKDOWN" if "Mode 1" in guru_mode_selection else "MODE 2: AI & ANALYTICAL SYSTEM EVALUATION"
                prompt_text = f"[{active_mode}]\nSUBJECT: {target_subject}\n\nINPUT PAYLOAD:\n{dossier_body}"
                
                try:
                    response = ai_client.models.generate_content(
                        model="gemini-2.5-flash",
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

                p_home, p_away = project_dynamic_nfl_scores(spread_val, total_val)
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
# Tab 5: Model Q-OVR vs. Database Ratings & Roster Lab
# -------------------------------------------------------------------------
with tab_ratings:
    st.subheader("🎮 Model Q-OVR vs. Database Ratings & Roster Lab")
    st.caption(
        "Audits Secondary-Weighted Database composite ratings against neutral-down line-of-scrimmage physics. "
        "Individual roster ratings directly inform AI qualitative evaluations."
    )

    try:
        df_team_ratings = load_ratings_data()
        df_rosters = load_rosters_data()
    except Exception as e:
        st.error(f"Ledger or Roster query failure: {e}")
        df_team_ratings = pd.DataFrame()
        df_rosters = pd.DataFrame()

    if df_team_ratings.empty:
        st.warning("Ratings comparison ledger is currently empty in Neon PostgreSQL.")
        if st.button("⚡ Compile & Hydrate Database & Rosters Now", type="primary", use_container_width=True):
            with st.spinner("Parsing Secondary Weighted Rankings spreadsheet and committing to database..."):
                try:
                    pipeline = QuantitativeRatingsPipeline(season=2026, excel_path="Madden_27_Secondary_Weighted_Rankings.xlsx")
                    pipeline.sync_data()
                    df_calculated = pipeline.calculate_q_ovr()
                    pipeline.persist_to_database(df_calculated)
                    st.cache_data.clear()
                    st.success("Database successfully hydrated with team ratings and player rosters.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Hydration failed: {ex}")
    else:
        with st.expander("📋 View League-Wide 32-Team Database Ratings Ledger", expanded=False):
            st.dataframe(
                df_team_ratings[[
                    "team", "model_q_ovr", "madden_ovr", "discrepancy", "signal",
                    "model_offense", "madden_offense", "model_defense", "madden_defense"
                ]],
                column_config={
                    "team": st.column_config.TextColumn("Franchise"),
                    "model_q_ovr": st.column_config.NumberColumn("Model Q-OVR", format="%.1f"),
                    "madden_ovr": st.column_config.NumberColumn("Database Overall", format="%.1f"),
                    "discrepancy": st.column_config.NumberColumn("Delta (Model - DB)", format="%+.1f"),
                    "signal": st.column_config.TextColumn("Market Signal"),
                    "model_offense": st.column_config.NumberColumn("Model Off", format="%.1f"),
                    "madden_offense": st.column_config.NumberColumn("DB Off", format="%.1f"),
                    "model_defense": st.column_config.NumberColumn("Model Def", format="%.1f"),
                    "madden_defense": st.column_config.NumberColumn("DB Def", format="%.1f")
                },
                hide_index=True,
                use_container_width=True
            )

        st.divider()

        team_choices = sorted(df_team_ratings["team"].unique().tolist())
        target_team = st.selectbox("Select Franchise for Roster & Unit Deep-Dive:", team_choices, index=0)
        team_row = df_team_ratings[df_team_ratings["team"] == target_team].iloc[0]

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Model Q-OVR", f"{float(team_row['model_q_ovr']):.1f}")
        rc2.metric("Database Overall", f"{float(team_row['madden_ovr']):.1f}", f"{float(team_row['discrepancy']):+.1f} vs Model")
        rc3.metric("Net Dropback EPA", f"{float(team_row['net_dropback_epa']):+.3f}")
        rc4.metric("Market Status", str(team_row["signal"]))

        st.markdown("#### 🏈 Franchise Roster & Player Talent Pool")
        team_players = df_rosters[df_rosters["team"] == target_team].sort_values("overall_rating", ascending=False) if not df_rosters.empty else pd.DataFrame()
        
        if not team_players.empty:
            st.dataframe(
                team_players[["player_name", "position", "overall_rating", "archetype", "tier_status"]],
                column_config={
                    "player_name": st.column_config.TextColumn("Player Name", width="medium"),
                    "position": st.column_config.TextColumn("Position", width="small"),
                    "overall_rating": st.column_config.NumberColumn("Overall OVR"),
                    "archetype": st.column_config.TextColumn("Archetype", width="medium"),
                    "tier_status": st.column_config.TextColumn("Tier / Status")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.caption("No individual player records found for this franchise.")

        st.markdown("#### 🧠 Research Director Tactical Cross-Examination (AI-Informed)")
        if st.button(f"Generate AI Audit with Roster Context: {target_team}", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing roster depth and spatiotemporal metrics for {target_team}..."):
                top_stars = team_players.head(6).to_dict(orient="records") if not team_players.empty else []
                dossier = {
                    "team": target_team,
                    "model_q_ovr": float(team_row["model_q_ovr"]),
                    "database_overall": float(team_row["madden_ovr"]),
                    "net_dropback_epa": float(team_row["net_dropback_epa"]),
                    "top_roster_stars": top_stars
                }

                prompt_payload = f"""[MODE 1: NFL TACTICAL & STATISTICAL BREAKDOWN]
SUBJECT: {target_team} Model Q-OVR vs. Database Secondary-Weighted Overall Audit
DOSSIER PAYLOAD:
{json.dumps(dossier, indent=2)}

TASK:
Provide an institutional film and sabermetric cross-examination. Evaluate how top roster players drive or constrain the franchise's Database Overall relative to neutral-down EPA mechanics."""

                try:
                    audit_res = ai_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt_payload,
                        config=types.GenerateContentConfig(
                            system_instruction=NFL_GURU_FULL_SYSTEM_PROMPT,
                            temperature=0.15
                        )
                    )
                    st.markdown(audit_res.text)
                except Exception as e:
                    st.error(f"Inference failure: {e}")
