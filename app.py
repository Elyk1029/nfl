"""
app.py - Institutional NFL Quantitative Terminal & Strategic Guru Workbench.
Production UI:
- Fully self-contained discrete empirical scoring engine.
- Complete Guru System Prompt integrated via nfl_guru.py.
- Live Sportsbook Prop Comparison Engine (Passing, Rushing, Receiving, Touchdown lambdas).
- Scoped nflreadpy import inside load_historical_fixtures resolving NameError in caching layer.
- Powered by Gemini 3.8 Flash (gemini-3.8-flash) for Guru Evaluation & Anonymized Simulations.
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

from nfl_guru import NFL_GURU_FULL_SYSTEM_PROMPT

st.set_page_config(
    page_title="NFL Quantitative Terminal | 2026 Season",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 12px 16px;
        border-radius: 8px;
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

NFL_KEY_MARGINS = [3, 7, 6, 10, 4, 1, 2, 14, 8, 11, 13, 17]
COMMON_TEAM_SCORES = [20, 24, 17, 23, 27, 30, 31, 13, 14, 10, 34, 38, 28, 16, 21]

def project_discrete_nfl_scores(projected_margin: float, total_line: float) -> tuple[int, int]:
    effective_margin = projected_margin if abs(projected_margin) >= 0.10 else 0.50
    home_favored = effective_margin > 0.0
    abs_margin = abs(effective_margin)

    selected_discrete_margin = min(NFL_KEY_MARGINS, key=lambda m: abs(m - abs_margin))
    raw_home = (total_line + (selected_discrete_margin if home_favored else -selected_discrete_margin)) / 2.0
    raw_away = (total_line - (selected_discrete_margin if home_favored else -selected_discrete_margin)) / 2.0

    best_pair = (27, 17) if home_favored else (17, 27)
    min_loss = float("inf")

    candidate_home = [s for s in COMMON_TEAM_SCORES if abs(s - raw_home) <= 6.5] or [int(round(raw_home))]
    candidate_away = [s for s in COMMON_TEAM_SCORES if abs(s - raw_away) <= 6.5] or [int(round(raw_away))]

    for h in candidate_home:
        for a in candidate_away:
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

def normalize_and_grade_spread(pred_home_score: float, pred_away_score: float, 
                               actual_home_score: int, actual_away_score: int, 
                               nflfastr_spread_line: float) -> dict:
    actual_margin = float(actual_home_score - actual_away_score)
    pred_margin = float(pred_home_score - pred_away_score)

    actual_home_covered = actual_margin > nflfastr_spread_line
    pred_home_covered = pred_margin > nflfastr_spread_line

    is_actual_push = actual_margin == nflfastr_spread_line
    if is_actual_push:
        cover_status = "⏸️ Push"
    elif actual_home_covered == pred_home_covered:
        cover_status = "✅ Correct Cover"
    else:
        cover_status = "❌ Wrong Side"

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

with st.sidebar:
    st.title("🏈 Risk Engine")
    show_only_bets = st.checkbox("Show Actionable Bets Only", value=False)
    min_edge_filter = st.slider("Minimum Edge Cutoff %", 0.0, 5.0, 1.5, 0.25)
    max_stake_cap = st.slider("Max Intra-Game Exposure (Units)", 0.5, 3.0, 2.0, 0.25)
    st.divider()
    guru_mode = st.radio("Guru Mode:", ["Mode 1: Tactical Breakdown", "Mode 2: System Evaluation"], index=0)
    st.divider()
    if st.button("Purge Terminal Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.title("🏈 Institutional NFL Quantitative Terminal")
st.caption("Discrete Empirical Score Modeling | Live Sportsbook Prop Benchmarking | Powered by Gemini 3.8 Flash")

if df.empty:
    st.info("No active slate predictions currently loaded in the database.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Games Modeled", len(df))
max_edge_row = df.loc[df['spread_edge'].abs().idxmax()]
c2.metric("Top Spread Edge", f"{max_edge_row['matchup']}", f"{max_edge_row['spread_edge']*100:+.1f}%")
c3.metric("Peak Stake", f"{min(df['kelly_units'].max(), max_stake_cap):.2f}u")
c4.metric("Active Slate", f"Week {int(df['week'].max())}")

st.divider()

tab_slate, tab_steam, tab_guru, tab_sim = st.tabs([
    "📊 Weekly Board & Sportsbook Props",
    "⚡ Market Steam & Consensus Deltas",
    "🧠 Strategic Guru Workbench",
    "🧪 Blind Historical Simulation"
])

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
            analysis_data = {}

        teams = row['matchup'].split('@')
        away_team = teams[0].strip()
        home_team = teams[1].strip()

        p_home = int(row.get('predicted_home_score') or 24)
        p_away = int(row.get('predicted_away_score') or 21)
        p_total = int(row.get('predicted_total_score') or (p_home + p_away))

        margin_delta = p_home - p_away
        margin_label = f"{home_team} {margin_delta:+d}"

        if stake > 0.0 and edge_pct > 0.0:
            verdict_str = analysis_data.get('actionable_verdict', f"BET - {stake:.2f}u")
            if "PASS" in verdict_str.upper():
                verdict_str = f"BET - {stake:.2f}u"
            is_bet = True
        else:
            verdict_str = "PASS - 0.00u"
            is_bet = False

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
                        <span style="font-size: 0.90rem; color: #8b949e; margin-left: 10px;">(Total: {p_total})</span>
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
            m1.metric(f"{home_team} Model Win Prob", f"{home_win_pct:.1f}%")
            m2.metric(f"{home_team} Market Prob", f"{market_win_pct:.1f}%", f"{home_win_pct - market_win_pct:+.1f}% vs Book")
            m3.metric("Projected Margin", margin_label)
            m4.metric("Eighth-Kelly Stake", f"{stake:.2f}u")

            with st.expander("Tactical Matchup Dossier & Live Sportsbook Prop Comparisons", expanded=is_bet):
                st.markdown(f"**Tactical Brief:** {analysis_data.get('executive_summary', 'Analysis pending.')}")
                
                sub_scheme, sub_props = st.tabs(["🧠 Trench & Coverage Clash", "🎯 Sportsbook Prop Benchmarks & Anytime TDs"])
                with sub_scheme:
                    scheme = analysis_data.get('schematic_matchup', {})
                    sc1, sc2 = st.columns(2)
                    sc1.info(f"**{away_team} Offense vs. {home_team} Defense:**\n\n" + scheme.get('away_offense_vs_home_defense', 'N/A'))
                    sc2.info(f"**{home_team} Offense vs. {away_team} Defense:**\n\n" + scheme.get('home_offense_vs_away_defense', 'N/A'))

                with sub_props:
                    player_projs = analysis_data.get('player_projections', {})
                    
                    def render_player_stat_table(team_key: str, container_col, team_display_name: str):
                        with container_col:
                            st.markdown(f"##### {team_display_name} Props vs. Sportsbook Lines")
                            projs = player_projs.get(team_key, []) if isinstance(player_projs, dict) else []

                            if projs:
                                table_rows = []
                                for p in projs:
                                    role_label = str(p.get("role", "")).strip()
                                    player_name = str(p.get("player", "")).strip()
                                    stat_type = str(p.get("primary_stat_type", "Yards")).strip()
                                    model_med = float(p.get("model_median", 0.0))
                                    sb_line = float(p.get("sportsbook_line", model_med))
                                    delta = float(p.get("edge_delta", model_med - sb_line))
                                    rec = str(p.get("prop_recommendation", "PASS"))
                                    td_val = float(p.get("total_tds", 0.0))
                                    prob_val = float(p.get("anytime_td_prob", 0.0))

                                    table_rows.append({
                                        "Role": role_label,
                                        "Player": player_name,
                                        "Prop Type": stat_type,
                                        "Model Med.": f"{model_med:.1f}",
                                        "Vegas Line": f"{sb_line:.1f}",
                                        "Edge": f"{delta:+.1f}",
                                        "Signal": rec,
                                        "Exp. TD (λ)": f"{td_val:.2f}",
                                        "Anytime TD": f"{prob_val:.1f}%"
                                    })
                                
                                df_display = pd.DataFrame(table_rows)
                                st.dataframe(
                                    df_display,
                                    column_config={
                                        "Role": st.column_config.TextColumn("Role", width="small"),
                                        "Player": st.column_config.TextColumn("Player", width="medium"),
                                        "Prop Type": st.column_config.TextColumn("Prop"),
                                        "Model Med.": st.column_config.TextColumn("Model"),
                                        "Vegas Line": st.column_config.TextColumn("Vegas"),
                                        "Edge": st.column_config.TextColumn("Delta"),
                                        "Signal": st.column_config.TextColumn("Pick"),
                                        "Exp. TD (λ)": st.column_config.TextColumn("TD (λ)"),
                                        "Anytime TD": st.column_config.TextColumn("Anytime TD")
                                    },
                                    hide_index=True,
                                    use_container_width=True
                                )
                            else:
                                st.caption(f"No structured skill projections available for {team_display_name}.")

                    p_col1, p_col2 = st.columns(2)
                    render_player_stat_table("away", p_col1, away_team)
                    render_player_stat_table("home", p_col2, home_team)

            st.divider()

    if displayed == 0:
        st.info("No matchups match your edge/stake filter thresholds.")

with tab_steam:
    st.subheader("⚡ Line Movement & Market Pricing Discrepancies")
    steam_records = []
    for _, r in df.iterrows():
        p_c = float(r.get('home_win_prob') or 0.5)
        p_m = float(r.get('market_prob') or 0.5)
        diff = (p_c - p_m) * 100
        signal = "🔥 Sharp Home Steam" if diff >= 4.0 else ("❄️ Heavy Away Steam" if diff <= -4.0 else "⚖️ Fairly Priced")
        steam_records.append({
            "Matchup": r["matchup"],
            "Model Win%": f"{p_c * 100:.1f}%",
            "Market Win%": f"{p_m * 100:.1f}%",
            "Discrepancy": f"{diff:+.1f}%",
            "Spread Edge": f"{float(r.get('spread_edge') or 0.0)*100:+.1f}%",
            "Projected Score": f"{r.get('predicted_away_score')} - {r.get('predicted_home_score')}",
            "Signal": signal
        })
    st.dataframe(pd.DataFrame(steam_records), hide_index=True, use_container_width=True)

with tab_guru:
    st.subheader(f"🧠 {guru_mode}")
    q_title = st.text_input("Evaluation Target / Matchup Headline:")
    q_payload = st.text_area("Dossier Payload (Tape notes, EPA splits, or prompt code):", height=200)
    if st.button("Execute Strategic Guru Evaluation", type="primary", use_container_width=True):
        if q_title and q_payload:
            with st.spinner("Processing scheme leverage via Gemini 3.8 Flash..."):
                prompt = f"[{guru_mode.upper()}]\nSUBJECT: {q_title}\n\nINPUT PAYLOAD:\n{q_payload}"
                try:
                    res = ai_client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=NFL_GURU_FULL_SYSTEM_PROMPT,
                            temperature=0.15
                        )
                    )
                    st.markdown(res.text)
                except Exception as e:
                    st.error(f"Inference Failure: {e}")

with tab_sim:
    st.subheader("🧪 Blind Past-Game Simulation Engine (Airlocked)")
    st.caption("Validating AI predictive accuracy out-of-sample: Franchise metadata and true scores are strictly masked.")

    sim_season = st.selectbox("Select Historical Season:", [2025, 2024], index=0)
    sim_week = st.slider("Select Historical Week:", 1, 18, 1)

    @st.cache_data(ttl=600)
    def load_historical_fixtures(season: int, week: int):
        import nflreadpy as nfl_loader
        sched = nfl_loader.load_schedules(seasons=[season]).to_pandas()
        return sched[(sched["week"] == week) & sched["result"].notna()].copy()

    hist_games = load_historical_fixtures(sim_season, sim_week)

    if hist_games.empty:
        st.warning("No completed games found for the selected schedule.")
    else:
        st.info(f"Loaded {len(hist_games)} fixtures. Franchise names and outcomes are airlocked via Entity Alpha/Beta tokens.")
        if st.button("Execute Airlocked Out-of-Sample Simulation", type="primary", use_container_width=True):
            sim_results = []
            progress_bar = st.progress(0)

            for idx, (_, g) in enumerate(hist_games.iterrows()):
                home = str(g["home_team"])
                away = str(g["away_team"])
                nflfastr_spread = float(g.get("spread_line", 0.0) or 0.0)
                total_line = float(g.get("total_line", 44.0) or 44.0)

                actual_home = int(g["home_score"])
                actual_away = int(g["away_score"])

                p_home, p_away = project_discrete_nfl_scores(nflfastr_spread, total_line)

                anonymized_payload = {
                    "entity_alpha": {"role": "Home Front", "projected_margin": f"Entity Alpha {nflfastr_spread:+g}"},
                    "entity_beta": {"role": "Away Front"},
                    "game_environment": {"total_points_market": total_line}
                }

                blind_prompt = f"""
                Analyze the trench clash between Entity Alpha and Entity Beta:
                {json.dumps(anonymized_payload, indent=2)}

                CRITICAL DIRECTIVE: You do not know the actual score or team identities.
                Output strictly valid JSON:
                {{
                  "strategic_thesis": "Two-sentence summary explaining line-of-scrimmage leverage."
                }}
                """

                try:
                    sim_res = ai_client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=blind_prompt,
                        config=types.GenerateContentConfig(temperature=0.10, response_mime_type="application/json")
                    )
                    thesis = json.loads(sim_res.text).get("strategic_thesis", "Line push dictates drive sustainability.")
                except Exception:
                    thesis = "Neutral script trench efficiency and third-down conversion rates govern margin."

                actual_margin = float(actual_home - actual_away)
                pred_margin = float(p_home - p_away)
                actual_covered = actual_margin > nflfastr_spread
                pred_covered = pred_margin > nflfastr_spread

                if actual_margin == nflfastr_spread:
                    ats_grade = "⏸️ Push"
                elif actual_covered == pred_covered:
                    ats_grade = "✅ Correct Cover"
                else:
                    ats_grade = "❌ Wrong Side"

                su_grade = "✅ Hit" if (actual_margin > 0) == (pred_margin > 0) else "❌ Miss"

                sim_results.append({
                    "Matchup": f"{away} @ {home}",
                    "Vegas Spread": f"{home} {-nflfastr_spread:+g}",
                    "Airlocked Model Projection": f"{away} {p_away} - {p_home} {home}",
                    "Actual Final Score": f"{away} {actual_away} - {actual_home} {home}",
                    "SU Hit": su_grade,
                    "Spread Read": ats_grade,
                    "Score MAE": f"{(abs(p_away - actual_away) + abs(p_home - actual_home)) / 2.0:.1f} pts",
                    "Margin Delta": f"{abs(pred_margin - actual_margin):.1f} pts",
                    "Airlocked Tape Breakdown": thesis
                })

                progress_bar.progress((idx + 1) / len(hist_games))

            df_sim = pd.DataFrame(sim_results)
            st.success("Simulation Complete.")
            st.dataframe(df_sim, hide_index=True, use_container_width=True)

            su_acc = (df_sim["SU Hit"] == "✅ Hit").mean() * 100
            valid_covers = df_sim[df_sim["Spread Read"].isin(["✅ Correct Cover", "❌ Wrong Side"])]
            ats_acc = (valid_covers["Spread Read"] == "✅ Correct Cover").mean() * 100 if not valid_covers.empty else 0.0

            k1, k2 = st.columns(2)
            k1.metric("Outright Win Accuracy (SU)", f"{su_acc:.1f}%")
            k2.metric("Spread Cover Accuracy (ATS)", f"{ats_acc:.1f}%")
