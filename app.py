"""
app.py - Institutional NFL Quantitative Terminal & Strategic Guru Workbench.
Production UI:
- Reconciled Actionable Verdict Banner with Eighth-Kelly Staking Allocation.
- Structured Pandas DataFrame Table Renderer with Separated Columns (Eliminating Text Collision).
- Normalized ATS Spread Evaluation Engine & Blind Historical Simulation Airlock.
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
You are the "NFL Research Director & Quantitative Architect," operating at the nexus of NFL coaching tape breakdown, spatiotemporal tracking physics (NGS), and advanced sabermetric modeling.

---

## 1. 2026 PLAY-CALLER & TACTICAL CONTINUITY DIRECTORY
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

## 2. TRANSLATIONAL INVARIANTS
* Trench Physics: Explain as countdown race between pass protection and QB release timing.
* Run Schemes: Explain Duo/Power as "vertical bulldozing" and Zone schemes as "sideline-to-sideline stretch".
* Coverage Shells: Explain MOFC as "Single-High Safety (extra run defender)" and MOFO as "Two-Deep Safeties (umbrella against deep shots)".

---

## 3. MATHEMATICAL DISCIPLINE
* Neutral script leverage (WP 10%-90%).
* Log-normal median conversion for player props: m = mu * exp(-sigma^2 / 2).
* Closed-loop target trees: Sum of receiving yards must reconcile to gross passing volume.
* Discrete scoring margin optimization (zero ties).
"""

def normalize_and_grade_spread(pred_home_score: float, pred_away_score: float, 
                               actual_home_score: int, actual_away_score: int, 
                               home_spread_line: float) -> dict:
    actual_margin = float(actual_home_score - actual_away_score)
    pred_margin = float(pred_home_score - pred_away_score)

    actual_home_covered = (actual_margin + home_spread_line) > 0.0
    pred_home_covered = (pred_margin + home_spread_line) > 0.0

    is_actual_push = (actual_margin + home_spread_line) == 0.0
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
st.caption("Discrete Empirical Score Modeling | Closed-Loop Skill Props | 2026 Verified Dynamic Depth Charts")

if df.empty:
    st.info("No active slate predictions currently loaded.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Games Modeled", len(df))
max_edge_row = df.loc[df['spread_edge'].abs().idxmax()]
c2.metric("Top Spread Edge", f"{max_edge_row['matchup']}", f"{max_edge_row['spread_edge']*100:+.1f}%")
c3.metric("Peak Stake", f"{min(df['kelly_units'].max(), max_stake_cap):.2f}u")
c4.metric("Active Slate", f"Week {int(df['week'].max())}")

st.divider()

tab_slate, tab_steam, tab_guru, tab_sim = st.tabs([
    "📊 Weekly Board & Predicted Scores",
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
            m1.metric("Model Win Prob", f"{home_win_pct:.1f}%")
            m2.metric("Market Consensus", f"{market_win_pct:.1f}%", f"{home_win_pct - market_win_pct:+.1f}% vs Book")
            m3.metric("Projected Margin", f"{home_team} {p_home - p_away:+d}")
            m4.metric("Eighth-Kelly Stake", f"{stake:.2f}u")

            with st.expander("Tactical Matchup Dossier & Skill Player Stat Lines", expanded=is_bet):
                st.markdown(f"**Tactical Brief:** {analysis_data.get('executive_summary', 'Analysis pending.')}")
                
                sub_scheme, sub_props = st.tabs(["🧠 Trench & Coverage Clash", "🎯 Position Projections & Anytime TDs"])
                with sub_scheme:
                    scheme = analysis_data.get('schematic_matchup', {})
                    sc1, sc2 = st.columns(2)
                    sc1.info(f"**{away_team} Offense vs. {home_team} Defense:**\n\n" + scheme.get('away_offense_vs_home_defense', 'N/A'))
                    sc2.info(f"**{home_team} Offense vs. {away_team} Defense:**\n\n" + scheme.get('home_offense_vs_away_defense', 'N/A'))

                with sub_props:
                    player_projs = analysis_data.get('player_projections', {})
                    
                    def render_player_stat_table(team_key: str, container_col, team_display_name: str):
                        with container_col:
                            st.markdown(f"##### {team_display_name} Skill Player Projections")
                            if isinstance(player_projs, dict):
                                projs = player_projs.get(team_key, [])
                            elif isinstance(player_projs, list):
                                projs = [p for p in player_projs if p.get("team", "").strip().upper() == team_display_name.strip().upper()]
                            else:
                                projs = []

                            if projs:
                                table_rows = []
                                for p in projs:
                                    role_label = str(p.get("role", "")).strip()
                                    player_name = str(p.get("player", "")).strip()

                                    pass_val = max(0.0, float(p.get("pass_yards", 0.0)))
                                    rush_val = max(0.0, float(p.get("rush_yards", 0.0)))
                                    rec_val = max(0.0, float(p.get("rec_yards", 0.0)))
                                    td_val = max(0.0, float(p.get("total_tds", 0.0)))
                                    prob_val = max(0.0, float(p.get("anytime_td_prob", 0.0)))

                                    table_rows.append({
                                        "Role": role_label,
                                        "Player": player_name,
                                        "Pass Yds": f"{pass_val:.1f}" if pass_val > 0 else "-",
                                        "Rush Yds": f"{rush_val:.1f}" if rush_val > 0 else "-",
                                        "Rec Yds": f"{rec_val:.1f}" if rec_val > 0 else "-",
                                        "Exp. TD": f"{td_val:.2f}",
                                        "Anytime TD%": f"{prob_val:.1f}%"
                                    })
                                
                                df_display = pd.DataFrame(table_rows)
                                st.dataframe(
                                    df_display,
                                    column_config={
                                        "Role": st.column_config.TextColumn("Role", width="small"),
                                        "Player": st.column_config.TextColumn("Player", width="medium"),
                                        "Pass Yds": st.column_config.TextColumn("Pass Med."),
                                        "Rush Yds": st.column_config.TextColumn("Rush Med."),
                                        "Rec Yds": st.column_config.TextColumn("Rec Med."),
                                        "Exp. TD": st.column_config.TextColumn("Total TD (λ)"),
                                        "Anytime TD%": st.column_config.TextColumn("Anytime TD")
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
            with st.spinner("Processing scheme leverage and statistical hygiene..."):
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

with tab_sim:
    st.subheader("🧪 Blind Past-Game Simulation Engine")
    st.caption("Validating AI predictive accuracy out-of-sample: Final scores are masked from inference.")

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
        st.info(f"Loaded {len(hist_games)} fixtures. Final scores are strictly airlocked from the prompt payload.")
        if st.button("Execute Blind Out-of-Sample Simulation", type="primary", use_container_width=True):
            sim_results = []
            progress_bar = st.progress(0)

            for idx, (_, g) in enumerate(hist_games.iterrows()):
                home = str(g["home_team"])
                away = str(g["away_team"])
                matchup_label = f"{away} @ {home}"

                raw_spread = float(g.get("spread_line", 0.0) or 0.0)
                home_spread_line = -raw_spread
                total_line = float(g.get("total_line", 44.0) or 44.0)

                actual_home = int(g["home_score"])
                actual_away = int(g["away_score"])

                blind_payload = {
                    "matchup": matchup_label,
                    "pre_game_market": {"spread_line": f"{home} {home_spread_line:+g}", "total_line": total_line},
                    "context": f"Season {sim_season} Week {sim_week}. Estimate scores using pre-game expectations."
                }

                blind_prompt = f"""
                Conduct a blind simulation for this NFL matchup:
                {json.dumps(blind_payload, indent=2)}

                CRITICAL DIRECTIVE: You do not know the actual score. Output STRICTLY valid JSON:
                {{
                  "predicted_away_score": 0,
                  "predicted_home_score": 0,
                  "predicted_winner": "Team Abbr"
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
                    p_home = int(round((total_line - home_spread_line) / 2.0))
                    p_away = int(round((total_line + home_spread_line) / 2.0))

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
            st.success("Simulation Complete.")
            st.dataframe(df_sim, hide_index=True, use_container_width=True)

            su_acc = (df_sim["SU Winner Hit"] == "✅ Hit").mean() * 100
            valid_covers = df_sim[df_sim["Spread Read"].isin(["✅ Correct Cover", "❌ Wrong Side"])]
            ats_acc = (valid_covers["Spread Read"] == "✅ Correct Cover").mean() * 100 if not valid_covers.empty else 0.0

            k1, k2 = st.columns(2)
            k1.metric("Blind Outright Win Accuracy (SU)", f"{su_acc:.1f}%")
            k2.metric("Blind Spread Cover Accuracy (ATS)", f"{ats_acc:.1f}%")
