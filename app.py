import os
import json
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

st.set_page_config(
    page_title="Institutional NFL Quantitative Terminal",
    page_icon="🏈",
    layout="wide"
)

db_url = os.environ.get("DATABASE_URL")
if not db_url:
    st.error("DATABASE_URL environment variable is not configured.")
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

@st.cache_data(ttl=300)
def load_predictions():
    engine = get_db_engine()
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

st.title("🏈 Institutional NFL Quantitative Engine")
st.caption("Bivariate Skellam Score Modeling | Early-Down EPA Isolation | Eighth-Kelly Unit Allocations")

if df.empty:
    st.info("No prediction data currently available.")
    st.stop()

# Header Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Games Modeled", len(df))
max_edge_row = df.loc[df['spread_edge'].abs().idxmax()]
c2.metric("Top Model Edge", f"{max_edge_row['matchup']}", f"{max_edge_row['spread_edge']*100:+.1f}%")
c3.metric("Peak Recommended Stake", f"{df['kelly_units'].max():.2f}u")
c4.metric("Active Slate", f"Week {int(df['week'].max())}")

st.divider()

# Sidebar Risk Controls
st.sidebar.header("Execution Filters")
min_spread_edge = st.sidebar.slider("Minimum Edge Cutoff %", 0.0, 10.0, 1.5, 0.25)

# Render Matchup Cards
for _, row in df.iterrows():
    spread_edge_pct = (row.get('spread_edge') or 0.0) * 100
    if abs(spread_edge_pct) < min_spread_edge:
        continue

    home_win_pct = (row.get('home_win_prob') or 0.5) * 100
    market_win_pct = (row.get('market_prob') or 0.5) * 100
    cover_pct = (row.get('spread_cover_prob') or 0.5) * 100
    kelly = row.get('kelly_units') or 0.0

    with st.container():
        cols = st.columns([2.5, 1.5, 1.5, 1.5, 1.5])
        cols[0].subheader(row['matchup'])
        cols[1].metric("Calibrated Home Win", f"{home_win_pct:.1f}%")
        cols[2].metric("Devigged Consensus", f"{market_win_pct:.1f}%")
        cols[3].metric("Skellam Cover Prob", f"{cover_pct:.1f}%", f"{spread_edge_pct:+.1f}% Edge")
        cols[4].metric("Eighth-Kelly", f"{kelly:.2f}u")

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

            with st.expander("Tactical Matchup Breakdown & Prop Projections"):
                st.write(f"**Tactical Brief:** {analysis_data.get('executive_summary', '')}")
                
                tab_scheme, tab_props = st.tabs(["🧠 Trench & Coverage Clash", "🎯 Target Tree & Props"])
                with tab_scheme:
                    st.markdown("**Away Offense vs. Home Front & Shell**")
                    st.write(analysis_data['schematic_matchup'].get('away_offense_vs_home_defense', 'N/A'))
                    st.markdown("**Home Offense vs. Away Front & Shell**")
                    st.write(analysis_data['schematic_matchup'].get('home_offense_vs_away_defense', 'N/A'))
                
                with tab_props:
                    projections = analysis_data.get('player_projections', {})
                    c_away, c_home = st.columns(2)
                    teams = row['matchup'].split('@')
                    
                    def render_prop_matrix(team_key, team_name, col):
                        with col:
                            st.markdown(f"#### {team_name.strip()} Output Projections")
                            tp = projections.get(team_key, {})
                            
                            qb = tp.get("QB", {})
                            rb = tp.get("RB", {})
                            wr = tp.get("WR", {})
                            te = tp.get("TE", {})
                            
                            table_rows = [
                                {
                                    "Role": "QB", "Player": qb.get("player", "N/A"), 
                                    "Pass Yds": f"{qb.get('projected_pass_yards', 0.0):.1f}",
                                    "Pass TD": f"{qb.get('projected_pass_tds', 0.0):.1f}",
                                    "Rush Yds": f"{qb.get('projected_rush_yards', 0.0):.1f}",
                                    "Rec": "-", "Rec Yds": "-"
                                },
                                {
                                    "Role": "RB", "Player": rb.get("player", "N/A"),
                                    "Pass Yds": "-", "Pass TD": "-",
                                    "Rush Yds": f"{rb.get('projected_rush_yards', 0.0):.1f}",
                                    "Rec": f"{rb.get('projected_receptions', 0.0):.1f}",
                                    "Rec Yds": f"{rb.get('projected_rec_yards', 0.0):.1f}"
                                },
                                {
                                    "Role": "WR1", "Player": wr.get("player", "N/A"),
                                    "Pass Yds": "-", "Pass TD": "-", "Rush Yds": "-",
                                    "Rec": f"{wr.get('projected_receptions', 0.0):.1f}",
                                    "Rec Yds": f"{wr.get('projected_rec_yards', 0.0):.1f}"
                                },
                                {
                                    "Role": "TE1", "Player": te.get("player", "N/A"),
                                    "Pass Yds": "-", "Pass TD": "-", "Rush Yds": "-",
                                    "Rec": f"{te.get('projected_receptions', 0.0):.1f}",
                                    "Rec Yds": f"{te.get('projected_rec_yards', 0.0):.1f}"
                                }
                            ]
                            st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
                            st.caption(f"**QB Film Note:** _{qb.get('analysis', '')}_")

                    render_prop_matrix("away_team", teams[0], c_away)
                    render_prop_matrix("home_team", teams[1], c_home)
        else:
            with st.expander("Analysis Logs"):
                st.write(row['analysis'])

        st.divider()
