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

@st.cache_data(ttl=300)
def load_predictions():
    engine = create_engine(db_url)
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
    df = pd.read_sql(query, engine)
    return df

df = load_predictions()

st.title("🏈 Institutional NFL Quantitative Engine")
st.caption("Discrete Key-Number Modeling | EPA Garbage-Time Filtering | Quarter-Kelly Unit Allocations")

if df.empty:
    st.info("No prediction data currently available.")
    st.stop()

# Header Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Games Modeled", len(df))
max_edge_row = df.loc[df['spread_edge'].abs().idxmax()]
c2.metric("Top Spread Edge", f"{max_edge_row['matchup']}", f"{max_edge_row['spread_edge']*100:+.1f}%")
c3.metric("Top Recommended Size", f"{df['kelly_units'].max():.2f}u")
c4.metric("Week", f"Week {int(df['week'].max())}")

st.divider()

# Interactive Filter Controls
st.sidebar.header("Risk Configuration")
min_spread_edge = st.sidebar.slider("Minimum Spread Edge %", 0.0, 10.0, 1.5, 0.5)

# Render Games
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
        cols[1].metric("Model Home Win", f"{home_win_pct:.1f}%")
        cols[2].metric("Market Devigged", f"{market_win_pct:.1f}%")
        cols[3].metric("Spread Cover", f"{cover_pct:.1f}%", f"{spread_edge_pct:+.1f}%")
        cols[4].metric("Kelly Size", f"{kelly:.2f}u")

        # JSON Analysis Parsing
        try:
            analysis_data = json.loads(row['analysis'])
        except Exception:
            analysis_data = {"executive_summary": row['analysis']}

        if "schematic_matchup" in analysis_data:
            verdict = analysis_data.get('actionable_verdict', 'PASS')
            if "PASS" in verdict.upper():
                st.info(f"**Recommendation:** {verdict}")
            else:
                st.success(f"**Recommendation:** {verdict}")

            with st.expander("Schematic Clash, Personnel & Projections"):
                st.write(f"**Executive Summary:** {analysis_data.get('executive_summary', '')}")
                
                tab_scheme, tab_props = st.tabs(["🧠 Schematic Analysis", "🎯 Player Projections"])
                with tab_scheme:
                    st.markdown("#### Away Offense vs. Home Defense")
                    st.write(analysis_data['schematic_matchup'].get('away_offense_vs_home_defense', ''))
                    st.markdown("#### Home Offense vs. Away Defense")
                    st.write(analysis_data['schematic_matchup'].get('home_offense_vs_away_defense', ''))
                with tab_props:
                    c_away, c_home = st.columns(2)
                    with c_away:
                        st.markdown(f"**{row['matchup'].split('@')[0].strip()} Skill Projections**")
                        away_props = analysis_data['player_projections'].get('away_team', {})
                        st.write(f"• **QB:** {away_props.get('QB', 'N/A')}")
                        st.write(f"• **RB:** {away_props.get('RB', 'N/A')}")
                        st.write(f"• **WR:** {away_props.get('WR', 'N/A')}")
                    with c_home:
                        st.markdown(f"**{row['matchup'].split('@')[1].strip()} Skill Projections**")
                        home_props = analysis_data['player_projections'].get('home_team', {})
                        st.write(f"• **QB:** {home_props.get('QB', 'N/A')}")
                        st.write(f"• **RB:** {home_props.get('RB', 'N/A')}")
                        st.write(f"• **WR:** {home_props.get('WR', 'N/A')}")
        else:
            with st.expander("Legacy Text Summary"):
                st.write(row['analysis'])

        st.write("---")
