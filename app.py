import os
import json
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

st.set_page_config(
    page_title="NFL Quantitative Betting Engine",
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
            (home_win_prob - market_prob) as edge,
            analysis
        FROM nfl_weekly_analysis
        ORDER BY game_id, week DESC;
    """
    df = pd.read_sql(query, engine)
    return df

df = load_predictions()

st.title("🏈 NFL Quantitative Intelligence & Betting Engine")
st.markdown("Automated XGBoost probability engine paired with qualitative AI schematic & player projections.")

if df.empty:
    st.info("No prediction data currently available.")
    st.stop()

# Header Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Games Tracked", len(df))
max_edge_row = df.loc[df['edge'].abs().idxmax()]
col2.metric("Top Market Edge", f"{max_edge_row['matchup']}", f"{max_edge_row['edge']*100:+.1f}%")
col3.metric("Latest Ingested Week", f"Week {int(df['week'].max())}")

st.divider()

# Sidebar filter
st.sidebar.header("Filter Adjustments")
min_edge = st.sidebar.slider("Minimum Edge %", 0, 25, 0)

# Game Cards
for _, row in df.iterrows():
    edge = row['edge'] * 100
    if abs(edge) < min_edge:
        continue

    home_prob = row['home_win_prob'] * 100
    market_prob = row['market_prob'] * 100
    
    with st.container():
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        c1.subheader(row['matchup'])
        c2.metric("Model Home Win", f"{home_prob:.1f}%")
        c3.metric("Market Implied", f"{market_prob:.1f}%")
        c4.metric("Calculated Edge", f"{edge:+.1f}%", delta_color="normal" if edge >= 0 else "inverse")
        
        # JSON Parsing block with a fallback for older markdown data
        try:
            analysis_data = json.loads(row['analysis'])
        except Exception:
            analysis_data = {"executive_summary": row['analysis']} 
            
        if "schematic_matchup" in analysis_data:
            st.markdown(f"**Actionable Verdict:** {analysis_data.get('actionable_verdict', 'N/A')}")
            with st.expander("Read Matchup, Schematic Breakdown, & Projections"):
                st.markdown(f"**Executive Summary:** {analysis_data.get('executive_summary', '')}")
                
                t1, t2 = st.tabs(["🧠 Schematic Analysis", "📈 Player Projections"])
                with t1:
                    st.markdown("### Away Offense vs Home Defense")
                    st.write(analysis_data['schematic_matchup'].get('away_offense_vs_home_defense', ''))
                    st.markdown("### Home Offense vs Away Defense")
                    st.write(analysis_data['schematic_matchup'].get('home_offense_vs_away_defense', ''))
                with t2:
                    st.markdown("### Away Team Projections")
                    away_proj = analysis_data['player_projections'].get('away_team', {})
                    st.write(f"- **QB:** {away_proj.get('QB_projection', 'N/A')}")
                    st.write(f"- **RB:** {away_proj.get('RB_projection', 'N/A')}")
                    st.write(f"- **WR:** {away_proj.get('WR_projection', 'N/A')}")
                    
                    st.markdown("### Home Team Projections")
                    home_proj = analysis_data['player_projections'].get('home_team', {})
                    st.write(f"- **QB:** {home_proj.get('QB_projection', 'N/A')}")
                    st.write(f"- **RB:** {home_proj.get('RB_projection', 'N/A')}")
                    st.write(f"- **WR:** {home_proj.get('WR_projection', 'N/A')}")
        else:
            with st.expander("Read Matchup Analysis (Legacy Format)"):
                st.markdown(row['analysis'])
        st.write("---")
            st.markdown(row['analysis'])
        st.write("")
