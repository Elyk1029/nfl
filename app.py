import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# Page Configuration
st.set_page_config(
    page_title="NFL Quantitative Betting Engine",
    page_icon="🏈",
    layout="wide"
)

# Database Connection
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
st.markdown("Automated XGBoost probability engine with qualitative AI edge analysis.")

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

# Game Cards
for _, row in df.iterrows():
    home_prob = row['home_win_prob'] * 100
    market_prob = row['market_prob'] * 100
    edge = row['edge'] * 100
    
    with st.container():
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        c1.subheader(row['matchup'])
        c2.metric("Model Home Win", f"{home_prob:.1f}%")
        c3.metric("Market Implied", f"{market_prob:.1f}%")
        c4.metric("Calculated Edge", f"{edge:+.1f}%", delta_color="normal" if edge >= 0 else "inverse")
        
        with st.expander("Read Matchup & Market Analysis"):
            st.markdown(row['analysis'])
        st.write("")
