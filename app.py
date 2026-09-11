"""
app.py - Institutional NFL Quantitative Terminal & Strategic Guru Workbench.
Enhanced UI/UX edition: Accessible visual hierarchy, plain-English tooltips,
and card-based matchup containers for general users.
"""
import os
import json
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="NFL Quant Terminal | 2026 Season",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for crisp, modern, scannable cards and badges
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1a1e24;
        border: 1px solid #2d3748;
        padding: 12px 16px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #a0aec0 !important;
    }
    div[data-testid="stMetricValue"] div {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #f7fafc !important;
    }
    
    /* Action Badges */
    .badge-bet {
        background-color: #276749;
        color: #c6f6d5;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #38a169;
    }
    .badge-pass {
        background-color: #2d3748;
        color: #cbd5e0;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #4a5568;
    }
    .badge-edge {
        background-color: #2b6cb0;
        color: #bee3f8;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Matchup Card Wrapper */
    .matchup-card {
        background-color: #14181d;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 20px;
        transition: border-color 0.2s ease-in-out;
    }
    .matchup-card:hover {
        border-color: #4a5568;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Credential Resolution (.streamlit/secrets.toml -> os.environ)
# ---------------------------------------------------------
def resolve_credential(key_name: str) -> str:
    try:
        if key_name in st.secrets and str(st.secrets[key_name]).strip():
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass

    env_val = os.environ.get(key_name)
    if env_val and str(env_val).strip():
        return str(env_val).strip()

    return ""

db_url = resolve_credential("DATABASE_URL")
api_key = resolve_credential("GEMINI_API_KEY")

if not db_url or not api_key:
    missing = []
    if not db_url: missing.append("DATABASE_URL (Neon PostgreSQL)")
    if not api_key: missing.append("GEMINI_API_KEY (Google GenAI)")

    st.error(f"⚠️ Missing Credentials: {', '.join(missing)}")
    st.info("Configure these credentials in `.streamlit/secrets.toml` or export them as shell environment variables.")
    st.stop()

@st.cache_resource
def get_db_engine(connection_string: str):
    return create_engine(connection_string, pool_size=5, max_overflow=10, pool_pre_ping=True, pool_recycle=300)

@st.cache_resource
def get_genai_client(key: str):
    return genai.Client(api_key=key)

engine = get_db_engine(db_url)
ai_client = get_genai_client(api_key)

# ---------------------------------------------------------
# 11/10 System Instruction: Research Director with Accessible Output
# ---------------------------------------------------------
GURU_SYSTEM_INSTRUCTION = """
# ROLE & IDENTITY
You are the "NFL Research Director & Quantitative Architect," operating at the nexus of NFL coaching tape breakdown, spatiotemporal tracking physics (NGS), and advanced sabermetric modeling.

Your dual mandate:
1. Deliver sharp, accessible, and analytically grounded NFL football breakdowns that translate complex film mechanics into intuitive, plain-English insights for an everyday sports fan.
2. Serve as an expert AI evaluator: Continuously audit user-submitted AI prompts, analytical frameworks, statistical models, and projection logic to eliminate statistical noise, correct proxy errors, and enforce production-grade quantitative rigor.

---

## 1. 2026 PLAY-CALLER & TACTICAL CONTINUITY DIRECTORY
Never analyze teams by helmet logos or legacy play-callers. Reflect confirmed 2026 leadership:
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

## 2. TRANSLATIONAL INVARIANTS (MAKING IT ACCESSIBLE)
* Trench Physics: Explain as a countdown race between offensive line pass protection and QB release timing.
* Run Schemes: Explain Duo/Power as "vertical bulldozing" and Zone schemes as "sideline-to-sideline stretch".
* Coverage Shells: Explain MOFC as "Single-High Safety (extra defender in the run box)" and MOFO as "Two-Deep Safeties (umbrella against deep passes)".

---

## 3. MATHEMATICAL DISCIPLINE
* Neutral script EPA filtering (Win Probability 10%-90%).
* Log-normal median player prop pricing: m = mu * exp(-sigma^2 / 2).
* Discrete key-number margin clustering (3, 7, 6, 10, 4, 14) and Eighth-Kelly sizing with push rate subtraction.
* Epistemic anti-hallucination: Never fabricate decimal precision stats if exact tracking data is absent.
"""

# ---------------------------------------------------------
# Database Ingestion Layer
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def load_predictions():
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
        data = pd.read_sql(query, conn)
    return data

try:
    df = load_predictions()
except Exception as e:
    st.error(f"Database Ingestion Error: {e}")
    st.stop()

# ---------------------------------------------------------
# Sidebar Controls & User Education
# ---------------------------------------------------------
with st.sidebar:
    st.title("🏈 Terminal Settings")
    
    st.markdown("### 🎯 Slate Filters")
    show_only_bets = st.checkbox("Show Actionable Bets Only (Hide Passes)", value=False)
    min_edge_filter = st.slider(
        "Minimum Edge %", 
        0.0, 5.0, 1.5, 0.25,
        help="Filters for games where the model's cover probability exceeds the sportsbook breakeven hurdle (52.38%) by at least this margin."
    )
    max_stake_cap = st.slider(
        "Max Single-Game Stake (Units)", 
        0.5, 3.0, 2.0, 0.25,
        help="Hard ceiling on recommended risk exposure per contest."
    )

    st.divider()
    st.markdown("### 🧠 Operational Mode")
    guru_mode = st.radio(
        "Workbench Focus:",
        ["Mode 1: Tactical Breakdown", "Mode 2: System Evaluation"],
        index=0,
        help="Mode 1 generates broadcast-ready film analysis. Mode 2 performs rigorous technical audits on prompts, code, and statistical features."
    )

    st.divider()
    with st.expander("📖 Metric Cheat Sheet"):
        st.markdown("""
        * **Model Win%:** Independent game simulation win probability.
        * **Market Implied%:** Vegas odds converted to true probability with house juice removed.
        * **Model Edge:** How much our projection beats the break-even line (+52.4%).
        * **Eighth-Kelly:** Mathematically optimal, risk-averse unit staking size.
        """)

    if st.button("🔄 Refresh Live Board", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------
# Dashboard Header & Global Summary Bar
# ---------------------------------------------------------
st.title("🏈 NFL Quantitative Betting Terminal")
st.markdown("##### Real-Time Scheme Leverage, Opponent-Adjusted EPA & Bayesian Median Lines")

if df.empty:
    st.warning("No active games available in the database.")
    st.stop()

# Aggregate Metrics
total_games = len(df)
active_bets = len(df[df['kelly_units'] > 0.0])
top_edge_row = df.loc[df['spread_edge'].abs().idxmax()]
top_stake_row = df.loc[df['kelly_units'].idxmax()]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(
    "Active Slate", 
    f"Week {int(df['week'].max())}", 
    f"{total_games} Total Matchups",
    help="Current NFL regular season scheduling block."
)
kpi2.metric(
    "Actionable Bets", 
    f"{active_bets} Opportunities",
    f"{(active_bets/total_games)*100:.0f}% of Board",
    help="Number of matchups where edge exceeds our risk hurdle."
)
kpi3.metric(
    "Top Market Edge", 
    f"{top_edge_row['matchup']}", 
    f"{top_edge_row['spread_edge']*100:+.1f}% Edge",
    help="Highest statistical variance between model and Vegas consensus line."
)
kpi4.metric(
    "Peak Stake", 
    f"{min(top_stake_row['kelly_units'], max_stake_cap):.2f} Units", 
    f"{top_stake_row['matchup']}",
    help="Highest conviction capital allocation calculated via Eighth-Kelly sizing."
)

st.divider()

# ---------------------------------------------------------
# Tabs: Dashboard View, Market Steam, and AI Workbench
# ---------------------------------------------------------
tab_board, tab_steam, tab_workbench = st.tabs([
    "📋 Matchup Board & Player Props",
    "⚡ Market Steam & Consensus Deltas",
    "🧠 Strategic Guru Workbench"
])

# =========================================================
# TAB 1: MATCHUP BOARD & PROPS
# =========================================================
with tab_board:
    displayed_count = 0

    for _, row in df.iterrows():
        spread_edge_pct = (row.get('spread_edge') or 0.0) * 100
        raw_kelly = float(row.get('kelly_units') or 0.0)
        allocated_stake = min(raw_kelly, max_stake_cap)
        
        # Apply Sidebar Filters
        if show_only_bets and allocated_stake <= 0.0:
            continue
        if abs(spread_edge_pct) < min_edge_filter:
            continue

        displayed_count += 1
        home_win_pct = (row.get('home_win_prob') or 0.5) * 100
        market_win_pct = (row.get('market_prob') or 0.5) * 100
        cover_pct = (row.get('spread_cover_prob') or 0.5) * 100

        # Parse JSON Analysis
        try:
            analysis_data = json.loads(row['analysis'])
        except Exception:
            analysis_data = {"executive_summary": row['analysis'], "player_projections": []}

        verdict_str = analysis_data.get('actionable_verdict', 'PASS - 0.00u')
        is_bet = "BET" in verdict_str.upper()

        # Render Matchup Container Card
        with st.container():
            # Card Header
            header_col1, header_col2 = st.columns([3, 1])
            with header_col1:
                st.markdown(f"### {row['matchup']}")
            with header_col2:
                if is_bet:
                    st.markdown(f'<div style="text-align: right;"><span class="badge-bet">🎯 {verdict_str}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="text-align: right;"><span class="badge-pass">⏸️ {verdict_str}</span></div>', unsafe_allow_html=True)

            # Metric Strip
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Model Win Probability", 
                f"{home_win_pct:.1f}%",
                help="Our simulation's projected outright win percentage for the home team."
            )
            m2.metric(
                "Vegas Implied Probability", 
                f"{market_win_pct:.1f}%", 
                f"{home_win_pct - market_win_pct:+.1f}% vs Book",
                help="Consensus sportsbook implied win probability with the vigorish removed."
            )
            m3.metric(
                "Cover Probability", 
                f"{cover_pct:.1f}%", 
                f"{spread_edge_pct:+.1f}% Net Edge",
                help="Probability of our recommended side covering the point spread."
            )
            m4.metric(
                "Recommended Stake", 
                f"{allocated_stake:.2f} Units",
                f"{'Active Allocation' if allocated_stake > 0 else 'No Bet'}",
                help="Optimal bankroll fraction calculated via Eighth-Kelly sizing."
            )

            # Detail Dropdown: Film Analysis & Player Props
            with st.expander("🔍 Tactical Tape Breakdown & Sportsbook Line Comparisons", expanded=is_bet):
                st.markdown(f"**Executive Game Note:** {analysis_data.get('executive_summary', 'Analysis pending.')}")
                
                subtab_tape, subtab_props = st.tabs(["🎥 Film & Scheme Clash", "📊 Player Yardage Benchmarks"])
                
                with subtab_tape:
                    scheme = analysis_data.get('schematic_matchup', {})
                    t_col1, t_col2 = st.columns(2)
                    with t_col1:
                        st.markdown("**🛡️ Away Offense vs. Home Defense**")
                        st.info(scheme.get('away_offense_vs_home_defense', 'Trench and coverage matchup breakdown unavailable.'))
                    with t_col2:
                        st.markdown("**⚔️ Home Offense vs. Away Defense**")
                        st.info(scheme.get('home_offense_vs_away_defense', 'Trench and coverage matchup breakdown unavailable.'))

                with subtab_props:
                    projections = analysis_data.get('player_projections', [])
                    teams = row['matchup'].split('@')
                    away_team_name = teams[0].strip()
                    home_team_name = teams[1].strip()

                    def display_team_props(team_abbr, container_col):
                        with container_col:
                            st.markdown(f"##### {team_abbr} Projected Yardage vs. Market")
                            team_props = [p for p in projections if p.get("team", "").strip().upper() == team_abbr.upper()]
                            
                            if team_props:
                                rows = []
                                for p in team_props:
                                    m_line = float(p.get('market_line', 0.0))
                                    p_val = float(p.get('projected_value', 0.0))
                                    delta = p_val - m_line
                                    pick = p.get('edge', 'PASS').upper()
                                    
                                    rows.append({
                                        "Player": f"{p.get('player', 'Unknown')} ({p.get('role', 'SKILL')})",
                                        "Stat": p.get('prop_category', 'Yards'),
                                        "Vegas Line": f"{m_line:.1f}",
                                        "AI Median": f"{p_val:.1f}",
                                        "Edge": f"{delta:+.1f}",
                                        "Action": pick,
                                        "Film Reason": p.get('tactical_rationale', '-')
                                    })
                                
                                prop_df = pd.DataFrame(rows)
                                st.dataframe(
                                    prop_df,
                                    column_config={
                                        "Player": st.column_config.TextColumn("Player"),
                                        "Stat": st.column_config.TextColumn("Category"),
                                        "Vegas Line": st.column_config.TextColumn("Book Line"),
                                        "AI Median": st.column_config.TextColumn("AI Projection (50th%)"),
                                        "Edge": st.column_config.TextColumn("Discrepancy"),
                                        "Action": st.column_config.TextColumn("Execution"),
                                        "Film Reason": st.column_config.TextColumn("Coaching Tape Rationale", width="large")
                                    },
                                    hide_index=True,
                                    use_container_width=True
                                )
                            else:
                                st.caption(f"No individual props modeled for {team_abbr}.")

                    p_col1, p_col2 = st.columns(2)
                    display_team_props(away_team_name, p_col1)
                    display_team_props(home_team_name, p_col2)

            st.markdown("<hr style='border: 1px solid #1a202c; margin: 24px 0;'>", unsafe_allow_html=True)

    if displayed_count == 0:
        st.info("No games match your current filter thresholds. Try decreasing the minimum edge slider.")

# =========================================================
# TAB 2: MARKET STEAM & DISCREPANCIES
# =========================================================
with tab_steam:
    st.subheader("⚡ Line Movement & Market Pricing Inefficiencies")
    st.markdown("Comparing pure quantitative models against sportsbooks reveals where public sentiment has created betting value.")

    steam_data = []
    for _, r in df.iterrows():
        p_cal = float(r.get('home_win_prob') or 0.5)
        p_mkt = float(r.get('market_prob') or 0.5)
        discrepancy = (p_cal - p_mkt) * 100
        edge = float(r.get('spread_edge') or 0.0) * 100
        stake = float(r.get('kelly_units') or 0.0)

        if discrepancy >= 4.0:
            signal = "🟢 Under-Priced (Home Value)"
        elif discrepancy <= -4.0:
            signal = "🔴 Over-Priced (Away Value)"
        else:
            signal = "⚪ Fairly Priced"

        steam_data.append({
            "Matchup": r["matchup"],
            "Model Win%": f"{p_cal*100:.1f}%",
            "Vegas Win%": f"{p_mkt*100:.1f}%",
            "Divergence": f"{discrepancy:+.1f}%",
            "Spread Edge": f"{edge:+.1f}%",
            "Rec. Stake": f"{min(stake, max_stake_cap):.2f}u",
            "Market Opportunity": signal
        })

    steam_df = pd.DataFrame(steam_data)
    st.dataframe(steam_df, hide_index=True, use_container_width=True)

# =========================================================
# TAB 3: STRATEGIC GURU WORKBENCH
# =========================================================
with tab_workbench:
    st.subheader(f"🧠 Interactive AI Evaluator ({guru_mode})")
    st.caption("Ask questions about any matchup, evaluate coaching schemes, or audit statistical betting prompts.")

    target_headline = st.text_input(
        "Topic or Matchup Headline:",
        placeholder="e.g., How does Detroit's interior offensive line handle New Orleans' 4-3 front?"
    )
    user_payload = st.text_area(
        "Details, Tape Notes, or Prompt Code to Audit:",
        height=200,
        placeholder="Enter your query, scheme notes, or code here..."
    )

    if st.button("Run Guru Evaluation", type="primary", use_container_width=True):
        if not target_headline or not user_payload:
            st.warning("Please provide both a topic headline and details.")
        else:
            with st.spinner("Analyzing scheme tape and quantitative baselines..."):
                prompt = f"[{guru_mode.upper()}]\nSUBJECT: {target_headline}\n\nDETAILS:\n{user_payload}"
                try:
                    res = ai_client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=GURU_SYSTEM_INSTRUCTION,
                            temperature=0.15
                        )
                    )
                    st.markdown("### 📋 Evaluation Verdict")
                    st.markdown(res.text)
                except Exception as e:
                    st.error(f"Analysis Generation Error: {e}")
