
import os
import json
import nflreadpy as nfl
import pandas as pd
from sqlalchemy import create_engine
from google import genai
from google.genai import types

# 1. Retrieve cloud credentials
db_url = os.environ.get("DATABASE_URL")
gemini_key = os.environ.get("GEMINI_API_KEY")

if not db_url or not gemini_key:
    raise ValueError("Missing DATABASE_URL or GEMINI_API_KEY in environment.")

engine = create_engine(db_url)
client = genai.Client(api_key=gemini_key)

# 2. Ingest schedule data
print("Loading NFL schedule...")
schedules = nfl.load_schedules(seasons=[2026]).to_pandas()

# Filter for upcoming games without recorded results
upcoming = schedules[schedules['result'].isna()].head(3)

# 3. Request analysis from Gemini
system_prompt = (
    "You are an expert NFL quantitative analyst. Break down the matchup "
    "using efficiency metrics (EPA/play, success rate) and projected lines. "
    "Provide a concise summary: 1. Key Mismatch, 2. Game Script, 3. Value Lean."
)

records = []
for _, game in upcoming.iterrows():
    matchup = f"{game['away_team']} @ {game['home_team']}"
    print(f"Analyzing matchup: {matchup}...")

    payload = {
        "game": matchup,
        "week": int(game['week']) if pd.notna(game['week']) else 1,
        "spread_line": float(game['spread_line']) if pd.notna(game['spread_line']) else 0.0,
        "total_line": float(game['total_line']) if pd.notna(game['total_line']) else 0.0
    }

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=json.dumps(payload),
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2
        )
    )

    records.append({
        "game_id": str(game['game_id']),
        "week": int(game['week']) if pd.notna(game['week']) else 1,
        "matchup": matchup,
        "analysis": response.text
    })

# 4. Save to remote Neon database
if records:
    df_results = pd.DataFrame(records)
    df_results.to_sql("nfl_weekly_analysis", engine, if_exists="append", index=False)
    print("Neon database updated successfully.")
