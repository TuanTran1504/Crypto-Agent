"""
Backfills Fear & Greed index history from alternative.me (free, no key needed).
Fetches last 500 days and saves to fear_greed_index table.

Run once (or periodically to refresh):
    python seed_fear_greed.py
"""
import httpx
import psycopg2
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")
DATABASE_URL = os.getenv("DATABASE_URL")


def main():
    print("Fetching Fear & Greed index (500 days) from alternative.me...")

    try:
        resp = httpx.get(
            "https://api.alternative.me/fng/",
            params={"limit": 0, "format": "json"},  # 0 = return all history
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        print(f"Fetched {len(data)} entries")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    saved = 0

    for entry in data:
        ts    = int(entry["timestamp"])
        value = float(entry["value"])
        date  = datetime.fromtimestamp(ts, tz=timezone.utc).date()

        cur.execute("""
            INSERT INTO fear_greed_index (date, value, fetched_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (date) DO UPDATE
                SET value      = EXCLUDED.value,
                    fetched_at = NOW()
        """, (date, value))
        saved += 1

    conn.commit()
    cur.close()
    conn.close()

    dates = sorted(
        datetime.fromtimestamp(int(e["timestamp"]), tz=timezone.utc).date()
        for e in data
    )
    print(f"Saved {saved} entries")
    print(f"From: {dates[-1]}  To: {dates[0]}")


if __name__ == "__main__":
    main()
