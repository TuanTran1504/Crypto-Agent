"""
Backfills BTC and ETH funding rate history from Bybit public API.

Expected output:
  BTC:  ~1800 days of daily averages (3 x 8h readings each)
  ETH:  ~1800 days of daily averages

Run once to backfill; safe to re-run (ON CONFLICT DO UPDATE).

    python seed_funding_rates_bybit.py
"""
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")
DATABASE_URL = os.getenv("DATABASE_URL")

BYBIT_URL = "https://api.bybit.com/v5/market/funding/history"
SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]
REQUEST_DELAY = 0.3  
MAX_PAGES = 500      


def fetch_all_funding_rates(symbol: str) -> list[dict]:
    """
    Fetch complete funding rate history for one symbol from Bybit.

    Uses endTime-based backwards pagination: each request fetches 200 entries
    ending at the timestamp of the oldest entry from the previous page.
    Bybit's cursor field does not expose the full history on the public
    endpoint — endTime pagination works reliably and reaches back to the
    first available reading (~Jan 2020 for BTC/ETH perpetuals).
    """
    bybit_symbol = f"{symbol}USDT"
    all_entries: list[dict] = []
    end_time_ms: int | None = None

    print(f"  Fetching {bybit_symbol} funding history from Bybit...")

    for page in range(1, MAX_PAGES + 1):
        params: dict = {
            "category": "linear",
            "symbol": bybit_symbol,
            "limit": 200,
        }
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        try:
            resp = httpx.get(BYBIT_URL, params=params, timeout=30)
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:
            print(f"    Page {page} failed: {exc}")
            break

        ret_code = body.get("retCode", -1)
        if ret_code != 0:
            print(f"    Bybit error (retCode={ret_code}): {body.get('retMsg')}")
            break

        entries = body.get("result", {}).get("list", [])
        if not entries:
            print(f"    No more entries at page {page}.")
            break

        all_entries.extend(entries)

        # Bybit returns entries newest-first; entries[-1] is the oldest on this page.
        oldest_ts = int(entries[-1]["fundingRateTimestamp"])
        oldest_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc)

        if page % 10 == 0:
            print(f"    Page {page}: {len(all_entries)} entries so far, oldest {oldest_dt.date()}")

        if len(entries) < 200:
            # Partial page means we've reached the very beginning of history.
            print(f"    Reached beginning of history at page {page} ({oldest_dt.date()}).")
            break

        # Set endTime to 1 ms before the oldest entry on this page.
        end_time_ms = oldest_ts - 1
        time.sleep(REQUEST_DELAY)

    return all_entries


def aggregate_to_daily(entries: list[dict]) -> dict:
    """Group 8h funding rate readings into daily averages (3 readings/day)."""
    by_date: dict = defaultdict(list)
    for entry in entries:
        ts = int(entry["fundingRateTimestamp"])
        rate = float(entry["fundingRate"])
        date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date()
        by_date[date].append(rate)
    return {date: sum(rates) / len(rates) for date, rates in by_date.items()}


def save_to_db(symbol: str, daily_rates: dict) -> int:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    saved = 0
    for date, avg_rate in daily_rates.items():
        cur.execute(
            """
            INSERT INTO funding_rates (symbol, date, rate_avg, fetched_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (symbol, date) DO UPDATE
                SET rate_avg   = EXCLUDED.rate_avg,
                    fetched_at = NOW()
            """,
            (symbol, date, avg_rate),
        )
        saved += 1
    conn.commit()
    cur.close()
    conn.close()
    return saved


def main():
    for symbol in SYMBOLS:
        print(f"\n{'─' * 50}")
        print(f"Symbol: {symbol}")
        entries = fetch_all_funding_rates(symbol)

        if not entries:
            print(f"  No entries fetched for {symbol}.")
            continue

        daily_rates = aggregate_to_daily(entries)
        dates = sorted(daily_rates.keys())
        print(f"  Raw entries:    {len(entries)}")
        print(f"  Daily averages: {len(daily_rates)}")
        print(f"  From: {dates[0]}  →  To: {dates[-1]}")

        saved = save_to_db(symbol, daily_rates)
        print(f"  Saved {saved} rows to funding_rates table.")


if __name__ == "__main__":
    main()
