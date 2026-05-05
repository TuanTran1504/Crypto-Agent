"""
Backfills BTC and ETH funding rate history from Binance futures public API.
Runs locally (not Modal) — your machine in Vietnam can access Binance futures,
Modal's US servers cannot (HTTP 451 geo-restriction).

Fetches ~333 days of history (1000 readings / 3 per day).
Aggregates to daily averages and saves to funding_rates table.

Run once:
    python seed_funding_rates.py
"""
import httpx
import psycopg2
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")
DATABASE_URL = os.getenv("DATABASE_URL")

SYMBOLS = ["BTC", "ETH"]


def fetch_funding_rates(symbol: str) -> list:
    """
    Fetch full funding rate history from Binance futures public API.
    Paginates backwards in time — Binance returns max 200 per request
    for this endpoint regardless of the limit param.
    """
    binance_sym = f"{symbol}USDT"
    all_entries = []
    end_time    = None

    for page in range(200):  # up to 200 pages = ~13,000 days, well beyond history
        params = {"symbol": binance_sym, "limit": 1000}
        if end_time:
            params["endTime"] = end_time

        try:
            resp = httpx.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            entries = resp.json()
        except Exception as e:
            print(f"  {symbol} page {page + 1} failed: {e}")
            break

        if not entries:
            break  # no more history

        all_entries.extend(entries)
        print(f"  {symbol}: page {page + 1} — {len(entries)} entries (total {len(all_entries)})")

        # Paginate backwards — set endTime to just before the oldest entry
        end_time = int(entries[0]["fundingTime"]) - 1
        time.sleep(0.2)

    return all_entries


def aggregate_to_daily(entries: list) -> dict:
    """Group 8h funding rate readings into daily averages."""
    by_date = defaultdict(list)
    for entry in entries:
        ts   = int(entry["fundingTime"])
        rate = float(entry["fundingRate"])
        date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date()
        by_date[date].append(rate)
    return {date: sum(rates) / len(rates) for date, rates in by_date.items()}


def save_to_db(symbol: str, daily_rates: dict) -> int:
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    saved = 0

    for date, avg_rate in daily_rates.items():
        cur.execute("""
            INSERT INTO funding_rates (symbol, date, rate_avg, fetched_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (symbol, date) DO UPDATE
                SET rate_avg   = EXCLUDED.rate_avg,
                    fetched_at = NOW()
        """, (symbol, date, avg_rate))
        saved += 1

    conn.commit()
    cur.close()
    conn.close()
    return saved


def main():
    for symbol in SYMBOLS:
        print(f"\nFetching {symbol} funding rate history from Binance...")
        entries = fetch_funding_rates(symbol)

        if not entries:
            print(f"  No data fetched for {symbol}")
            continue

        daily_rates = aggregate_to_daily(entries)
        dates       = sorted(daily_rates.keys())
        print(f"  Aggregated {len(entries)} readings → {len(daily_rates)} daily averages")
        print(f"  From: {dates[0]}  To: {dates[-1]}")

        saved = save_to_db(symbol, daily_rates)
        print(f"  Saved {saved} rows to funding_rates table")


if __name__ == "__main__":
    main()
