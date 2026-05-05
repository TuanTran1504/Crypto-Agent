"""
Backfills BTC and ETH 30-minute candle data from Binance public API.
Fetches 5 years of history with pagination (48 candles/day).
No API key needed — Binance public endpoint.

Run once:
    python seed_crypto.py
"""
import httpx
import psycopg2
from psycopg2.extras import execute_batch
import os
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent.parent / ".env")
DATABASE_URL = os.getenv("DATABASE_URL")

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOLS = [
    ("BTCUSDT", "BTC"),
    ("ETHUSDT", "ETH"),
    ("SOLUSDT", "SOL"),
    ("XRPUSDT", "XRP"),
]


def fetch_klines(symbol: str, years: int = 5) -> list:
    """
    Fetches 30-minute OHLCV candles from Binance with pagination.
    Returns 5 years of history (~87,600 candles for 30m intervals).
    Binance API limit: 1000 candles per request, so we paginate.
    """
    all_candles = []
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=365 * years)
    start_ms = int(start.timestamp() * 1000)
    
    request_count = 0
    while True:
        request_count += 1
        if request_count > 100:
            print(f"    Warning: {symbol} hit 100 requests limit, stopping")
            break
        
        params = {
            "symbol": symbol,
            "interval": "30m",
            "startTime": start_ms,
            "limit": 1000,
        }
        
        resp = httpx.get(BINANCE_URL, params=params, timeout=30)
        resp.raise_for_status()
        candles = resp.json()
        
        if not candles:
            break  # No more data
        
        all_candles.extend(candles)
        
        # Update start time for next batch (last candle's time + 1)
        last_time = int(candles[-1][0])
        start_ms = last_time + 1
        
        print(f"    Fetched batch {request_count}: {len(candles)} candles")
        time.sleep(0.1)  # Be nice to the API
        
        if len(candles) < 1000:
            break  # Got all available data
    
    print(f"    Total requests: {request_count}, Total candles: {len(all_candles)}")
    return all_candles


def parse_candles(raw: list, display_symbol: str) -> list:
    """
    Parses Binance kline format into clean records.
    
    Binance kline format:
    [open_time, open, high, low, close, volume, close_time, ...]
    """
    records = []
    for c in raw:
        open_time = datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc)
        open_p  = float(c[1])
        high_p  = float(c[2])
        low_p   = float(c[3])
        close_p = float(c[4])
        volume  = float(c[5])
        change  = ((close_p - open_p) / open_p) * 100 if open_p > 0 else 0

        records.append({
            "symbol":     display_symbol,
            "fetched_at": open_time,
            "open_usd":   open_p,
            "high_usd":   high_p,
            "low_usd":    low_p,
            "close_usd":  close_p,
            "volume_usd": volume * close_p,  # convert to USD volume
            "change_pct": round(change, 4),
            "interval_minutes": 30,  # 30-minute candles
        })
    return records


def save_to_supabase(records: list) -> tuple[int, int]:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Prepare batch data
    batch_data = [
        (
            r["symbol"],
            r["fetched_at"],
            r["open_usd"],
            r["high_usd"],
            r["low_usd"],
            r["close_usd"],
            r["volume_usd"],
            r["change_pct"],
            r["interval_minutes"],
        )
        for r in records
    ]

    # Use batch insert for efficiency (thousands of records)
    execute_batch(
        cur,
        """
        INSERT INTO crypto_prices
            (symbol, fetched_at, open_usd, high_usd, low_usd,
             close_usd, volume_usd, change_pct, source, interval_minutes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'binance', %s)
        ON CONFLICT (symbol, fetched_at) DO NOTHING
        """,
        batch_data,
        page_size=1000,
    )

    conn.commit()
    cur.close()
    conn.close()
    return len(records), 0


def main():
    for binance_symbol, display_symbol in SYMBOLS:
        print(f"\nFetching {display_symbol} 30-minute history (5 years)...")
        raw = fetch_klines(binance_symbol, years=5)
        records = parse_candles(raw, display_symbol)

        if not records:
            print(f"  No data fetched for {display_symbol}")
            continue

        print(f"  Parsed {len(records)} 30-minute candles")
        print(f"  From: {records[0]['fetched_at'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  To:   {records[-1]['fetched_at'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        inserted, skipped = save_to_supabase(records)
        print(f"  Saved {inserted} new rows · Skipped {skipped} duplicates")


if __name__ == "__main__":
    main()