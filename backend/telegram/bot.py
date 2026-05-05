"""
Telegram bot for crypto-agent trading notifications and queries.

Commands:
  /start   - welcome message
  /status  - live account summary
  /live    - live open positions with unrealized P&L
  /trades  - last 10 closed live trades
  /pnl     - live P&L summary
  /review  - manually trigger a forced policy review
"""

import asyncio
import importlib
import logging
import os
import sys
from datetime import timezone
from pathlib import Path

import psycopg2
from binance.um_futures import UMFutures
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

ROOT = Path(__file__).resolve().parent.parent.parent
SCHEDULE_DIR = ROOT / "backend" / "schedule"
load_dotenv(ROOT / ".env")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
DATABASE_URL = os.getenv("DATABASE_URL", "")
LIVE_API_KEY = os.getenv("BINANCE_LIVE_API_KEY", "")
LIVE_API_SECRET = os.getenv("BINANCE_LIVE_SECRET_KEY", "")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

UTC = timezone.utc
REVIEW_LOCK = asyncio.Lock()


def get_db():
    return psycopg2.connect(DATABASE_URL)


def get_live_client():
    if not LIVE_API_KEY or not LIVE_API_SECRET:
        return None
    return UMFutures(key=LIVE_API_KEY, secret=LIVE_API_SECRET)


def trades_table_for_account(account_type=None):
    if account_type == "live":
        return "trades_live"
    return "trades"


def fmt_pnl(val):
    if val is None:
        return "-"
    return f"+{val:.2f}" if val >= 0 else f"{val:.2f}"


def fmt_pct(val):
    if val is None:
        return "-"
    return f"+{val:.2f}%" if val >= 0 else f"{val:.2f}%"


def fmt_price(val):
    if val is None:
        return "-"
    return f"{val:.2f}"


def review_chat_is_authorized(update: Update) -> bool:
    if CHAT_ID <= 0:
        return False
    chat = update.effective_chat
    return bool(chat and chat.id == CHAT_ID)


def build_review_override_reason(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    chat = update.effective_chat
    user = update.effective_user
    note = " ".join(getattr(context, "args", []) or []).strip()
    user_text = "unknown"
    if user:
        username = f":{user.username}" if user.username else ""
        user_text = f"{user.id}{username}"
    chat_id = chat.id if chat else 0
    reason = f"telegram:/review chat={chat_id} user={user_text}"
    if note:
        reason = f"{reason} note={note}"
    return reason


def format_review_result(result: dict) -> str:
    lines = [f"Policy review status: {result.get('status', 'unknown')}"]
    if result.get("run_id") is not None:
        lines.append(f"run_id: {result['run_id']}")
    if result.get("policy_id") is not None:
        lines.append(
            "policy: "
            f"id={result['policy_id']} "
            f"version={result.get('policy_version', '-')} "
            f"status={result.get('policy_status', '-')}"
        )
    reason = str(result.get("reason") or "").strip()
    if reason:
        lines.append(f"reason: {reason}")
    manual_override = dict(result.get("manual_override") or {})
    if manual_override.get("force_review"):
        lines.append("manual_override: force_review=true")
    validation = dict(result.get("validation") or {})
    proposal_score = dict(validation.get("proposal_score") or {})
    if proposal_score:
        lines.append(
            f"proposal_score: {proposal_score.get('score', '-')} "
            f"({proposal_score.get('verdict', '-')})"
        )
    return "\n".join(lines)


def get_policy_review_runner():
    if str(SCHEDULE_DIR) not in sys.path:
        sys.path.insert(0, str(SCHEDULE_DIR))
    return importlib.import_module("run_policy_review")


def fetch_open_trades(account_type=None):
    conn = get_db()
    try:
        cur = conn.cursor()
        table_name = trades_table_for_account(account_type)
        if account_type:
            cur.execute(
                f"""
                SELECT symbol, side, entry_price, stop_loss, take_profit,
                       quantity, leverage, opened_at,
                       COALESCE(account_type, 'testnet') as account_type
                FROM {table_name}
                WHERE status = 'OPEN'
                  AND COALESCE(account_type, 'testnet') = %s
                ORDER BY opened_at DESC
                """,
                (account_type,),
            )
        else:
            cur.execute(
                f"""
                SELECT symbol, side, entry_price, stop_loss, take_profit,
                       quantity, leverage, opened_at,
                       COALESCE(account_type, 'testnet') as account_type
                FROM {table_name}
                WHERE status = 'OPEN'
                ORDER BY opened_at DESC
                """
            )
        return cur.fetchall()
    finally:
        conn.close()


def fetch_closed_trades(limit=10, account_type=None):
    conn = get_db()
    try:
        cur = conn.cursor()
        table_name = trades_table_for_account(account_type)
        if account_type:
            cur.execute(
                f"""
                SELECT symbol, side, entry_price, exit_price,
                       pnl_usdt, pnl_pct, close_reason, opened_at, closed_at,
                       COALESCE(account_type, 'testnet') as account_type
                FROM {table_name}
                WHERE status = 'CLOSED'
                  AND COALESCE(account_type, 'testnet') = %s
                ORDER BY closed_at DESC
                LIMIT %s
                """,
                (account_type, limit),
            )
        else:
            cur.execute(
                f"""
                SELECT symbol, side, entry_price, exit_price,
                       pnl_usdt, pnl_pct, close_reason, opened_at, closed_at,
                       COALESCE(account_type, 'testnet') as account_type
                FROM {table_name}
                WHERE status = 'CLOSED'
                ORDER BY closed_at DESC
                LIMIT %s
                """,
                (limit,),
            )
        return cur.fetchall()
    finally:
        conn.close()


def fetch_pnl_summary(account_type=None):
    conn = get_db()
    try:
        cur = conn.cursor()
        table_name = trades_table_for_account(account_type)
        if account_type:
            cur.execute(
                f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl_usdt <= 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl_usdt) as total_pnl,
                    AVG(pnl_usdt) as avg_pnl
                FROM {table_name}
                WHERE status = 'CLOSED'
                  AND COALESCE(account_type, 'testnet') = %s
                """,
                (account_type,),
            )
        else:
            cur.execute(
                f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl_usdt <= 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl_usdt) as total_pnl,
                    AVG(pnl_usdt) as avg_pnl
                FROM {table_name}
                WHERE status = 'CLOSED'
                """
            )
        return cur.fetchone()
    finally:
        conn.close()


def fetch_live_account_balance():
    client = get_live_client()
    if not client:
        return None
    try:
        account = client.account()
        for asset in account.get("assets", []):
            if asset.get("asset") == "USDT":
                wallet = float(asset.get("walletBalance", 0) or 0)
                unrealized = float(asset.get("unrealizedProfit", 0) or 0)
                return round(wallet + unrealized, 2)
    except Exception as exc:
        log.warning(f"Live balance fetch failed: {exc}")
    return None


def fetch_live_positions():
    client = get_live_client()
    if not client:
        return {}
    positions = {}
    try:
        risk = client.get_position_risk()
        rows = risk if isinstance(risk, list) else [risk]
        for pos in rows:
            amt = float(pos.get("positionAmt", 0) or 0)
            if abs(amt) <= 0:
                continue
            symbol = (pos.get("symbol") or "").replace("USDT", "")
            if not symbol:
                continue
            positions[symbol] = {
                "amount": abs(amt),
                "entry_price": float(pos.get("entryPrice", 0) or 0),
                "mark_price": float(pos.get("markPrice", 0) or 0),
                "upnl": float(pos.get("unRealizedProfit", 0) or 0),
            }
    except Exception as exc:
        log.warning(f"Live positions fetch failed: {exc}")
    return positions


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Crypto Agent Bot\n\n"
        "Commands:\n"
        "/status - live balance, win rate, total P&L\n"
        "/live - live open trades with unrealized P&L\n"
        "/trades - last 10 live closed trades\n"
        "/pnl - live account P&L summary\n"
        "/livepnl - same as /pnl\n"
        "/review [optional note] - force a manual policy review"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    row = fetch_pnl_summary(account_type="live")
    balance = fetch_live_account_balance()
    open_rows = fetch_open_trades(account_type="live")

    total, wins, losses, total_pnl, avg_pnl = row if row else (0, 0, 0, 0, 0)
    wins = wins or 0
    losses = losses or 0
    total = total or 0
    total_pnl = total_pnl or 0
    win_rate = (wins / total * 100) if total else 0

    await update.message.reply_text(
        "*Live Account Status*\n\n"
        f"Balance: `{fmt_price(balance)} USDT`\n"
        f"Open trades: `{len(open_rows)}`\n"
        f"Win rate: `{win_rate:.1f}%`\n"
        f"Total P&L: `{fmt_pnl(total_pnl)} USDT`\n"
        f"Closed trades: `{total}`",
        parse_mode="Markdown",
    )


async def cmd_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = fetch_open_trades(account_type="live")
    live_positions = fetch_live_positions()
    if not rows:
        await update.message.reply_text("No open live positions.")
        return

    lines = ["*Live Account Positions*\n"]
    for r in rows:
        symbol, side, entry, sl, tp, qty, lev, opened_at, acc = r
        live_pos = live_positions.get(symbol, {})
        mark_price = live_pos.get("mark_price")
        upnl = live_pos.get("upnl")
        side_icon = "LONG" if side == "BUY" else "SHORT"
        lines.append(
            f"*{symbol}* {side_icon}\n"
            f"  Entry: `{fmt_price(entry)}` | Qty: `{qty}` | Lev: `{lev}x`\n"
            f"  Mark: `{fmt_price(mark_price)}` | uPnL: `{fmt_pnl(upnl)} USDT`\n"
            f"  SL: `{fmt_price(sl)}` | TP: `{fmt_price(tp)}`\n"
            f"  Opened: {opened_at.strftime('%m/%d %H:%M') if opened_at else '-'}\n"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = fetch_closed_trades(limit=10, account_type="live")
    if not rows:
        await update.message.reply_text("No closed live trades yet.")
        return

    lines = ["*Last 10 Live Trades*\n"]
    for r in rows:
        symbol, side, entry, exit_p, pnl, pnl_pct, reason, opened_at, closed_at, acc = r
        outcome = "WIN" if (pnl or 0) > 0 else "LOSS"
        lines.append(
            f"{outcome} *{symbol}* {side} -> "
            f"`{fmt_pnl(pnl)} USDT` ({fmt_pct(pnl_pct)}) - {reason or '-'}\n"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    row = fetch_pnl_summary(account_type="live")
    if not row or not row[0]:
        await update.message.reply_text("No closed live trades yet.")
        return

    total, wins, losses, total_pnl, avg_pnl = row
    wins = wins or 0
    losses = losses or 0
    win_rate = (wins / total * 100) if total else 0

    await update.message.reply_text(
        "*Live Account P&L*\n\n"
        f"Total trades: `{total}`\n"
        f"Wins: `{wins}` | Losses: `{losses}`\n"
        f"Win rate: `{win_rate:.1f}%`\n"
        f"Total P&L: `{fmt_pnl(total_pnl)} USDT`\n"
        f"Avg per trade: `{fmt_pnl(avg_pnl)} USDT`",
        parse_mode="Markdown",
    )


async def cmd_livepnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_pnl(update, context)


async def cmd_review(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if CHAT_ID <= 0:
        await update.message.reply_text(
            "Manual review is disabled because TELEGRAM_CHAT_ID is not configured."
        )
        return

    if not review_chat_is_authorized(update):
        chat = update.effective_chat
        log.warning("Rejected /review from unauthorized chat_id=%s", getattr(chat, "id", None))
        await update.message.reply_text("Unauthorized chat for manual policy review.")
        return

    if REVIEW_LOCK.locked():
        await update.message.reply_text("A manual policy review is already running. Please wait.")
        return

    review_reason = build_review_override_reason(update, context)
    await update.message.reply_text(
        "Starting manual policy review with gate bypass.\n"
        f"reason: {review_reason}"
    )

    try:
        async with REVIEW_LOCK:
            runner = get_policy_review_runner()
            result = await asyncio.to_thread(
                runner.run_policy_review_once,
                force_review=True,
                force_review_reason=review_reason,
            )
    except Exception as exc:
        log.exception("Manual /review failed")
        await update.message.reply_text(f"Manual policy review failed: {exc}")
        return

    await update.message.reply_text(format_review_result(result))


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("live", cmd_live))
    app.add_handler(CommandHandler("trades", cmd_trades))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("livepnl", cmd_livepnl))
    app.add_handler(CommandHandler("review", cmd_review))

    log.info("Telegram bot started - polling...")
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
