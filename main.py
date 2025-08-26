#!/usr/bin/env python3
"""
bx_trender_async_final.py

Mínimos cambios para ejecutar con `await`:
- único loop asyncio (asyncio.run(main_async()))
- inicializa bot con `await bot.get_me()`
- envía mensajes con `await bot.send_message(...)`
- tareas bloqueantes (yfinance, sqlite, pandas) se ejecutan con asyncio.to_thread
"""

import os
import time
import sqlite3
import json
import datetime as dt
from pathlib import Path
from typing import List, Optional

import asyncio
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import yfinance as yf
import pandas_ta as ta
from telegram import Bot

# ---------------- CONFIG ----------------
DB_FILE = Path("prices.db")
TICKERS: List[str] = ["AAPL", "MSFT", "NVDA"]
BACKFILL_YEARS = 3
UPDATE_PERIOD = "60d"
REQUEST_DELAY = 1.0

SHORT_L1 = 5
SHORT_L2 = 20
SHORT_L3 = 15
T3_LENGTH = 5
T3_V = 0.7

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

print(f"TELEGRAM_TOKEN: {TELEGRAM_TOKEN}")
print(f"TELEGRAM_CHAT_ID: {TELEGRAM_CHAT_ID}")

# global bot (async)
bot: Optional[Bot] = None

# ---------------- sync helpers (used via to_thread) ----------------
def get_conn():
    return sqlite3.connect(DB_FILE)

def init_db_sync():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                color TEXT NOT NULL,
                bx_value REAL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS state (
                ticker TEXT PRIMARY KEY,
                last_color TEXT,
                updated_at TEXT
            )
            """
        )
        conn.commit()

def df_from_sql_sync(ticker: str) -> Optional[pd.DataFrame]:
    with get_conn() as conn:
        try:
            df = pd.read_sql(f"SELECT * FROM prices_{ticker}", conn, index_col="date", parse_dates=["date"])
            df.index.name = "date"
            return df.sort_index()
        except Exception:
            return None

def save_df_to_sql_sync(df: pd.DataFrame, ticker: str):
    if df is None or df.empty:
        return
    df_to_save = df.copy()
    df_to_save.index = pd.to_datetime(df_to_save.index)
    with get_conn() as conn:
        df_to_save.to_sql(f"prices_{ticker}", conn, if_exists="replace", index_label="date")

def backfill_ticker_sync(ticker: str, years: int = BACKFILL_YEARS):
    period = f"{years}y"
    print(f"[BACKFILL] {ticker} period={period}")
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, multi_level_index=False, progress=False)
        if df is None or df.empty:
            print(f"[WARN] Backfill: no data for {ticker}")
            return
        df.index = pd.to_datetime(df.index)
        save_df_to_sql_sync(df, ticker)
        print(f"[BACKFILL] {ticker} saved rows={len(df)}")
    except Exception as e:
        print(f"[ERROR] backfill {ticker}: {e}")

def incremental_update_ticker_sync(ticker: str, period: str = UPDATE_PERIOD):
    print(f"[UPDATE] {ticker} period={period}")
    try:
        existing = df_from_sql_sync(ticker)
        df_new = yf.download(ticker, period=period, interval="1d", auto_adjust=False, multi_level_index=False, progress=False)
        if df_new is None or df_new.empty:
            print(f"[WARN] No incremental data for {ticker}")
            return
        df_new.index = pd.to_datetime(df_new.index)
        if existing is None or existing.empty:
            merged = df_new
        else:
            merged = pd.concat([existing, df_new[~df_new.index.isin(existing.index)]])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        save_df_to_sql_sync(merged, ticker)
        print(f"[UPDATE] {ticker} rows={len(merged)} last={merged.index.max().date()}")
    except Exception as e:
        print(f"[ERROR] incremental_update {ticker}: {e}")

# ---------------- indicator fallbacks (sync) ----------------
def _ema(series: pd.Series, length: int) -> pd.Series:
    try:
        out = ta.ema(series, length=length)
    except Exception:
        out = None
    if out is None or (isinstance(out, pd.Series) and out.isna().all()):
        out = series.ewm(span=length, adjust=False).mean()
    return out.reindex(series.index)

def _rsi(series: pd.Series, length: int) -> pd.Series:
    try:
        out = ta.rsi(series, length=length)
    except Exception:
        out = None
    if out is None or (isinstance(out, pd.Series) and out.isna().all()):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.ewm(alpha=1/length, adjust=False).mean()
        ma_down = down.ewm(alpha=1/length, adjust=False).mean()
        rs = ma_up / ma_down
        out = 100 - (100 / (1 + rs))
    return out.reindex(series.index)

def _t3(series: pd.Series, length: int, v: float) -> pd.Series:
    try:
        out = ta.t3(series, length=length, v=v)
    except Exception:
        out = None
    if out is None or (isinstance(out, pd.Series) and out.isna().all()):
        xe1 = series.ewm(span=length, adjust=False).mean()
        xe2 = xe1.ewm(span=length, adjust=False).mean()
        xe3 = xe2.ewm(span=length, adjust=False).mean()
        xe4 = xe3.ewm(span=length, adjust=False).mean()
        xe5 = xe4.ewm(span=length, adjust=False).mean()
        xe6 = xe5.ewm(span=length, adjust=False).mean()
        b = v
        c1 = -b * b * b
        c2 = 3 * b * b + 3 * b * b * b
        c3 = -6 * b * b - 3 * b - 3 * b * b * b
        c4 = 1 + 3 * b + b * b * b + 3 * b * b
        out = c1 * xe6 + c2 * xe5 + c3 * xe4 + c4 * xe3
        out = out.reindex(series.index)
    return out

# ---------------- BX Trender calc (sync) ----------------
def compute_bx_trender_sync(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    if "Close" not in dfc.columns:
        raise RuntimeError("DataFrame sin columna Close")

    close = pd.to_numeric(dfc["Close"], errors="coerce").ffill()
    if close.dropna().shape[0] < max(SHORT_L2, SHORT_L3, T3_LENGTH) + 10:
        raise RuntimeError("Datos insuficientes para calcular indicadores (pide más historial)")

    ema1 = _ema(close, SHORT_L1)
    ema2 = _ema(close, SHORT_L2)
    diff = ema1 - ema2

    rsi_diff = _rsi(diff, SHORT_L3)
    short_term = rsi_diff - 50

    ma_short = _t3(short_term.ffill(), T3_LENGTH, T3_V)

    dfc["short_xtrender"] = short_term
    dfc["ma_short_xtrender"] = ma_short
    dfc["bx_value"] = dfc["ma_short_xtrender"]

    def color_from_ma_diff(x):
        if pd.isna(x):
            return None
        return "green" if x > 0 else ("red" if x < 0 else None)

    dfc["bx_color"] = dfc["ma_short_xtrender"].diff().apply(color_from_ma_diff)

    prev_short = dfc["short_xtrender"].shift(1)

    def state_from_short(curr, prev_val):
        if pd.isna(curr) or pd.isna(prev_val):
            return None
        if curr > 0:
            return "green_hh" if curr > prev_val else "green_lh"
        else:
            return "red_hl" if curr > prev_val else "red_ll"

    dfc["bx_state"] = [
        state_from_short(c, p) for c, p in zip(dfc["short_xtrender"].tolist(), prev_short.tolist())
    ]

    return dfc

# ---------------- state & alerts (sync DB) ----------------
def get_last_color_db_sync(ticker: str) -> Optional[str]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT last_color FROM state WHERE ticker = ?", (ticker,))
        r = cur.fetchone()
        return r[0] if r else None

def set_last_color_db_sync(ticker: str, color: str):
    now = dt.datetime.utcnow().isoformat()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO state (ticker, last_color, updated_at) VALUES (?, ?, ?)", (ticker, color, now))
        conn.commit()

def save_alert_db_sync(ticker: str, date: str, color: str, bx_value: Optional[float]):
    now = dt.datetime.utcnow().isoformat()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO alerts (ticker, date, color, bx_value, created_at) VALUES (?, ?, ?, ?, ?)", (ticker, date, color, bx_value, now))
        conn.commit()

# ---------------- notifier (async using await) ----------------
async def send_telegram_async(msg: str, max_retries: int = 2) -> bool:
    global bot
    if not bot:
        print("[WARN] Bot no inicializado. Mensaje no enviado:", msg)
        return False
    if not TELEGRAM_CHAT_ID:
        print("[WARN] TELEGRAM_CHAT_ID no configurado. Mensaje no enviado:", msg)
        return False
    try:
        chat_id = int(TELEGRAM_CHAT_ID)
    except Exception as e:
        print("[ERROR] TELEGRAM_CHAT_ID inválido:", TELEGRAM_CHAT_ID, e)
        return False

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            res = await bot.send_message(chat_id=chat_id, text=msg)
            print(f"✅ Mensaje enviado (intent {attempt}). message_id={getattr(res, 'message_id', None)}")
            return True
        except Exception as e:
            last_exc = e
            print(f"[WARN] Intento {attempt} falló al enviar Telegram: {e}")
            await asyncio.sleep(1)
    print("[ERROR] No se pudo enviar mensaje por telegram tras", max_retries, "intentos. Excepción:", last_exc)
    return False

# ---------------- main async flow ----------------
async def run_backfill_and_updates_async(tickers: List[str]):
    needs_backfill = False
    if not DB_FILE.exists():
        needs_backfill = True
    else:
        tbl = await asyncio.to_thread(df_from_sql_sync, tickers[0])
        if tbl is None or tbl.empty:
            needs_backfill = True

    if needs_backfill:
        print("[INIT] Backfill inicial (esto puede tardar unos segundos por ticker)...")
        for t in tickers:
            await asyncio.to_thread(backfill_ticker_sync, t, BACKFILL_YEARS)
            await asyncio.sleep(REQUEST_DELAY)
    else:
        print("[INIT] DB existente. Saltando backfill.")

    print("[INIT] Actualizando incrementalmente...")
    for t in tickers:
        await asyncio.to_thread(incremental_update_ticker_sync, t, UPDATE_PERIOD)
        await asyncio.sleep(REQUEST_DELAY)

async def process_and_notify_async(tickers: List[str]):
    human_map = {
        "green_hh": "GREEN (Higher High)",
        "green_lh": "GREEN (Lower High)",
        "red_hl": "RED (Higher Low)",
        "red_ll": "RED (Lower Low)"
    }

    for t in tickers:
        try:
            df = await asyncio.to_thread(df_from_sql_sync, t)
            if df is None or df.empty:
                print(f"[WARN] Sin datos en DB para {t}")
                continue
            df_calc = await asyncio.to_thread(compute_bx_trender_sync, df)
        except Exception as e:
            print(f"[ERROR] cálculo para {t}: {e}")
            dbg_name = f"debug_{t}.csv"
            try:
                await asyncio.to_thread(df.to_csv, dbg_name)
                print(f"  (datos guardados en {dbg_name})")
            except Exception:
                pass
            continue

        last_idx = df_calc.index.max()
        last_date = last_idx.date().isoformat()
        bx_state = df_calc.loc[last_idx, "bx_state"]
        bx_val = df_calc.loc[last_idx, "bx_value"] if not pd.isna(df_calc.loc[last_idx, "bx_value"]) else None

        if bx_state is None:
            print(f"{t} {last_date} -> estado indeterminado")
            continue

        prev = await asyncio.to_thread(get_last_color_db_sync, t)
        if prev != bx_state:
            human = human_map.get(bx_state, bx_state)
            msg = f"{t} — BX Trender daily cambió a {human} el {last_date}\nBX_value={bx_val:.6f}"
            sent = await send_telegram_async(msg)
            if sent:
                await asyncio.to_thread(save_alert_db_sync, t, last_date, bx_state, bx_val)
                await asyncio.to_thread(set_last_color_db_sync, t, bx_state)
                print(f"[ALERT] {t} {last_date} -> {bx_state}")
            else:
                print(f"[WARN] No se guardó alerta para {t} porque no se pudo enviar mensaje.")
        else:
            print(f"{t} {last_date} -> sin cambio ({bx_state})")

async def main_async():
    global bot
    # init DB (sync via thread)
    await asyncio.to_thread(init_db_sync)

    # init bot async
    if TELEGRAM_TOKEN:
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
            await bot.get_me()
            print("[OK] Telegram Bot inicializado correctamente.")
        except Exception as e:
            bot = None
            print("[ERROR] No se pudo inicializar Telegram Bot:", e)
    else:
        print("[WARN] TELEGRAM_TOKEN no configurado.")

    # backfill + updates
    await run_backfill_and_updates_async(TICKERS)

    # process and notify
    await process_and_notify_async(TICKERS)

    print("Done.")

if __name__ == "__main__":
    asyncio.run(main_async())
