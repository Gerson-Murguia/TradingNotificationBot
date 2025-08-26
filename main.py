#!/usr/bin/env python3
"""
bx_trender_full.py

Funcionalidad:
- Backfill inicial (3 años por defecto) y cache en SQLite.
- Actualizaciones incrementales diarias (últimos 60 días).
- Cálculo BX Trender (RSI(EMA(diff)) - 50, luego T3) usando pandas-ta con fallbacks.
- Notificaciones por Telegram cuando el color daily cambia (verde/red).
- Guarda alertas en la tabla `alerts` de la DB y guarda datos debug si hay fallos.

Configura variables de entorno:
  TELEGRAM_TOKEN
  TELEGRAM_CHAT_ID

Uso:
  python bx_trender_full.py
"""

import os
import time
import sqlite3
import json
import datetime as dt
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf
import pandas_ta as ta
from telegram import Bot

# ---------------- CONFIG ----------------
DB_FILE = Path("prices.db")
TICKERS: List[str] = ["AAPL", "MSFT", "NVDA"]   # ajusta a tu portafolio
BACKFILL_YEARS = 3      # backfill inicial (años)
UPDATE_PERIOD = "60d"   # actualizaciones incrementales diarias (seguridad)
REQUEST_DELAY = 1.0     # segundos entre requests a yfinance para ser respetuoso

# BX Trender params (según tu Pine)
SHORT_L1 = 5
SHORT_L2 = 20
SHORT_L3 = 15
T3_LENGTH = 5
T3_V = 0.7

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

# ---------------- DB helpers ----------------
def get_conn():
    return sqlite3.connect(DB_FILE)

def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        # tabla de alerts
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
        # tabla de state (último color conocido)
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

# ---------------- data download / cache ----------------
def df_from_sql(ticker: str) -> Optional[pd.DataFrame]:
    with get_conn() as conn:
        try:
            df = pd.read_sql(f"SELECT * FROM prices_{ticker}", conn, index_col="date", parse_dates=["date"])
            # standardize column names if necessary
            df.index.name = "date"
            return df.sort_index()
        except Exception:
            return None

def save_df_to_sql(df: pd.DataFrame, ticker: str):
    # df must have index datetime and columns: Open, High, Low, Close, Adj Close, Volume (names from yfinance)
    if df is None or df.empty:
        return
    df_to_save = df.copy()
    df_to_save.index = pd.to_datetime(df_to_save.index)
    with get_conn() as conn:
        df_to_save.to_sql(f"prices_{ticker}", conn, if_exists="replace", index_label="date")

def backfill_ticker(ticker: str, years: int = BACKFILL_YEARS):
    period = f"{years}y"
    print(f"[BACKFILL] {ticker} period={period}")
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False,multi_level_index=False,progress=False)
        if df is None or df.empty:
            print(f"[WARN] Backfill: no data for {ticker}")
            return
        # normalize column names (some yfinance returns MultiIndex in batch mode; here single ticker ok)
        df.index = pd.to_datetime(df.index)
        save_df_to_sql(df, ticker)
        print(f"[BACKFILL] {ticker} saved rows={len(df)}")
    except Exception as e:
        print(f"[ERROR] backfill {ticker}: {e}")

def incremental_update_ticker(ticker: str, period: str = UPDATE_PERIOD):
    print(f"[UPDATE] {ticker} period={period}")
    try:
        # existing data
        existing = df_from_sql(ticker)
        last_date = None
        if existing is not None and not existing.empty:
            last_date = existing.index.max().date()
        df_new = yf.download(ticker, period=period, interval="1d", auto_adjust=False,multi_level_index=False, progress=False)
        if df_new is None or df_new.empty:
            print(f"[WARN] No incremental data for {ticker}")
            return
        df_new.index = pd.to_datetime(df_new.index)
        # merge
        if existing is None or existing.empty:
            merged = df_new
        else:
            # append only rows after last_date
            merged = pd.concat([existing, df_new[~df_new.index.isin(existing.index)]])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        save_df_to_sql(merged, ticker)
        print(f"[UPDATE] {ticker} rows={len(merged)} last={merged.index.max().date()}")
    except Exception as e:
        print(f"[ERROR] incremental_update {ticker}: {e}")

# ---------------- indicator fallbacks ----------------
def _ema(series: pd.Series, length: int) -> pd.Series:
    try:
        out = ta.ema(series, length=length)
    except Exception:
        out = None
    if out is None or (isinstance(out, pd.Series) and out.isna().all()):
        out = series.ewm(span=length, adjust=False).mean()
    # ensure same index
    out = out.reindex(series.index)
    return out

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
    out = out.reindex(series.index)
    return out

def _t3(series: pd.Series, length: int, v: float) -> pd.Series:
    try:
        out = ta.t3(series, length=length, v=v)
    except Exception:
        out = None
    if out is None or (isinstance(out, pd.Series) and out.isna().all()):
        # fallback implement T3 via 6 EMA cascade like Pine
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

# ---------------- BX Trender calc ----------------
def compute_bx_trender(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with columns: short_xtrender, ma_short_xtrender, bx_value, bx_color"""
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

    def color_from_diff(x):
        if pd.isna(x):
            return None
        return "green" if x > 0 else ("red" if x < 0 else None)

    dfc["bx_color"] = dfc["ma_short_xtrender"].diff().apply(color_from_diff)
    return dfc

# ---------------- state & alerts ----------------
def get_last_color_db(ticker: str) -> Optional[str]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT last_color FROM state WHERE ticker = ?", (ticker,))
        r = cur.fetchone()
        return r[0] if r else None

def set_last_color_db(ticker: str, color: str):
    now = dt.datetime.utcnow().isoformat()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO state (ticker, last_color, updated_at) VALUES (?, ?, ?)", (ticker, color, now))
        conn.commit()

def save_alert_db(ticker: str, date: str, color: str, bx_value: Optional[float]):
    now = dt.datetime.utcnow().isoformat()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO alerts (ticker, date, color, bx_value, created_at) VALUES (?, ?, ?, ?, ?)", (ticker, date, color, bx_value, now))
        conn.commit()

# ---------------- notifier ----------------
def send_telegram(msg: str):
    if not bot:
        print("[WARN] TELEGRAM no configurado. Mensaje:", msg)
        return
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
    except Exception as e:
        print("[ERROR] Envío Telegram:", e)

# ---------------- main flow ----------------
def run_backfill_and_updates(tickers: List[str]):
    # backfill initial if DB missing or price tables missing
    needs_backfill = False
    if not DB_FILE.exists():
        needs_backfill = True
    else:
        # check first ticker table exists
        tbl = df_from_sql(tickers[0])
        if tbl is None or tbl.empty:
            needs_backfill = True

    if needs_backfill:
        print("[INIT] Backfill inicial (esto puede tardar unos segundos por ticker)...")
        for t in tickers:
            backfill_ticker(t, BACKFILL_YEARS)
            time.sleep(REQUEST_DELAY)
    else:
        print("[INIT] DB existente. Saltando backfill.")

    # siempre hacemos incremental update
    print("[INIT] Actualizando incrementalmente...")
    for t in tickers:
        incremental_update_ticker(t, period=UPDATE_PERIOD)
        time.sleep(REQUEST_DELAY)

def process_and_notify(tickers: List[str]):
    for t in tickers:
        try:
            df = df_from_sql(t)
            if df is None or df.empty:
                print(f"[WARN] Sin datos en DB para {t}")
                continue
            # compute
            df_calc = compute_bx_trender(df)
        except Exception as e:
            print(f"[ERROR] cálculo para {t}: {e}")
            # save debug CSV
            dbg_name = f"debug_{t}.csv"
            try:
                df.to_csv(dbg_name)
                print(f"  (datos guardados en {dbg_name})")
            except Exception:
                pass
            continue

        last_idx = df_calc.index.max()
        last_date = last_idx.date().isoformat()
        bx_color = df_calc.loc[last_idx, "bx_color"]
        bx_val = df_calc.loc[last_idx, "bx_value"] if not pd.isna(df_calc.loc[last_idx, "bx_value"]) else None

        if bx_color is None:
            print(f"{t} {last_date} -> color indeterminado")
            continue

        prev = get_last_color_db(t)
        if prev != bx_color:
            msg = f"{t} — BX Trender daily cambió a {bx_color.upper()} el {last_date}\nBX_value={bx_val:.6f}"
            send_telegram(msg)
            save_alert_db(t, last_date, bx_color, bx_val)
            set_last_color_db(t, bx_color)
            print(f"[ALERT] {t} {last_date} -> {bx_color}")
        else:
            print(f"{t} {last_date} -> sin cambio ({bx_color})")

# ---------------- entrypoint ----------------
def main():
    init_db()
    run_backfill_and_updates(TICKERS)
    process_and_notify(TICKERS)
    print("Done.")

if __name__ == "__main__":
    main()
