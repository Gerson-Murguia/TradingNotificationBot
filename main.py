"""
bx_trender_alert.py

Script mínimo para: descargar cierres diarios (yfinance), calcular una aproximación
al BX Trender usando T3 smoothing (pandas_ta) y enviar notificaciones por Telegram
cuando el color (verde/rojo) cambie al cierre diario.

Notas:
- Esta es una IMPLEMENTACIÓN PRÁCTICA y aproximada del BX Trender. Si quieres
  que sea exactamente igual a tu versión de TradingView, pégame el Pine Script
  y lo porto exactamente.
- Guarda el estado en `state.json` para no repetir notificaciones.

Requisitos:
pip install yfinance pandas pandas_ta python-telegram-bot

Uso:
- Exporta variables de entorno TELEGRAM_TOKEN y TELEGRAM_CHAT_ID, o ponlas en el
  archivo `config` dentro del script (no recomendado para producción).
- Edita la lista TICKERS.
- Ejecuta justo DESPUÉS del cierre del mercado (p.ej. 15:10 hora Lima / 16:10 ET
  durante sesión normal).

"""

import os
import json
import datetime as dt
from pathlib import Path
from typing import List, Dict, Optional

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from telegram import Bot

# ---------------- CONFIG ----------------
TICKERS: List[str] = [
    "AAPL", "MSFT", "NVDA"
]
LOOKBACK_DAYS = 120  # historial para cálculo
STATE_FILE = Path("state.json")

# Parámetros "propios" que mencionaste en tu contexto
# (esto es una aproximación — podemos adaptarlo si tienes Pine exacto)
SHORT_L1 = 5
SHORT_L2 = 20
SHORT_L3 = 5
LONG_L1 = 20
LONG_L2 = 5

# Telegram (usa variables de entorno si puedes)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    # No frills: permito ponerlas aquí si estás testeando — pero mejor usa envvars
    TELEGRAM_TOKEN = TELEGRAM_TOKEN or ""
    TELEGRAM_CHAT_ID = TELEGRAM_CHAT_ID or ""

bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

# ---------------------------------------


def fetch_daily(ticker: str, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Descarga OHLC diario con yfinance y retorna DataFrame con índice fecha."""
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    df = yf.download(ticker, start=start.isoformat(), end=(end + dt.timedelta(days=1)).isoformat(),
                     interval="1d", progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    df = df.dropna()
    return df


def compute_bx_trender(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula una versión práctica del "BX Trender".

    En esta aproximación se usan varios T3 (suavizados) y se promedia para obtener
    un valor de referencia. El "color" lo decidimos comparando el valor actual
    con el anterior.
    """
    close = df["Close"].copy()

    # Evitar warnings si pandas_ta devuelve NaNs al inicio
    df = df.copy()

    # Calculamos tres T3 con longitudes distintas (ajustables)
    df[f"t3_s1"] = ta.t3(close, length=SHORT_L1, v=0.7)
    df[f"t3_s2"] = ta.t3(close, length=SHORT_L2, v=0.7)
    df[f"t3_s3"] = ta.t3(close, length=SHORT_L3, v=0.7)
    df[f"t3_l1"] = ta.t3(close, length=LONG_L1, v=0.7)
    df[f"t3_l2"] = ta.t3(close, length=LONG_L2, v=0.7)

    # BX value: promedio simple de los T3 seleccionados (esto es una elección práctica)
    df["bx_value"] = df[["t3_s1", "t3_s2", "t3_s3", "t3_l1", "t3_l2"]].mean(axis=1)

    # Color: si bx_value > bx_value.shift(1) -> verde, si < -> rojo, si NaN -> None
    df["bx_color"] = df["bx_value"].diff().apply(lambda x: "green" if x > 0 else ("red" if x < 0 else None))

    return df


def load_state() -> Dict[str, str]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_state(state: Dict[str, str]):
    STATE_FILE.write_text(json.dumps(state))


def send_telegram(msg: str):
    if not bot:
        print("[WARN] TELEGRAM TOKEN no configurado; no se envía mensaje. Mensaje: ", msg)
        return
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)


def check_and_notify(ticker: str, df: pd.DataFrame, state: Dict[str, str]):
    df = compute_bx_trender(df)
    last_date = df.index[-1].date()
    bx_color = df["bx_color"].iloc[-1]

    key = ticker
    prev_color = state.get(key)

    # Si no hay color (NaN por corto historial), no hacemos nada
    if bx_color is None or pd.isna(bx_color):
        print(f"{ticker} {last_date} -> color indeterminado (datos insuficientes)")
        return

    # Si cambió respecto al estado guardado, notificamos
    if prev_color != bx_color:
        msg = (f"{ticker} — BX Trender daily cambió a {bx_color.upper()} el {last_date}\n"
               f"BX_value={df['bx_value'].iloc[-1]:.6f} (prev={df['bx_value'].iloc[-2]:.6f})")
        send_telegram(msg)
        print("Enviado:", msg)
        state[key] = bx_color
    else:
        print(f"{ticker} {last_date} -> sin cambio ({bx_color})")


def main(tickers: List[str]):
    state = load_state()
    for t in tickers:
        try:
            df = fetch_daily(t)
            check_and_notify(t, df, state)
        except Exception as e:
            print(f"Error con {t}: {e}")
    save_state(state)


if __name__ == "__main__":
    main(TICKERS)
