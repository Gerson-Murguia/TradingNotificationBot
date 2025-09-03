#!/usr/bin/env python3
"""
BX Trender Bot - Implementación orientada a objetos
Versión extendida: soporta listas separadas (candidates / portfolio), reglas de confirmación
y detección de entradas/salidas según matriz de decisión. Mantiene batching/telegram.
"""

from calendar import week
import json
import os
import sqlite3
import datetime as dt
import logging
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import asyncio
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from telegram import Bot

# Cargar variables de entorno
load_dotenv()

@dataclass
class BotConfig:
    """Configuración del bot"""

    database_file: str
    # Configuración específica por timeframe
    backfill_years: Dict[str, int]
    update_period: str
    request_delay: float
    #tickers: List[str]
    timeframes: List[str]
    candidates: List[str]
    portfolio: List[str]
    short_l1: int
    short_l2: int
    short_l3: int
    t3_length: int
    t3_v: float
    long_l1: int
    long_l2: int
    telegram_token: str
    telegram_chat_id: str
    telegram_max_retries: int
    telegram_retry_delay_base: int
    # Configuración de batching
    batching_enabled: bool
    max_alerts_per_batch: int
    batch_timeout_seconds: int
    summary_enabled: bool
    summary_time: str
    log_level: str
    log_file_level: str
    log_console_level: str
    log_dir: str
    max_log_files: int
    health_check_interval: int
    metrics_enabled: bool
    alert_on_errors: bool
    # Confirmations + filters
    confirm_daily: int
    confirm_weekly: int
    confirm_monthly: int
    bx_value_min_abs: float
    volume_min_factor: float
    alert_cooldown_days: int
    # Alert types configuration
    alert_types: Dict[str, str]
    re_alert_cooldown_hours: int


class Metrics:
    """Clase para manejar métricas del bot"""

    def __init__(self):
        self.alerts_sent = 0
        self.errors_count = 0
        self.last_update = None
        self.start_time = dt.datetime.now()
        self.data_updates = 0
        self.calculations = 0

    def record_alert(self, ticker: str, timeframe: str):
        """Registra una alerta enviada"""
        self.alerts_sent += 1

    def record_error(self, error_type: str):
        """Registra un error"""

        self.errors_count += 1

    def record_data_update(self, ticker: str, timeframe: str, rows: int):
        self.data_updates += 1
        self.last_update = dt.datetime.now()

    def record_calculation(self, ticker: str, timeframe: str):
        """Registra un cálculo de indicador"""
        self.calculations += 1

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del bot"""
        uptime = dt.datetime.now() - self.start_time
        return {
            "uptime_seconds": uptime.total_seconds(),
            "alerts_sent": self.alerts_sent,
            "errors_count": self.errors_count,
            "data_updates": self.data_updates,
            "calculations": self.calculations,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


class DatabaseManager:
    """Manejador de base de datos"""

    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.db_file = Path(config.database_file)

    def get_conn(self):
        """Obtiene conexión a la base de datos con manejo de errores mejorado"""
        try:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            self.logger.error(f"Error al conectar a la base de datos: {e}")
            raise

    def init_db(self):
        """Inicializa la base de datos con manejo de errores robusto"""
        try:
            with self.get_conn() as conn:
                cur = conn.cursor()
                
                # Tabla unificada para precios con múltiples timeframes
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        date TEXT NOT NULL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        adj_close REAL,
                        volume INTEGER,
                        created_at TEXT NOT NULL,
                        UNIQUE(ticker, timeframe, date)
                    )
                    """
                )

                # Tabla para alertas accionables
                # INFO: Esta tabla es la que se usa para las alertas accionables de multi-timeframe
                # Variables:
                # - alert_id: id único, p.ej. candidate_entry_AAPL_20250830
                # - ticker: ticker
                # - action: ENTRY_HIGH / ENTRY_MED / EXIT / WATCH / NO_ENTRY
                # - alert_type: candidate_entry / portfolio_exit / state_change ...
                # - score: score calculado (opcional)
                # - priority: 'Alta'|'Media'|'Baja'
                # - states_json: JSON con daily/weekly/monthly states, bx_values, dates, vols
                # - note: motivos humanos / resumen
                # - created_at: creación en UTC iso
                # - triggered_at: fecha del cierre que generó la alerta (ISO)
                # - sent: 0 no enviado, 1 enviado
                # - sent_at: cuando se envió (UTC)
                # - cooldown_until: ISO hasta cuando no re-alertar
                # - UNIQUE(ticker, alert_type, triggered_at): evita duplicados por la misma señal
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS alerts_v2 (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT UNIQUE,
                        ticker TEXT NOT NULL,
                        action TEXT,
                        alert_type TEXT,
                        score REAL,
                        priority TEXT,
                        states_json TEXT,
                        note TEXT,
                        created_at TEXT NOT NULL,
                        triggered_at TEXT,
                        sent INTEGER DEFAULT 0,
                        sent_at TEXT,
                        cooldown_until TEXT,
                        UNIQUE(ticker, alert_type, triggered_at)
                    )
                    """
                )
                
                # Tabla de state (último color conocido / estado)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS state (
                        ticker TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        last_color TEXT,
                        updated_at TEXT,
                        PRIMARY KEY (ticker, timeframe)
                    )
                    """
                )
                
                # Índices para optimizar consultas
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_prices_ticker_timeframe_date ON prices(ticker, timeframe, date)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_prices_timeframe_date ON prices(timeframe, date)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_alerts_v2_ticker_action ON alerts_v2(ticker, action)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_alerts_v2_created_at ON alerts_v2(created_at)"
                )
                conn.commit()
                self.logger.info("Base de datos inicializada correctamente")
        except Exception as e:
            self.logger.error(f"Error al inicializar la base de datos: {e}")
            raise

    def get_data(self, ticker: str, timeframe: str = "1d") -> Optional[pd.DataFrame]:
        """Obtiene datos de un ticker y timeframe específico con validación mejorada"""
        try:
            with self.get_conn() as conn:
                df = pd.read_sql(
                    """
                    SELECT date, open, high, low, close, adj_close, volume
                    FROM prices 
                    WHERE ticker = ? AND timeframe = ?
                    ORDER BY date
                    """,
                    conn,
                    params=[ticker, timeframe],
                    index_col="date",
                    parse_dates=["date"],
                )
                
                # Validación de datos
                if df is None or df.empty:
                    self.logger.warning(
                        f"No se encontraron datos para {ticker} {timeframe}"
                    )
                    return None
                required_columns = ["open", "high", "low", "close"]
                missing_columns = [
                    col for col in required_columns if col not in df.columns
                ]
                if missing_columns:
                    self.logger.error(
                        f"Columnas faltantes para {ticker} {timeframe}: {missing_columns}"
                    )
                    return None
                # Normalizar nombres para cálculo
                if "adj_close" in df.columns:
                    df = df.rename(columns={"adj_close": "Adj Close"})
                if "close" in df.columns:
                    df = df.rename(columns={"close": "Close"})
                self.logger.debug(
                    f"Datos cargados para {ticker} {timeframe}: {len(df)} filas"
                )
                return df
        except Exception as e:
            self.logger.error(f"Error al leer datos para {ticker} {timeframe}: {e}")
            return None

    def save_data(self, df: pd.DataFrame, ticker: str, timeframe: str):
        """Guarda DataFrame en la tabla unificada con validación"""
        if df is None or df.empty:
            self.logger.warning(
                f"DataFrame vacío para {ticker} {timeframe}, no se guarda"
            )
            return
        try:
            df_to_save = df.copy()
            
            # Normalizar nombres de columnas (yfinance usa "Adj Close", tabla usa "adj_close")
            rename_map = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
                "adj_close": "adj_close",
            }
            df_to_save.rename(columns=rename_map, inplace=True)
            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                if col not in df_to_save.columns:
                    df_to_save[col] = None
            df_to_save = df_to_save.reset_index()
            date_col = df_to_save.columns[0]
            df_to_save["date"] = pd.to_datetime(df_to_save[date_col]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            df_to_save["ticker"] = ticker
            df_to_save["timeframe"] = timeframe
            df_to_save["created_at"] = dt.datetime.utcnow().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            records = df_to_save[
                [
                    "ticker",
                    "timeframe",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                    "created_at",
                ]
            ].to_records(index=False)
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.executemany(
                    """
                    INSERT OR IGNORE INTO prices
                    (ticker, timeframe, date, open, high, low, close, adj_close, volume, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )
                conn.commit()
                self.logger.debug(
                    f"Guardados {cur.rowcount} registros para {ticker} {timeframe} (INSERT OR IGNORE)"
                )
        except Exception as e:
            self.logger.error(f"Error al guardar datos para {ticker} {timeframe}: {e}")
            raise

    def get_last_color(self, ticker: str, timeframe: str) -> Optional[str]:
        """Obtiene el último color conocido desde la base de datos"""
        try:
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT last_color FROM state WHERE ticker = ? AND timeframe = ?",
                    (ticker, timeframe),
                )
                r = cur.fetchone()
                return r[0] if r else None
        except Exception as e:
            self.logger.error(
                f"Error al obtener último color para {ticker} {timeframe}: {e}"
            )
            return None

    def set_last_color(self, ticker: str, timeframe: str, color: str):
        """Actualiza el último color conocido en la base de datos"""
        try:
            now = dt.datetime.utcnow().isoformat()
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT OR REPLACE INTO state (ticker, timeframe, last_color, updated_at) VALUES (?, ?, ?, ?)",
                    (ticker, timeframe, color, now),
                )
                conn.commit()
                self.logger.debug(
                    f"Estado actualizado para {ticker} {timeframe}: {color}"
                )
        except Exception as e:
            self.logger.error(
                f"Error al actualizar estado para {ticker} {timeframe}: {e}"
            )
            raise

    def save_alert_v2(
        self,
        alert_id: str,
        ticker: str,
        action: str,
        alert_type: str,
        states: dict,
        score: Optional[float] = None,
        priority: Optional[str] = None,
        note: Optional[str] = None,
        triggered_at: Optional[str] = None,
    ) -> bool:
        """
        Inserta una alerta multi-timeframe en alerts_v2.
        Retorna True si se insertó (alerta nueva), False si ya existía (duplicada/ignored).
        """
        try:
            now = dt.datetime.utcnow().isoformat()
            states_json = json.dumps(states, default=str, ensure_ascii=False)
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT OR IGNORE INTO alerts_v2
                    (alert_id, ticker, action, alert_type, score, priority, states_json, note, created_at, triggered_at, sent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (alert_id, ticker, action, alert_type, score, priority, states_json, note, now, triggered_at),
                )
                conn.commit()
                inserted = cur.rowcount > 0
                if inserted:
                    self.logger.debug(f"Alerta multi-timeframe insertada: {alert_id} / {ticker} / {action}")
                else:
                    self.logger.debug(f"Alerta multi-timeframe DUPLICADA (ignorada): {alert_id}")
                return inserted
        except Exception as e:
            self.logger.error(f"Error al guardar alerta multi-timeframe {alert_id}: {e}")
            raise


class IndicatorCalculator:
    """Calculador de indicadores técnicos"""

    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def _ema(self, series: pd.Series, length: int) -> pd.Series:
        """Calcula EMA con fallback robusto"""
        try:
            out = ta.ema(series, length=length)
        except Exception as e:
            self.logger.debug(f"EMA fallback para length={length}: {e}")
            out = None
        if out is None or (isinstance(out, pd.Series) and out.isna().all()):
            self.logger.debug(f"Usando fallback manual para EMA length={length}")
            out = series.ewm(span=length, adjust=False).mean()
        return out.reindex(series.index)

    def _rsi(self, series: pd.Series, length: int) -> pd.Series:
        """Calcula RSI con fallback robusto"""
        try:
            out = ta.rsi(series, length=length)
        except Exception as e:
            self.logger.debug(f"RSI fallback para length={length}: {e}")
            out = None
        if out is None or (isinstance(out, pd.Series) and out.isna().all()):
            self.logger.debug(f"Usando fallback manual para RSI length={length}")
            delta = series.diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ma_up = up.ewm(alpha=1 / length, adjust=False).mean()
            ma_down = down.ewm(alpha=1 / length, adjust=False).mean()
            rs = ma_up / ma_down
            out = 100 - (100 / (1 + rs))
        return out.reindex(series.index)

    def _t3(self, series: pd.Series, length: int, v: float) -> pd.Series:
        """Calcula T3 con fallback robusto"""
        try:
            out = ta.t3(series, length=length, v=v)
        except Exception as e:
            self.logger.debug(f"T3 fallback para length={length}, v={v}: {e}")
            out = None
        if out is None or (isinstance(out, pd.Series) and out.isna().all()):
            self.logger.debug(f"Usando fallback manual para T3 length={length}, v={v}")
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

    def compute_bx_trender(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula short_xtrender (RSI sobre EMA diff) y long_xtrender (RSI sobre EMA),
        aplica T3 al short (ma_short_xtrender) y devuelve columnas:
          - short_xtrender, ma_short_xtrender, bx_value
          - long_xtrender
          - bx_color (simple green/red según ma_short suba/baje)
          - bx_state (== short_state) y long_state (ambos con las 4 variantes)
        """
        try:
            dfc = df.copy()
            if "Close" not in dfc.columns:
                raise RuntimeError("DataFrame sin columna Close")
            
            # Close como serie limpia
            close = pd.to_numeric(dfc["Close"], errors="coerce").ffill()
            min_required = (
                max(
                    self.config.short_l2,
                    self.config.short_l3,
                    self.config.t3_length,
                    self.config.long_l1,
                    self.config.long_l2,
                )
                + 20
            )
            if close.dropna().shape[0] < min_required:
                raise RuntimeError(
                    f"Datos insuficientes para calcular indicadores. Se requieren al menos {min_required} puntos de datos, pero solo hay {close.dropna().shape[0]}"
                )
            self.logger.debug(
                f"Calculando BX Trender para {len(close)} puntos de datos"
            )
            # --- short term (igual que Pine Script de TradingView de BxTrender) ---
            ema1 = self._ema(close, self.config.short_l1)
            ema2 = self._ema(close, self.config.short_l2)
            diff = ema1 - ema2
            rsi_diff = self._rsi(diff, self.config.short_l3)
            short_term = rsi_diff - 50
            # ma (T3) sobre shortTerm (como en Pine Script de BX Trender)
            ma_short = self._t3(
                short_term.ffill(), self.config.t3_length, self.config.t3_v
            )
            # --- long term (igual que Pine Script de BX Trender) ---
            ema_long = self._ema(close, self.config.long_l1)
            long_term = self._rsi(ema_long, self.config.long_l2) - 50
            # Volcamos columnas
            dfc["short_xtrender"] = short_term
            dfc["ma_short_xtrender"] = ma_short
            dfc["bx_value"] = dfc["ma_short_xtrender"]
            dfc["long_xtrender"] = long_term
            
            # bx_color simple: verde si ma_short sube vs anterior, rojo si baja
            def color_from_ma_diff(x):
                if pd.isna(x):
                    return None
                return "green" if x > 0 else ("red" if x < 0 else None)

            dfc["bx_color"] = dfc["ma_short_xtrender"].diff().apply(color_from_ma_diff)
            # ----- 4 estados (exacto comportamiento Pine) -----
            # Para short: usamos short_xtrender comparado con su previo
            prev_short = dfc["short_xtrender"].shift(1)

            def short_state_from(curr, prev_val):
                if pd.isna(curr) or pd.isna(prev_val):
                    return None
                if curr > 0:
                    # curr > 0: positive -> green shades
                    return "green_hh" if curr > prev_val else "green_lh"
                else:
                    # curr <= 0: negative -> red shades
                    return "red_hl" if curr > prev_val else "red_ll"

            dfc["short_state"] = [
                short_state_from(c, p)
                for c, p in zip(dfc["short_xtrender"].tolist(), prev_short.tolist())
            ]
            # Para long: mismo criterio aplicado a long_xtrender
            prev_long = dfc["long_xtrender"].shift(1)

            def long_state_from(curr, prev_val):
                if pd.isna(curr) or pd.isna(prev_val):
                    return None
                if curr > 0:
                    return "green_hh" if curr > prev_val else "green_lh"
                else:
                    return "red_hl" if curr > prev_val else "red_ll"

            dfc["long_state"] = [
                long_state_from(c, p)
                for c, p in zip(dfc["long_xtrender"].tolist(), prev_long.tolist())
            ]
            dfc["bx_state"] = dfc["short_state"]
            self.logger.debug("Cálculo de BX Trender completado exitosamente")
            return dfc
        except Exception as e:
            self.logger.error(f"Error en cálculo de BX Trender: {e}")
            raise


class DataManager:
    """Manejador de datos de mercado"""

    def __init__(
        self, config: BotConfig, db_manager: DatabaseManager, logger: logging.Logger
    ):
        self.config = config
        self.db_manager = db_manager
        self.logger = logger

    def backfill_ticker(self, ticker: str, timeframe: str, years: int = None):
        """Backfill inicial para un ticker y timeframe específico con manejo de errores mejorado"""
        
        if years is None:
            # Usar configuración específica por timeframe
            years = self.config.backfill_years.get(timeframe, 4) # Default a 4 años si no está configurado
        period = f"{years}y"
        self.logger.info(
            f"Iniciando backfill para {ticker} {timeframe} period={period} ({years} años)"
        )
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=timeframe,
                auto_adjust=False,
                multi_level_index=False,
                progress=False,
            )
            if df is None or df.empty:
                self.logger.warning(
                    f"Backfill: no se obtuvieron datos para {ticker} {timeframe}"
                )
                return
            # Validar datos mínimos
            if len(df) < 10:
                self.logger.warning(
                    f"Backfill: datos insuficientes para {ticker} {timeframe} ({len(df)} filas)"
                )
                return
            
            df.index = pd.to_datetime(df.index)
            self.db_manager.save_data(df, ticker, timeframe)
            self.logger.info(
                f"Backfill completado para {ticker} {timeframe}: {len(df)} filas guardadas"
            )
            
        except Exception as e:
            self.logger.error(f"Error en backfill para {ticker} {timeframe}: {e}")
            raise

    def incremental_update_ticker(
        self, ticker: str, timeframe: str, period: str = None
    ):
        """Actualización incremental para un ticker y timeframe específico"""
        
        if period is None:
            period = self.config.update_period
        self.logger.info(
            f"Iniciando actualización incremental para {ticker} {timeframe} period={period}"
        )
        try:
            existing = self.db_manager.get_data(ticker, timeframe)
            df_new = yf.download(
                ticker,
                period=period,
                interval=timeframe,
                auto_adjust=False,
                multi_level_index=False,
                progress=False,
            )
            if df_new is None or df_new.empty:
                self.logger.warning(
                    f"No se obtuvieron datos incrementales para {ticker} {timeframe}"
                )
                return
            df_new.index = pd.to_datetime(df_new.index)
            if existing is None or existing.empty:
                merged = df_new
                self.logger.info(f"Primera carga de datos para {ticker} {timeframe}")
            else:
                # Append only rows after last existing index
                new_data = df_new[~df_new.index.isin(existing.index)]
                if new_data.empty:
                    self.logger.info(f"No hay datos nuevos para {ticker} {timeframe}")
                    return
                merged = pd.concat([existing, new_data])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            # Guardar solo los datos nuevos
            self.db_manager.save_data(df_new, ticker, timeframe)
            self.logger.info(
                f"Actualización incremental completada para {ticker} {timeframe}: {len(df_new)} nuevas filas, última fecha: {df_new.index.max().date()}"
            )
        except Exception as e:
            self.logger.error(
                f"Error en actualización incremental para {ticker} {timeframe}: {e}"
            )
            raise


class NotificationBatch:
    """Clase para manejar el batching de notificaciones"""
    
    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.pending_alerts: List[Dict[str, Any]] = []
        self.last_batch_time = dt.datetime.now()
        self.daily_alerts: List[Dict[str, Any]] = []
        self.last_summary_date = dt.datetime.now().date()

    def add_alert(
    self, ticker: str, timeframe: str, date: str, state: str, value: Optional[float]
    ) -> bool:
        """
        Añade una alerta al batch y retorna True si debe enviarse inmediatamente.

        - Mantiene pending_alerts y daily_alerts (para resumen).
        - Ya no usa config.critical_states.
        - Reglas de envío inmediato (por defecto):
            * timeframe == "multi" y state == "EXIT"  -> enviar inmediatamente
            * timeframe == "multi" y state == "ENTRY_HIGH" -> enviar inmediatamente
            * timeframe == "multi" y state == "ENTRY_MED"  -> agregar al batch (no inmediato)
            * timeframe != "multi" (single-TF) -> no envía inmediato (solo se acumula para resumen)
        - Si querés forzar comportamiento distinto, llamá a add_alert con timeframe="multi"
        y state apropiado (ENTRY_HIGH/ENTRY_MED/EXIT).
        """
        alert = {
            "ticker": ticker,
            "timeframe": timeframe,
            "date": date,
            "state": state,
            "value": value,
            "timestamp": dt.datetime.now(),
        }

        self.pending_alerts.append(alert)
        self.daily_alerts.append(alert)

        # Decidir envío inmediato según reglas multi-TF
        try:
            # Si es una alerta multi-timeframe (la que nos importa ahora)
            if timeframe == "multi":
                action = (state or "").upper()
                if action == "EXIT":
                    # Salidas del portafolio -> enviar ahora (critico)
                    self.logger.info(f"Multi-TF EXIT detectado: {ticker} -> {state}")
                    return True
                if action == "ENTRY_HIGH":
                    # Entradas de alta prioridad -> enviar ahora
                    self.logger.info(f"Multi-TF ENTRY_HIGH detectado: {ticker} -> {state}")
                    return True
                # ENTRY_MED u otras acciones multi-TF se acumulan en batch
                return False

            # Para single-timeframe: no enviar inmediato (solo resumir).
            return False

        except Exception as e:
            # Si algo falla, no bloqueamos: no enviamos inmediatamente y dejamos en batch
            self.logger.debug("add_alert error al decidir envío inmediato: %s", e)
            return False

    def should_send_batch(self) -> bool:
        """Determina si se debe enviar el batch actual"""
        
        if not self.pending_alerts:
            return False
        # Enviar si alcanzamos el máximo de alertas por batch
        if len(self.pending_alerts) >= self.config.max_alerts_per_batch:
            return True
        
        # Enviar si ha pasado el timeout
        time_since_last = (dt.datetime.now() - self.last_batch_time).total_seconds()
        if time_since_last >= self.config.batch_timeout_seconds:
            return True
        return False

    def get_batch_message(self) -> str:
        """Genera el mensaje del batch actual agrupado por timeframe y luego por color"""
        
        if not self.pending_alerts:
            return ""
        human_map = {
            "green_hh": "🟢⬆️ GREEN HH",
            "green_lh": "🟢⬇️ GREEN LH",
            "red_hl": "🟠⬆️ RED HL",
            "red_ll": "🔴⬇️ RED LL",
        }
        
        # Agrupar primero por timeframe, luego por estado
        alerts_by_timeframe = {}
        for alert in self.pending_alerts:
            timeframe = alert["timeframe"]
            state = alert["state"]
            alerts_by_timeframe.setdefault(timeframe, {})
            alerts_by_timeframe[timeframe].setdefault(state, [])
            alerts_by_timeframe[timeframe][state].append(alert)
        # Construir mensaje
        lines = ["📊 *RESUMEN DE ALERTAS - BX Trender*"]
        # Ordenar timeframes (1d, 1wk, 1mo)
        timeframe_order = ["1d", "1wk", "1mo"]
        sorted_timeframes = sorted(
            alerts_by_timeframe.keys(),
            key=lambda x: timeframe_order.index(x) if x in timeframe_order else 999,
        )
        for timeframe in sorted_timeframes:
            lines.append(f"\n*{timeframe.upper()}*")
            # Ordenar estados (green_hh, green_lh, red_hl, red_ll)
            state_order = ["green_hh", "green_lh", "red_hl", "red_ll"]
            sorted_states = sorted(
                alerts_by_timeframe[timeframe].keys(),
                key=lambda x: state_order.index(x) if x in state_order else 999,
            )
            for state in sorted_states:
                alerts = alerts_by_timeframe[timeframe][state]
                human_state = human_map.get(state, state)
                lines.append(f"  {human_state} ({len(alerts)}):")
                for alert in alerts:
                    value_str = (
                        f" ({alert['value']:.4f})" if alert["value"] is not None else ""
                    )
                    lines.append(
                        f"    • {alert['ticker']} - {alert['date']}{value_str}"
                    )
        lines.append(f"\n_Generated at {dt.datetime.now().strftime('%H:%M:%S')}_")
        return "\n".join(lines)

    def get_daily_summary(self) -> str:
        """Genera el resumen diario de alertas agrupado por timeframe"""
        if not self.daily_alerts:
            return ""
        # Filtrar alertas del día actual
        today = dt.datetime.now().date()
        today_alerts = [a for a in self.daily_alerts if a["timestamp"].date() == today]
        if not today_alerts:
            return ""
        human_map = {
            "green_hh": "🟢⬆️ GREEN HH",
            "green_lh": "🟢⬇️ GREEN LH",
            "red_hl": "🟠⬆️ RED HL",
            "red_ll": "🔴⬇️ RED LL",
        }
        alerts_by_timeframe = {}
        for alert in today_alerts:
            timeframe = alert["timeframe"]
            state = alert["state"]
            alerts_by_timeframe.setdefault(timeframe, {})
            alerts_by_timeframe[timeframe].setdefault(state, 0)
            alerts_by_timeframe[timeframe][state] += 1
        lines = [f"📈 *RESUMEN DAILY - {today.strftime('%Y-%m-%d')}*"]
        lines.append(f"Total signals: {len(today_alerts)}")
        timeframe_order = ["1d", "1wk", "1mo"]
        sorted_timeframes = sorted(
            alerts_by_timeframe.keys(),
            key=lambda x: timeframe_order.index(x) if x in timeframe_order else 999,
        )
        for timeframe in sorted_timeframes:
            lines.append(f"\n*{timeframe.upper()}*")
            state_order = ["green_hh", "green_lh", "red_hl", "red_ll"]
            sorted_states = sorted(
                alerts_by_timeframe[timeframe].keys(),
                key=lambda x: state_order.index(x) if x in state_order else 999,
            )
            for state in sorted_states:
                count = alerts_by_timeframe[timeframe][state]
                human_state = human_map.get(state, state)
                lines.append(f"  {human_state}: {count}")
        ticker_counts = {}
        for alert in today_alerts:
            ticker = alert["ticker"]
            ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
        if ticker_counts:
            top_tickers = sorted(
                ticker_counts.items(), key=lambda x: x[1], reverse=True
            )[:3]
            lines.append(f"\n*Most Active Tickers:*")
            for ticker, count in top_tickers:
                lines.append(f"• {ticker}: {count} signals")
        return "\n".join(lines)

    def clear_batch(self):
        """Limpia el batch actual"""
        self.pending_alerts = []
        self.last_batch_time = dt.datetime.now()

    def should_send_daily_summary(self) -> bool:
        """Determina si se debe enviar el resumen diario"""
        if not self.config.summary_enabled:
            return False
        
        try:
            current_time = dt.datetime.now().time()
            summary_hour, summary_minute = map(int, self.config.summary_time.split(":"))
            summary_time = dt.time(summary_hour, summary_minute)
            # Enviar si es la hora del resumen y no se ha enviado hoy
            if (
                current_time.hour == summary_time.hour
                and current_time.minute == summary_time.minute
                and dt.datetime.now().date() != self.last_summary_date
            ):
                return True
        except Exception as e:
            self.logger.error(f"Error al verificar hora del resumen: {e}")
        return False

    def mark_summary_sent(self):
        """Marca que se envió el resumen diario"""
        self.last_summary_date = dt.datetime.now().date()


class NotificationManager:
    """Manejador de notificaciones"""
    
    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.bot: Optional[Bot] = None
        self.batch_manager = NotificationBatch(config, logger)

    async def initialize_bot(self):
        """Inicializa el bot de Telegram"""
        if not self.config.telegram_token:
            self.logger.warning("TELEGRAM_TOKEN no configurado.")
            return False
        try:
            self.bot = Bot(token=self.config.telegram_token)
            await self.bot.get_me()
            self.logger.info("Telegram Bot inicializado correctamente.")
            return True
        except Exception as e:
            self.bot = None
            self.logger.error("No se pudo inicializar Telegram Bot: %s", e)
            return False

    async def send_message(self, msg: str) -> bool:
        """Envía mensaje por Telegram con retry"""
        if not self.bot:
            self.logger.warning(
                "Bot no inicializado. Mensaje no enviado: %s",
                msg[:100] + "..." if len(msg) > 100 else msg,
            )
            return False
        if not self.config.telegram_chat_id:
            self.logger.warning(
                "TELEGRAM_CHAT_ID no configurado. Mensaje no enviado: %s",
                msg[:100] + "..." if len(msg) > 100 else msg,
            )
            return False
        try:
            chat_id = int(self.config.telegram_chat_id)
        except Exception as e:
            self.logger.error(
                "TELEGRAM_CHAT_ID inválido: %s - %s", self.config.telegram_chat_id, e
            )
            return False
        last_exc = None
        for attempt in range(1, self.config.telegram_max_retries + 1):
            try:
                res = await self.bot.send_message(chat_id=chat_id, text=msg)
                self.logger.info(
                    "✅ Mensaje enviado (intento %d). message_id=%s",
                    attempt,
                    getattr(res, "message_id", None),
                )
                return True
            except Exception as e:
                last_exc = e
                self.logger.warning(
                    "Intento %d falló al enviar Telegram: %s", attempt, e
                )
                if attempt < self.config.telegram_max_retries:
                    await asyncio.sleep(self.config.telegram_retry_delay_base**attempt)
        self.logger.error(
            "No se pudo enviar mensaje por telegram tras %d intentos. Excepción: %s",
            self.config.telegram_max_retries,
            last_exc,
        )
        return False


class BXTrenderBot:
    """Clase principal del BX Trender Bot"""

    def __init__(self, config_file: str = "config.yaml"):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        self.metrics = Metrics()
        self.db_manager = DatabaseManager(self.config, self.logger)
        self.indicator_calc = IndicatorCalculator(self.config, self.logger)
        self.data_manager = DataManager(self.config, self.db_manager, self.logger)
        self.notification_manager = NotificationManager(self.config, self.logger)

    def _load_config(self, config_file: str) -> BotConfig:
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                
            # Cargar variables de entorno para Telegram
            telegram_token = os.getenv(
                "TELEGRAM_TOKEN", config_data.get("telegram", {}).get("token", "")
            ).strip()
            telegram_chat_id = os.getenv(
                "TELEGRAM_CHAT_ID", config_data.get("telegram", {}).get("chat_id", "")
            ).strip()
            
            # Cargar configuración de batching con valores por defecto
            batching_config = config_data.get("telegram", {}).get("batching", {})
            trading = config_data.get("trading", {})
            confirm = config_data.get("confirmations", {})
            alerts_config = config_data.get("alerts", {})
            
            return BotConfig(
                database_file=config_data["database"]["file"],
                backfill_years=config_data["database"]["backfill_years"],
                update_period=config_data["database"]["update_period"],
                request_delay=config_data["database"]["request_delay"],
                timeframes=config_data["trading"].get(
                    "timeframes", ["1d", "1wk", "1mo"]
                ),
                #Lista de acciones candidatas para el screener de BX Trender
                candidates=trading.get("candidates", trading.get("candidates", [])),
                #Lista de acciones actuales en el portafolio
                portfolio=trading.get("portfolio", []),
                short_l1=config_data["indicators"]["short_l1"],
                short_l2=config_data["indicators"]["short_l2"],
                short_l3=config_data["indicators"]["short_l3"],
                t3_length=config_data["indicators"]["t3_length"],
                t3_v=config_data["indicators"]["t3_v"],
                long_l1=config_data["indicators"]["long_l1"],
                long_l2=config_data["indicators"]["long_l2"],
                telegram_token=telegram_token,
                telegram_chat_id=telegram_chat_id,
                telegram_max_retries=config_data["telegram"]["max_retries"],
                telegram_retry_delay_base=config_data["telegram"]["retry_delay_base"],
                # Configuración de batching
                batching_enabled=batching_config.get("enabled", True),
                max_alerts_per_batch=batching_config.get("max_alerts_per_batch", 5),
                batch_timeout_seconds=batching_config.get("batch_timeout_seconds", 30),
                summary_enabled=batching_config.get("summary_enabled", True),
                summary_time=batching_config.get("summary_time", "18:00"),
                log_level=config_data["logging"]["level"],
                log_file_level=config_data["logging"]["file_level"],
                log_console_level=config_data["logging"]["console_level"],
                log_dir=config_data["logging"]["log_dir"],
                max_log_files=config_data["logging"]["max_log_files"],
                health_check_interval=config_data["monitoring"][
                    "health_check_interval"
                ],
                metrics_enabled=config_data["monitoring"]["metrics_enabled"],
                alert_on_errors=config_data["monitoring"]["alert_on_errors"],
                confirm_daily=confirm.get("daily", 2),
                confirm_weekly=confirm.get("weekly", 1),
                confirm_monthly=confirm.get("monthly", 1),
                bx_value_min_abs=confirm.get("bx_value_min_abs", 0.0),
                volume_min_factor=confirm.get("volume_min_factor", 0.5),
                alert_cooldown_days=confirm.get("alert_cooldown_days", 1),
                alert_types=alerts_config.get("types", {
                    "high": "ALTA - Revisar para entrada",
                    "medium": "MEDIA - Vigilancia, considerar",
                    "critical": "CRÍTICA - Considerar salida/reducir",
                    "noentry": "NO ENTRAR - No cumple requisitos para entrada",
                    "info": "INFO - Cambios menores, sin acción"
                }),
                re_alert_cooldown_hours=alerts_config.get("re_alert_cooldown_hours", 24),
            )
        except Exception as e:
            raise RuntimeError(
                f"Error al cargar configuración desde {config_file}: {e}"
            )

    def _setup_logging(self) -> logging.Logger:
        """Configura logging estructurado con archivo y consola"""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(exist_ok=True)
        # Configurar formato
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Handler para archivo
        file_handler = logging.FileHandler(
            log_dir / f"trading_bot_{dt.datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(getattr(logging, self.config.log_file_level))
        file_handler.setFormatter(formatter)
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.log_console_level))
        console_handler.setFormatter(formatter)
        
        # Configurar logger principal
        logger = logging.getLogger('BXTrenderBot')
        logger.setLevel(getattr(logging, self.config.log_level))
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Evitar duplicación de logs
        logger.propagate = False
        return logger

    async def health_check(self) -> bool:
        """Verifica que todos los componentes estén funcionando"""
        try:
            # Verificar base de datos
            self.db_manager.get_conn().close()
            
            # Verificar bot de Telegram
            if self.notification_manager.bot:
                await self.notification_manager.bot.get_me()
            # Verificar datos recientes
            for ticker in self.config.candidates[:1] + self.config.portfolio[:1]:
                for timeframe in self.config.timeframes[:1]:
                    data = self.db_manager.get_data(ticker, timeframe)
                    if data is None or data.empty:
                        self.logger.warning(
                            "Health check: No hay datos recientes para %s %s",
                            ticker,
                            timeframe,
                        )
                        return False
            self.logger.info(
                "Health check: Todos los componentes funcionando correctamente"
            )
            return True
        except Exception as e:
            self.logger.error("Health check falló: %s", e)
            return False

    async def run_backfill_and_updates(self):
        """Backfill y actualización para múltiples tickers y timeframes con manejo de errores mejorado"""
        needs_backfill = False
        if not self.db_manager.db_file.exists():
            needs_backfill = True
            self.logger.info("Base de datos no existe, iniciando backfill completo")
        else:
            # Chequea si tenemos data para el primer ticker y timeframe
            try:
                tbl = self.db_manager.get_data(
                    self.config.candidates[0]
                    if self.config.candidates
                    else self.config.portfolio[0],
                    self.config.timeframes[0],
                )
                if tbl is None or tbl.empty:
                    needs_backfill = True
                    self.logger.info(
                        "Base de datos existe pero sin datos, iniciando backfill"
                    )
            except Exception as e:
                self.logger.warning(
                    "Error al verificar datos existentes, iniciando backfill: %s", e
                )
                needs_backfill = True
        if needs_backfill:
            self.logger.info("Iniciando backfill inicial para todos los timeframes...")
            tickers_all = list(
                set(
                    (self.config.candidates or [])
                    + (self.config.portfolio or [])
                )
            )
            for t in tickers_all:
                for tf in self.config.timeframes:
                    try:
                        await asyncio.to_thread(
                            self.data_manager.backfill_ticker, t, tf
                        )
                        await asyncio.sleep(self.config.request_delay)
                    except Exception as e:
                        self.logger.error("Error en backfill para %s %s: %s", t, tf, e)
                        self.metrics.record_error("backfill")
                        continue
        else:
            self.logger.info("Base de datos existente con datos. Saltando backfill.")
        self.logger.info("Iniciando actualización incremental...")
        tickers_all = list(
            set(
                (self.config.candidates or [])
                + (self.config.portfolio or [])
            )
        )
        for t in tickers_all:
            for tf in self.config.timeframes:
                try:
                    await asyncio.to_thread(
                        self.data_manager.incremental_update_ticker, t, tf
                    )
                    await asyncio.sleep(self.config.request_delay)
                except Exception as e:
                    self.logger.error(
                        "Error en actualización incremental para %s %s: %s", t, tf, e
                    )
                    self.metrics.record_error("update")
                    continue

    # ---- Lógica de evaluación / confirmaciones ----
    def _sustained_state(
        self, ticker: str, timeframe: str, required: int
    ) -> Tuple[bool, Optional[str]]:
        """Devuelve (bool, state) si el mismo bx_state se repite en las últimas `required` barras."""
        df = self.db_manager.get_data(ticker, timeframe)
        if df is None:
            return False, None
        dfc = self.indicator_calc.compute_bx_trender(df)
        dfc = dfc.dropna(subset=["bx_state"])
        if len(dfc) < required:
            return False, None
        last_states = dfc["bx_state"].iloc[-required:].tolist()
        if all(s == last_states[0] for s in last_states):
            return True, last_states[0]
        return False, None

    def _latest_state(self, ticker: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Retorna el último estado conocido para el ticker/timeframe.
        Devuelve None si no hay datos o no hay columna de estado.
        """
        
        df = self.db_manager.get_data(ticker, timeframe)
        if df is None:
            return None
        dfc = self.indicator_calc.compute_bx_trender(df)
        dfc_valid = (
            dfc.dropna(subset=["bx_state", "bx_value"])
            if "bx_value" in dfc.columns
            else dfc.dropna(subset=["bx_state"])
        )
        if dfc_valid.empty:
            return None
        last_idx = dfc_valid.index[-1]
        return {
            "date": last_idx,
            "bx_state": dfc_valid.loc[last_idx, "bx_state"],
            "bx_value": float(dfc_valid.loc[last_idx, "bx_value"])
            if "bx_value" in dfc_valid.columns
            else None,
        }


    def _volume_metrics(self, ticker: str, timeframe: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Retorna (last_volume, median20_volume) para el ticker/timeframe.
        Devuelve (None, None) si no hay datos o no hay columna de volumen.
        """
        df = self.db_manager.get_data(ticker, timeframe)
        if df is None or df.empty:
            return None, None

        vol_col = None
        for c in ("volume", "Volume"):
            if c in df.columns:
                vol_col = c
                break
        if vol_col is None:
            return None, None

        vols = pd.to_numeric(df[vol_col], errors="coerce").dropna()
        if vols.empty:
            return None, None

        last = float(vols.iloc[-1])
        # usar median20 sólo si hay al menos 20 puntos, si no usar la mediana disponible
        median20 = float(vols.tail(20).median()) if len(vols) >= 20 else float(vols.median())
        
        return last, median20


    def _pattern_score(
        self,
        consecutive_daily_state: Optional[str],
        consecutive_weekly_state: Optional[str],
        consecutive_monthly_state: Optional[str],
        latest_daily_state: Optional[Dict[str, Any]],
        latest_weekly_state: Optional[Dict[str, Any]],
        latest_monthly_state: Optional[Dict[str, Any]],
        consecutive_daily_ok: bool = False,
        consecutive_weekly_ok: bool = False,
        consecutive_monthly_ok: bool = False
    ) -> Tuple[float, List[str]]:
        """
        Evalúa objetivamente una combinación multi-timeframe y devuelve (score, reasons).
        Aquí se aplican TODOS los bonus y penalizaciones (volumen, bx_value, pullback, confirmaciones).
        """
        reasons: List[str] = []

        base = {"green_hh": 3.0, "green_lh": 1.5, "red_hl": -0.5, "red_ll": -2.0, None: 0.0}

        # Pesos por timeframe
        weight_daily, weight_weekly, weight_monthly = 1.0, 1.2, 1.5

        # Puntajes base por timeframe
        score_daily = base.get(consecutive_daily_state, 0.0) * weight_daily
        score_weekly = base.get(consecutive_weekly_state, 0.0) * weight_weekly
        score_monthly = base.get(consecutive_monthly_state, 0.0) * weight_monthly
        reasons.append(f"base_scores: D={score_daily:.2f}, W={score_weekly:.2f}, M={score_monthly:.2f}")

        score = score_daily + score_weekly + score_monthly

        # Bonus por magnitud de bx_value (saturado)
        def bx_bonus(x: Optional[float]) -> float:
            if x is None:
                return 0.0
            return 0.5 * (abs(x) / (1.0 + abs(x)))  # valor en (0, 0.5)

        score += bx_bonus(latest_daily_state.get("bx_value") if latest_daily_state else None) * weight_daily
        score += bx_bonus(latest_weekly_state.get("bx_value") if latest_weekly_state else None) * weight_weekly
        score += bx_bonus(latest_monthly_state.get("bx_value") if latest_monthly_state else None) * weight_monthly
        reasons.append("bx_bonus applied")

        # --- Penalizaciones ---
        # Contra-macro: monthly rojo pero daily verde -> penaliza
        if (
            consecutive_monthly_state
            and consecutive_monthly_state.startswith("red")
            and consecutive_daily_state
            and consecutive_daily_state.startswith("green")
        ):
            score -= 1.0
            reasons.append("counter_monthly_penalty")

        # Pullback diario recuperándose (red_hl): penalización moderada
        if consecutive_daily_state == "red_hl" or (latest_daily_state and latest_daily_state.get("bx_state") == "red_hl"):
            score -= 0.7
            reasons.append("daily_red_hl_penalty")

        # Penalización por falta de confirmaciones sostenidas (más peso a monthly/weekly)
        if not consecutive_daily_ok:
            score -= 0.4
            reasons.append("no_sustained_daily")
        if not consecutive_weekly_ok:
            score -= 0.6
            reasons.append("no_sustained_weekly")
        if not consecutive_monthly_ok:
            score -= 0.8
            reasons.append("no_sustained_monthly")

        # Penalización por volumen bajo (usar datos si están en latest_daily_state o fallback a _volume_metrics)
        try:
            vol = None
            median20 = None
            if latest_daily_state:
                vol = latest_daily_state.get("volume")
                median20 = latest_daily_state.get("median20")
            # fallback: si no vienen, intentar obtener desde DB si disponemos del ticker en latest_daily_state
            if (vol is None or median20 is None) and latest_daily_state and latest_daily_state.get("ticker"):
                try:
                    last_vol, med20 = self._volume_metrics(latest_daily_state.get("ticker"), "1d")
                    if last_vol is not None and med20 is not None:
                        vol = last_vol
                        median20 = med20
                except Exception:
                    vol = None
                    median20 = None
            if vol is not None and median20 is not None:
                try:
                    vol_min_factor = float(self.config.volume_min_factor)
                except Exception:
                    vol_min_factor = 1.0
                if vol < (median20 * vol_min_factor):
                    score -= 0.5
                    reasons.append("low_volume_penalty")
        except Exception:
            self.logger.debug("Warning: volumen no disponible para penalización.")

        # Penalización por bx_value demasiado bajo (daily)
        try:
            if latest_daily_state and latest_daily_state.get("bx_value") is not None:
                try:
                    bx_min_abs = float(self.config.bx_value_min_abs)
                except Exception:
                    bx_min_abs = 0.0
                if abs(latest_daily_state.get("bx_value")) < bx_min_abs:
                    score -= 0.8
                    reasons.append("low_bx_value_penalty")
        except Exception:
            self.logger.debug("Warning: bx_value no disponible para penalización.")

        return score, reasons


    def _check_candidate_entry(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Evaluación de candidate: devuelve siempre un dict de evaluación (ENTRY_HIGH / ENTRY_MED / NO_ENTRY)
        salvo que no haya datos (en cuyo caso retorna None).
        """
        # confirmaciones sostenidas (bool, state)
        consecutive_daily_ok, consecutive_daily_state = self._sustained_state(ticker, "1d", self.config.confirm_daily)
        consecutive_weekly_ok, consecutive_weekly_state = self._sustained_state(ticker, "1wk", self.config.confirm_weekly)
        consecutive_monthly_ok, consecutive_monthly_state = self._sustained_state(ticker, "1mo", self.config.confirm_monthly)

        # obtener últimos valores (incluye bx_value)
        latest_daily_state = self._latest_state(ticker, "1d")
        latest_weekly_state = self._latest_state(ticker, "1wk")
        latest_monthly_state = self._latest_state(ticker, "1mo")

        # si no hay datos mínimos para evaluar, devolvemos None (no podemos evaluar)
        if latest_daily_state is None and latest_weekly_state is None and latest_monthly_state is None:
            return None

        # enriquecer latest_daily_state con volumen/median20 si es posible (para que pattern_score pueda usarlo)
        try:
            last_vol, median20 = self._volume_metrics(ticker, "1d")
            if latest_daily_state is None:
                latest_daily_state = {"date": None, "bx_state": None, "bx_value": None}
            # añadir campos extras si existen
            if last_vol is not None:
                latest_daily_state["volume"] = last_vol
            if median20 is not None:
                latest_daily_state["median20"] = median20
            # añadir ticker por si pattern_score quiere intentar una segunda búsqueda (opcional)
            latest_daily_state["ticker"] = ticker
        except Exception:
            # no queremos fallar por volumen
            pass

        # calcular score objetivo
        score, reasons = self._pattern_score(
            consecutive_daily_state,
            consecutive_weekly_state,
            consecutive_monthly_state,
            latest_daily_state,
            latest_weekly_state,
            latest_monthly_state,
            consecutive_daily_ok,
            consecutive_weekly_ok,
            consecutive_monthly_ok
        )

        # umbrales (ajustables): >=4 ENTRY_HIGH, >=1.5 ENTRY_MED, < 1.5 NO_ENTRY 
        if score >= 4.0:
            action = "ENTRY_HIGH"
        elif score >= 1.5:
            action = "ENTRY_MED"
        else:
            action = "NO_ENTRY"

        return {
            "ticker": ticker,
            "action": action,
            "score": float(score),
            "reason": " | ".join(reasons) if reasons else "NO_REASONS_DATA",
            "states": {"daily": latest_daily_state, "weekly": latest_weekly_state, "monthly": latest_monthly_state},
        }

    def _check_portfolio_exit(self, ticker: str) -> Optional[Dict[str, Any]]:
        consecutive_daily_ok, consecutive_daily_state = self._sustained_state(ticker, "1d", self.config.confirm_daily)
        consecutive_weekly_ok, consecutive_weekly_state = self._sustained_state(
            ticker, "1wk", self.config.confirm_weekly
        )
        if (
            consecutive_daily_ok
            and consecutive_weekly_ok
            and consecutive_daily_state.startswith("red")
            and consecutive_weekly_state.startswith("red")
        ):
            latest_daily_state = self._latest_state(ticker, "1d")
            latest_weekly_state = self._latest_state(ticker, "1wk")
            return {
                "ticker": ticker,
                "action": "EXIT",
                "reason": "Daily & Weekly red confirmed",
                "states": {"daily": latest_daily_state, "weekly": latest_weekly_state},
            }
        return None

    def _format_candidate_msg(self, info: Dict[str, Any]) -> str:
        ticker = info["ticker"]
        reason = info["reason"]
        daily_state  = info["states"]["daily"]
        weekly_state = info["states"]["weekly"]
        monthly_state = info["states"]["monthly"]
        msg = (
            f"CANDIDATO — {ticker} — {info['action']}\n"
            f"Ticker: {ticker}\n"
            f"Acción: {info['action']}\n"
            f"Prioridad: {'Alta' if info['action'] == 'ENTRY_HIGH' else 'Media'}\n"
            f"Patrón detectado: Monthly={monthly_state['bx_state']} ({monthly_state['date'].date()}), Weekly={weekly_state['bx_state']} ({weekly_state['date'].date()}), Daily={daily_state['bx_state']} ({daily_state['date'].date()})\n"
            f"BX values: monthly={monthly_state.get('bx_value'):.3f if monthly_state.get('bx_value') is not None else None}, weekly={weekly_state.get('bx_value'):.3f if weekly_state.get('bx_value') is not None else None}, daily={daily_state.get('bx_value'):.3f if daily_state.get('bx_value') is not None else None}\n"
            f"Razón breve: {reason}\n"
            f"Sugerencia rápida: Revisar niveles y calcular sizing basados en riesgo.\n"
            f"ID alerta: candidate_entry_{ticker}_{daily_state['date'].date()}"
        )
        return msg

    def _format_portfolio_msg(self, info: Dict[str, Any]) -> str:
        ticker = info["ticker"]
        reason = info["reason"]
        daily_state = info["states"]["daily"]
        weekly_state = info["states"]["weekly"]
        msg = (
            f"PORTAFOLIO — {ticker} — Crítica — {info['action']}\n"
            f"Ticker: {ticker}\n"
            f"Acción: {info['action']}\n"
            f"Prioridad: Crítica\n"
            f"Estado detectado: Weekly={weekly_state['bx_state']} ({weekly_state['date'].date()}), Daily={daily_state['bx_state']} ({daily_state['date'].date()})\n"
            f"BX values: weekly={weekly_state.get('bx_value'):.3f if weekly_state.get('bx_value') is not None else None}, daily={daily_state.get('bx_value'):.3f if daily_state.get('bx_value') is not None else None}\n"
            f"Razón breve: {reason}\n"
            f"Sugerencia rápida: considerar reducción parcial o salida completa según tu gestión de riesgo.\n"
            f"ID alerta: portfolio_exit_{ticker}_{daily_state['date'].date()}"
        )
        return msg

    def _determine_alert_type(self, info: Dict[str, Any]) -> str:
        """
        Determina el tipo de alerta (critical/high/medium/noentry/info) basado
        en el resultado multi-timeframe `info` que generan _check_candidate_entry
        o _check_portfolio_exit.

        - info expected keys:
            - action: 'ENTRY_HIGH' | 'ENTRY_MED' | 'EXIT' | ...
            - score: float (puede ser None)
            - states: {'daily': {...}, 'weekly': {...}, 'monthly': {...}} (opcional)

        Lógica:
        - EXIT (salida de portafolio) => prioridad CRÍTICA (siempre importante).
        - ENTRY_HIGH => 'high' o 'critical' si score muy alto o si contra-macro es fuerte.
        - ENTRY_MED => 'medium' (o 'noentry' si score muy bajo).
        - Si no hay score, usar heurístico por action.
        """
        try:
            action = (info.get("action") or "").upper()
            score = info.get("score")

            # Defaults
            if action == "EXIT":
                return "critical"

            # si es ENTRY_HIGH, juzgar por score
            if action == "ENTRY_HIGH":
                # score puede ser None — en ese caso asumimos 'high'
                if score is None:
                    return "high"
                # umbrales ajustables: >=6.0 -> critical, >=4.0 -> high
                if score >= 6.0:
                    return "critical"
                if score >= 4.0:
                    return "high"
                # score relativamente bajo aun cuando action sea ENTRY_HIGH -> high de todos modos
                return "high"

            # ENTRY_MED -> medium o noentry si score muy bajo
            if action == "ENTRY_MED":
                if score is None:
                    return "medium"
                if score >= 2.5:
                    return "medium"
                # muy bajo -> no entrar
                return "noentry"

            # Otros actions / WATCH / INFO
            if action in ("WATCH", "NO_ACTION", "NOENTRY"):
                return "info"

            # Fallback según score si existe
            if score is not None:
                if score >= 6.0:
                    return "critical"
                if score >= 4.0:
                    return "high"
                if score >= 1.5:
                    return "medium"
                return "noentry"

            return "info"
        except Exception as e:
            # En caso de error, devolver info para no bloquear flujo
            self.logger.debug("Error determinando alert_type a partir de info: %s", e)
            return "info"

    async def process_and_notify(self):

        # Calcular y guardar cambios por timeframe
        tickers_all = list(
            set((self.config.candidates or []) + (self.config.portfolio or []))
        )
        for t in tickers_all:
            for tf in self.config.timeframes:
                try:
                    df = self.db_manager.get_data(t, tf)
                    if df is None or df.empty:
                        self.logger.warning("Sin datos en DB para %s %s", t, tf)
                        continue
                    df_calc = self.indicator_calc.compute_bx_trender(df)
                    self.metrics.record_calculation(t, tf)
                except Exception as e:
                    self.logger.error("Error en cálculo para %s %s: %s", t, tf, e)
                    self.metrics.record_error("calculation")
                    continue
                try:
                    last_idx = df_calc.index.max()
                    last_date = last_idx.date().isoformat()
                    bx_state = df_calc.loc[last_idx, "bx_state"]
                    bx_val = (
                        df_calc.loc[last_idx, "bx_value"]
                        if not pd.isna(df_calc.loc[last_idx, "bx_value"])
                        else None
                    )
                    if bx_state is None:
                        self.logger.info(
                            "%s %s %s -> estado indeterminado", t, tf, last_date
                        )
                        continue
                    prev = self.db_manager.get_last_color(t, tf)
                    if prev != bx_state:
                         # Actualizamos sólo el estado (histórico). No enviamos alertas single-timeframe.
                        try:
                            self.db_manager.set_last_color(t, tf, bx_state)
                            self.metrics.record_alert(t, tf)  # se registra el cambio
                            # Añadir al batch (opcional) para resumen diario — usa formato simple
                            try:
                                self.notification_manager.batch_manager.add_alert(
                                    t, tf, last_date, bx_state, bx_val
                                )
                            except Exception:
                                pass
                            self.logger.info("STATE UPDATED: %s %s %s -> %s", t, tf, last_date, bx_state)
                        except Exception as exc:
                            self.logger.error("Error al actualizar estado para %s %s: %s", t, tf, exc)
                    else:
                        self.logger.debug(
                            "%s %s %s -> sin cambio (%s)", t, tf, last_date, bx_state
                        )
                except Exception as e:
                    self.logger.error(
                        "Error al procesar notificación para %s %s: %s", t, tf, e
                    )
                    self.metrics.record_error("notification")
                    continue
        # Ahora evaluar flujos separados: CANDIDATES (posibles entradas) y PORTFOLIO (posibles salidas)
        # CANDIDATES
        for c in self.config.candidates or []:
            try:
                info = self._check_candidate_entry(c)
                if info:
                    # construir alert_id usando la fecha del daily trigger
                    triggered_date_iso = str(info["states"]["daily"]["date"])
                    alert_id = f"candidate_entry_{c}_{pd.to_datetime(triggered_date_iso).date().isoformat()}"
                    
                    # duplicado / cooldown check usando la tabla state (get_last_color / set_last_color)
                    last_state_entry = self.db_manager.get_last_color(c, "candidate_entry")
                    if last_state_entry == alert_id:
                        self.logger.debug("Ya se envió alerta candidate para %s reciente: %s", c, alert_id)
                        continue
                
                    # insertar en alerts_v2 (si ya existía, save_alert_v2 retornará False)
                    inserted = self.db_manager.save_alert_v2(
                        alert_id=alert_id,
                        ticker=c,
                        action=info["action"],
                        alert_type="candidate_entry",
                        states=info["states"],
                        score=info.get("score"),
                        priority="high" if info.get("action") == "ENTRY_HIGH" else "medium",
                        note=info.get("reason"),
                        triggered_at=triggered_date_iso,
                    )
                    if not inserted:
                        self.logger.debug("Alerta candidate ya existente en alerts_v2: %s — no reenvío", alert_id)
                    # marcar en state table para compatibilidad y evitar re-alertas inmediatas
                    try:
                        self.db_manager.set_last_color(c, "candidate_entry", alert_id)
                    except Exception:
                        pass

                    # añadir al batch summary (multi-timeframe)
                    try:
                        self.notification_manager.batch_manager.add_alert(
                            ticker=c,
                            timeframe="multi",
                            date=triggered_date_iso,
                            state=info["action"],
                            value=info.get("score"),
                        )
                    except Exception:
                        pass
                     # enviar mensaje formateado
                    msg = self._format_candidate_msg(info)
                    sent = await self.notification_manager.send_message(msg)
                    if sent:
                        self.metrics.record_alert(c, "candidate")
                    else:
                        self.logger.warning("No se pudo enviar alerta candidate %s", alert_id)
            except Exception as e:
                self.logger.error("Error al evaluar candidato %s: %s", c, e)
                self.metrics.record_error("candidate_check")
        # PORTFOLIO
        for p in self.config.portfolio or []:
            try:
                info = self._check_portfolio_exit(p)
                if info:
                    triggered_date_iso = str(info["states"]["daily"]["date"])
                    alert_id = f"portfolio_exit_{p}_{pd.to_datetime(triggered_date_iso).date().isoformat()}"

                    # duplicado / cooldown check usando la tabla state
                    last_state_exit = self.db_manager.get_last_color(p, "portfolio_exit")
                    if last_state_exit == alert_id:
                        self.logger.debug("Ya se envió alerta portfolio para %s reciente: %s", p, alert_id)
                        continue

                    inserted = self.db_manager.save_alert_v2(
                        alert_id=alert_id,
                        ticker=p,
                        action=info["action"],
                        alert_type="portfolio_exit",
                        states=info["states"],
                        score=None,
                        priority="critical",
                        note=info.get("reason"),
                        triggered_at=triggered_date_iso,
                    )

                    if not inserted:
                        self.logger.debug("Alerta portfolio ya existente en alerts_v2: %s — no reenvío", alert_id)
                        try:
                            self.db_manager.set_last_color(p, "portfolio_exit", alert_id)
                        except Exception:
                            pass
                        continue

                    try:
                        self.db_manager.set_last_color(p, "portfolio_exit", alert_id)
                    except Exception:
                        pass

                    # añadir al batch summary (multi-timeframe)
                    try:
                        self.notification_manager.batch_manager.add_alert(
                            ticker=p,
                            timeframe="multi",
                            date=triggered_date_iso,
                            state=info["action"],
                            value=None,
                        )
                    except Exception:
                        pass

                    msg = self._format_portfolio_msg(info)
                    sent = await self.notification_manager.send_message(msg)
                    if sent:
                        self.metrics.record_alert(p, "portfolio")
                    else:
                        self.logger.warning("No se pudo enviar alerta portfolio %s", alert_id) 
                    
            except Exception as e:
                self.logger.error("Error al evaluar portfolio %s: %s", p, e)
                self.metrics.record_error("portfolio_check")
        # Procesar batching de notificaciones
        await self._process_batch_notifications()
        
        # Verificar si se debe enviar resumen diario
        await self._check_daily_summary()

    async def _process_batch_notifications(self):
        """Procesa las notificaciones en batch"""
        if not self.config.batching_enabled:
            return
        
        batch_manager = self.notification_manager.batch_manager
        
        # Enviar batch si es necesario
        if batch_manager.should_send_batch():
            batch_msg = batch_manager.get_batch_message()
            if batch_msg:
                sent = await self.notification_manager.send_message(batch_msg)
                if sent:
                    self.logger.info(
                        f"Batch enviado con {len(batch_manager.pending_alerts)} alertas"
                    )
                    batch_manager.clear_batch()
                else:
                    self.logger.warning("No se pudo enviar el batch de notificaciones")

    async def _check_daily_summary(self):
        """Verifica y envía el resumen diario si es necesario"""
        if not self.config.batching_enabled or not self.config.summary_enabled:
            return
        batch_manager = self.notification_manager.batch_manager
        if batch_manager.should_send_daily_summary():
            summary_msg = batch_manager.get_daily_summary()
            if summary_msg:
                sent = await self.notification_manager.send_message(summary_msg)
                if sent:
                    self.logger.info("Resumen diario enviado")
                    batch_manager.mark_summary_sent()
                else:
                    self.logger.warning("No se pudo enviar el resumen diario")

    async def run(self):
        """Ejecuta el bot completo con batching continuo"""
        try:
            # Inicializar base de datos
            await asyncio.to_thread(self.db_manager.init_db)

            # Inicializar bot de Telegram
            await self.notification_manager.initialize_bot()

            # Ejecutar backfill y actualizaciones
            await self.run_backfill_and_updates()

            # Procesar y notificar inicial
            await self.process_and_notify()
            if self.config.batching_enabled:
                self.logger.info("Iniciando bucle de batching continuo...")
                while True:
                    try:
                        # Esperar antes del siguiente ciclo
                        await asyncio.sleep(self.config.batch_timeout_seconds)
                        
                        # Procesar batch pendiente
                        await self._process_batch_notifications()
                        
                        # Verificar resumen diario
                        await self._check_daily_summary()
                        
                        # Health check periódico
                        if self.config.health_check_interval > 0:
                            await self.health_check()
                    except KeyboardInterrupt:
                        self.logger.info(
                            "Bucle de batching interrumpido por el usuario"
                        )
                        break
                    except Exception as e:
                        self.logger.error("Error en bucle de batching: %s", e)
                        self.metrics.record_error("batching_loop")
                        await asyncio.sleep(60)
            if self.config.metrics_enabled:
                stats = self.metrics.get_stats()
                self.logger.info("Estadísticas finales: %s", stats)
            self.logger.info("Proceso completado exitosamente.")
        except Exception as e:
            self.logger.error("Error crítico en run: %s", e)
            self.metrics.record_error("critical")
            raise


def main():
    try:
        bot = BXTrenderBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"Error fatal en la aplicación: {e}")
        raise


if __name__ == "__main__":
    main()
