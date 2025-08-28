#!/usr/bin/env python3
"""
BX Trender Bot - Implementación orientada a objetos
Clase principal para el bot de trading con indicador BX Trender
"""

import os
import time
import sqlite3
import json
import datetime as dt
import logging
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
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
    backfill_years: Dict[str, int]  # Configuración específica por timeframe
    update_period: str
    request_delay: float
    tickers: List[str]
    timeframes: List[str]
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
    critical_states: List[str]
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
        """Registra una actualización de datos"""
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
            "last_update": self.last_update.isoformat() if self.last_update else None
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
                
                # Tabla de alerts
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        date TEXT NOT NULL,
                        color TEXT NOT NULL,
                        bx_value REAL,
                        created_at TEXT NOT NULL
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
                cur.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker_timeframe_date ON prices(ticker, timeframe, date)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_prices_timeframe_date ON prices(timeframe, date)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ticker_timeframe_date ON alerts(ticker, timeframe, date)")
                
                conn.commit()
                self.logger.info("Base de datos inicializada correctamente")
        except Exception as e:
            self.logger.error(f"Error al inicializar la base de datos: {e}")
            raise
    
    def get_data(self, ticker: str, timeframe: str = "1d") -> Optional[pd.DataFrame]:
        """Obtiene datos de un ticker y timeframe específico con validación mejorada"""
        try:
            with self.get_conn() as conn:
                df = pd.read_sql("""
                    SELECT date, open, high, low, close, adj_close, volume
                    FROM prices 
                    WHERE ticker = ? AND timeframe = ?
                    ORDER BY date
                """, conn, params=[ticker, timeframe], 
                    index_col="date", parse_dates=["date"])
                
                # Validación de datos
                if df is None or df.empty:
                    self.logger.warning(f"No se encontraron datos para {ticker} {timeframe}")
                    return None
                
                # Validar que tenemos las columnas necesarias
                required_columns = ['open', 'high', 'low', 'close']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    self.logger.error(f"Columnas faltantes para {ticker} {timeframe}: {missing_columns}")
                    return None
                
                # Asegurar que las columnas estén en el formato esperado
                if 'adj_close' in df.columns:
                    df = df.rename(columns={'adj_close': 'Adj Close'})
                if 'close' in df.columns:
                    df = df.rename(columns={'close': 'Close'})
                
                self.logger.debug(f"Datos cargados para {ticker} {timeframe}: {len(df)} filas")
                return df
                
        except Exception as e:
            self.logger.error(f"Error al leer datos para {ticker} {timeframe}: {e}")
            return None
    
    def save_data(self, df: pd.DataFrame, ticker: str, timeframe: str):
        """Guarda DataFrame en la tabla unificada con validación mejorada"""
        if df is None or df.empty:
            self.logger.warning(f"DataFrame vacío para {ticker} {timeframe}, no se guarda")
            return
        
        try:
            df_to_save = df.copy()
            
            # Normalizar nombres de columnas (yfinance usa "Adj Close", tabla usa "adj_close")
            if 'Adj Close' in df_to_save.columns:
                df_to_save = df_to_save.rename(columns={'Adj Close': 'adj_close'})
            
            df_to_save['ticker'] = ticker
            df_to_save['timeframe'] = timeframe
            df_to_save['created_at'] = dt.datetime.utcnow().isoformat()
            df_to_save.reset_index(inplace=True)
            
            with self.get_conn() as conn:
                # Usar replace para evitar duplicados
                df_to_save.to_sql('prices', conn, if_exists='append', index=False, method='multi')
                self.logger.debug(f"Guardados {len(df_to_save)} registros para {ticker} {timeframe}")
                
        except Exception as e:
            self.logger.error(f"Error al guardar datos para {ticker} {timeframe}: {e}")
            raise
    
    def get_last_color(self, ticker: str, timeframe: str) -> Optional[str]:
        """Obtiene el último color conocido desde la base de datos"""
        try:
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT last_color FROM state WHERE ticker = ? AND timeframe = ?", (ticker, timeframe))
                r = cur.fetchone()
                return r[0] if r else None
        except Exception as e:
            self.logger.error(f"Error al obtener último color para {ticker} {timeframe}: {e}")
            return None
    
    def set_last_color(self, ticker: str, timeframe: str, color: str):
        """Actualiza el último color conocido en la base de datos"""
        try:
            now = dt.datetime.utcnow().isoformat()
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("INSERT OR REPLACE INTO state (ticker, timeframe, last_color, updated_at) VALUES (?, ?, ?, ?)", 
                           (ticker, timeframe, color, now))
                conn.commit()
                self.logger.debug(f"Estado actualizado para {ticker} {timeframe}: {color}")
        except Exception as e:
            self.logger.error(f"Error al actualizar estado para {ticker} {timeframe}: {e}")
            raise
    
    def save_alert(self, ticker: str, timeframe: str, date: str, color: str, bx_value: Optional[float]):
        """Guarda una alerta en la base de datos"""
        try:
            now = dt.datetime.utcnow().isoformat()
            with self.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("INSERT INTO alerts (ticker, timeframe, date, color, bx_value, created_at) VALUES (?, ?, ?, ?, ?, ?)", 
                           (ticker, timeframe, date, color, bx_value, now))
                conn.commit()
                self.logger.debug(f"Alerta guardada para {ticker} {timeframe} {date}: {color}")
        except Exception as e:
            self.logger.error(f"Error al guardar alerta para {ticker} {timeframe}: {e}")
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
            ma_up = up.ewm(alpha=1/length, adjust=False).mean()
            ma_down = down.ewm(alpha=1/length, adjust=False).mean()
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
            
            # Validación de datos más robusta
            min_required = max(self.config.short_l2, self.config.short_l3, self.config.t3_length, 
                              self.config.long_l1, self.config.long_l2) + 20
            if close.dropna().shape[0] < min_required:
                raise RuntimeError(f"Datos insuficientes para calcular indicadores. Se requieren al menos {min_required} puntos de datos, pero solo hay {close.dropna().shape[0]}")

            self.logger.debug(f"Calculando BX Trender para {len(close)} puntos de datos")

            # --- short term (igual que Pine) ---
            ema1 = self._ema(close, self.config.short_l1)
            ema2 = self._ema(close, self.config.short_l2)
            diff = ema1 - ema2
            rsi_diff = self._rsi(diff, self.config.short_l3)
            short_term = rsi_diff - 50  # shortTermXtrender

            # ma (T3) sobre shortTerm (como en Pine)
            ma_short = self._t3(short_term.ffill(), self.config.t3_length, self.config.t3_v)

            # --- long term (igual que Pine) ---
            ema_long = self._ema(close, self.config.long_l1)
            long_term = self._rsi(ema_long, self.config.long_l2) - 50  # longTermXtrender

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
            dfc["short_state"] = [short_state_from(c, p) for c, p in zip(dfc["short_xtrender"].tolist(), prev_short.tolist())]

            # Para long: mismo criterio aplicado a long_xtrender
            prev_long = dfc["long_xtrender"].shift(1)
            def long_state_from(curr, prev_val):
                if pd.isna(curr) or pd.isna(prev_val):
                    return None
                if curr > 0:
                    return "green_hh" if curr > prev_val else "green_lh"
                else:
                    return "red_hl" if curr > prev_val else "red_ll"
            dfc["long_state"] = [long_state_from(c, p) for c, p in zip(dfc["long_xtrender"].tolist(), prev_long.tolist())]

            # Por compatibilidad/retro: dejamos bx_state igual al short_state (alertas actuales usan short)
            dfc["bx_state"] = dfc["short_state"]

            self.logger.debug("Cálculo de BX Trender completado exitosamente")
            return dfc
            
        except Exception as e:
            self.logger.error(f"Error en cálculo de BX Trender: {e}")
            raise

class DataManager:
    """Manejador de datos de mercado"""
    
    def __init__(self, config: BotConfig, db_manager: DatabaseManager, logger: logging.Logger):
        self.config = config
        self.db_manager = db_manager
        self.logger = logger
    
    def backfill_ticker(self, ticker: str, timeframe: str, years: int = None):
        """Backfill inicial para un ticker y timeframe específico con manejo de errores mejorado"""
        if years is None:
            # Usar configuración específica por timeframe
            years = self.config.backfill_years.get(timeframe, 4)  # Default a 4 años si no está configurado
            
        period = f"{years}y"
        self.logger.info(f"Iniciando backfill para {ticker} {timeframe} period={period} ({years} años)")
        
        try:
            df = yf.download(ticker, period=period, interval=timeframe, 
                            auto_adjust=False, multi_level_index=False, progress=False)
            
            if df is None or df.empty:
                self.logger.warning(f"Backfill: no se obtuvieron datos para {ticker} {timeframe}")
                return
            
            # Validar datos mínimos
            if len(df) < 10:
                self.logger.warning(f"Backfill: datos insuficientes para {ticker} {timeframe} ({len(df)} filas)")
                return
            
            df.index = pd.to_datetime(df.index)
            self.db_manager.save_data(df, ticker, timeframe)
            self.logger.info(f"Backfill completado para {ticker} {timeframe}: {len(df)} filas guardadas")
            
        except Exception as e:
            self.logger.error(f"Error en backfill para {ticker} {timeframe}: {e}")
            raise
    
    def incremental_update_ticker(self, ticker: str, timeframe: str, period: str = None):
        """Actualización incremental para un ticker y timeframe específico con validación mejorada"""
        if period is None:
            period = self.config.update_period
            
        self.logger.info(f"Iniciando actualización incremental para {ticker} {timeframe} period={period}")
        
        try:
            existing = self.db_manager.get_data(ticker, timeframe)
            df_new = yf.download(ticker, period=period, interval=timeframe, 
                               auto_adjust=False, multi_level_index=False, progress=False)
            
            if df_new is None or df_new.empty:
                self.logger.warning(f"No se obtuvieron datos incrementales para {ticker} {timeframe}")
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
            
            # Guardar solo los datos nuevos, no todo el merged
            self.db_manager.save_data(df_new, ticker, timeframe)
            self.logger.info(f"Actualización incremental completada para {ticker} {timeframe}: {len(df_new)} nuevas filas, última fecha: {df_new.index.max().date()}")
            
        except Exception as e:
            self.logger.error(f"Error en actualización incremental para {ticker} {timeframe}: {e}")
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
    
    def add_alert(self, ticker: str, timeframe: str, date: str, state: str, value: Optional[float]):
        """Añade una alerta al batch"""
        alert = {
            'ticker': ticker,
            'timeframe': timeframe,
            'date': date,
            'state': state,
            'value': value,
            'timestamp': dt.datetime.now()
        }
        
        self.pending_alerts.append(alert)
        self.daily_alerts.append(alert)
        
        # Verificar si es un estado crítico que requiere notificación inmediata
        if state in self.config.critical_states:
            self.logger.info(f"Estado crítico detectado: {ticker} {timeframe} -> {state}")
            return True  # Indicar que debe enviarse inmediatamente
        
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
            "green_hh": "🟢⬆️ LIGHT GREEN",
            "green_lh": "🟢⬇️ GREEN", 
            "red_hl": "🟠⬆️ LIGHT RED",
            "red_ll": "🔴⬇️ RED"
        }
        
        # Agrupar primero por timeframe, luego por estado
        alerts_by_timeframe = {}
        for alert in self.pending_alerts:
            timeframe = alert['timeframe']
            state = alert['state']
            
            if timeframe not in alerts_by_timeframe:
                alerts_by_timeframe[timeframe] = {}
            
            if state not in alerts_by_timeframe[timeframe]:
                alerts_by_timeframe[timeframe][state] = []
            
            alerts_by_timeframe[timeframe][state].append(alert)
        
        # Construir mensaje
        lines = ["📊 *BATCH ALERTS - BX Trender*"]
        
        # Ordenar timeframes (1d, 1wk, 1mo)
        timeframe_order = ["1d", "1wk", "1mo"]
        sorted_timeframes = sorted(alerts_by_timeframe.keys(), 
                                 key=lambda x: timeframe_order.index(x) if x in timeframe_order else 999)
        
        for timeframe in sorted_timeframes:
            lines.append(f"\n*{timeframe.upper()}*")
            
            # Ordenar estados (green_hh, green_lh, red_hl, red_ll)
            state_order = ["green_hh", "green_lh", "red_hl", "red_ll"]
            sorted_states = sorted(alerts_by_timeframe[timeframe].keys(),
                                 key=lambda x: state_order.index(x) if x in state_order else 999)
            
            for state in sorted_states:
                alerts = alerts_by_timeframe[timeframe][state]
                human_state = human_map.get(state, state)
                lines.append(f"  {human_state} ({len(alerts)}):")
                
                for alert in alerts:
                    value_str = f" ({alert['value']:.4f})" if alert['value'] is not None else ""
                    lines.append(f"    • {alert['ticker']} - {alert['date']}{value_str}")
        
        lines.append(f"\n_Generated at {dt.datetime.now().strftime('%H:%M:%S')}_")
        
        return "\n".join(lines)
    
    def get_daily_summary(self) -> str:
        """Genera el resumen diario de alertas agrupado por timeframe"""
        if not self.daily_alerts:
            return ""
        
        # Filtrar alertas del día actual
        today = dt.datetime.now().date()
        today_alerts = [a for a in self.daily_alerts if a['timestamp'].date() == today]
        
        if not today_alerts:
            return ""
        
        human_map = {
            "green_hh": "🟢💪 LIGHT GREEN",
            "green_lh": "🟢 GREEN",
            "red_hl": "🟠💪 LIGHT RED", 
            "red_ll": "🔴 RED"
        }
        
        # Agrupar por timeframe
        alerts_by_timeframe = {}
        for alert in today_alerts:
            timeframe = alert['timeframe']
            state = alert['state']
            
            if timeframe not in alerts_by_timeframe:
                alerts_by_timeframe[timeframe] = {}
            
            if state not in alerts_by_timeframe[timeframe]:
                alerts_by_timeframe[timeframe][state] = 0
            
            alerts_by_timeframe[timeframe][state] += 1
        
        lines = [f"📈 *DAILY SUMMARY - {today.strftime('%Y-%m-%d')}*"]
        lines.append(f"Total signals: {len(today_alerts)}")
        
        # Ordenar timeframes (1d, 1wk, 1mo)
        timeframe_order = ["1d", "1wk", "1mo"]
        sorted_timeframes = sorted(alerts_by_timeframe.keys(), 
                                 key=lambda x: timeframe_order.index(x) if x in timeframe_order else 999)
        
        for timeframe in sorted_timeframes:
            lines.append(f"\n*{timeframe.upper()}*")
            
            # Ordenar estados (green_hh, green_lh, red_hl, red_ll)
            state_order = ["green_hh", "green_lh", "red_hl", "red_ll"]
            sorted_states = sorted(alerts_by_timeframe[timeframe].keys(),
                                 key=lambda x: state_order.index(x) if x in state_order else 999)
            
            for state in sorted_states:
                count = alerts_by_timeframe[timeframe][state]
                human_state = human_map.get(state, state)
                lines.append(f"  {human_state}: {count}")
        
        # Top tickers más activos
        ticker_counts = {}
        for alert in today_alerts:
            ticker = alert['ticker']
            ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
        
        if ticker_counts:
            top_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:3]
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
            summary_hour, summary_minute = map(int, self.config.summary_time.split(':'))
            summary_time = dt.time(summary_hour, summary_minute)
            
            # Enviar si es la hora del resumen y no se ha enviado hoy
            if (current_time.hour == summary_time.hour and 
                current_time.minute == summary_time.minute and
                dt.datetime.now().date() != self.last_summary_date):
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
        """Envía mensaje por Telegram con retry mejorado"""
        if not self.bot:
            self.logger.warning("Bot no inicializado. Mensaje no enviado: %s", msg[:100] + "..." if len(msg) > 100 else msg)
            return False
        if not self.config.telegram_chat_id:
            self.logger.warning("TELEGRAM_CHAT_ID no configurado. Mensaje no enviado: %s", msg[:100] + "..." if len(msg) > 100 else msg)
            return False
        
        try:
            chat_id = int(self.config.telegram_chat_id)
        except Exception as e:
            self.logger.error("TELEGRAM_CHAT_ID inválido: %s - %s", self.config.telegram_chat_id, e)
            return False

        last_exc = None
        for attempt in range(1, self.config.telegram_max_retries + 1):
            try:
                res = await self.bot.send_message(chat_id=chat_id, text=msg)
                self.logger.info("✅ Mensaje enviado (intento %d). message_id=%s", attempt, getattr(res, 'message_id', None))
                return True
            except Exception as e:
                last_exc = e
                self.logger.warning("Intento %d falló al enviar Telegram: %s", attempt, e)
                if attempt < self.config.telegram_max_retries:
                    await asyncio.sleep(self.config.telegram_retry_delay_base ** attempt)  # Exponential backoff
        
        self.logger.error("No se pudo enviar mensaje por telegram tras %d intentos. Excepción: %s", self.config.telegram_max_retries, last_exc)
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
        """Carga la configuración desde archivo YAML"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Cargar variables de entorno para Telegram
            telegram_token = os.getenv("TELEGRAM_TOKEN", config_data.get("telegram", {}).get("token", "")).strip()
            telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", config_data.get("telegram", {}).get("chat_id", "")).strip()
            
            # Cargar configuración de batching con valores por defecto
            batching_config = config_data.get("telegram", {}).get("batching", {})
            
            return BotConfig(
                database_file=config_data["database"]["file"],
                backfill_years=config_data["database"]["backfill_years"],
                update_period=config_data["database"]["update_period"],
                request_delay=config_data["database"]["request_delay"],
                tickers=config_data["trading"]["tickers"],
                timeframes=config_data["trading"]["timeframes"],
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
                critical_states=batching_config.get("critical_states", ["green_hh", "red_ll"]),
                summary_enabled=batching_config.get("summary_enabled", True),
                summary_time=batching_config.get("summary_time", "18:00"),
                log_level=config_data["logging"]["level"],
                log_file_level=config_data["logging"]["file_level"],
                log_console_level=config_data["logging"]["console_level"],
                log_dir=config_data["logging"]["log_dir"],
                max_log_files=config_data["logging"]["max_log_files"],
                health_check_interval=config_data["monitoring"]["health_check_interval"],
                metrics_enabled=config_data["monitoring"]["metrics_enabled"],
                alert_on_errors=config_data["monitoring"]["alert_on_errors"]
            )
        except Exception as e:
            raise RuntimeError(f"Error al cargar configuración desde {config_file}: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Configura logging estructurado con archivo y consola"""
        # Crear directorio de logs si no existe
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        # Configurar formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
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
            for ticker in self.config.tickers[:1]:  # Solo verificar el primer ticker
                for timeframe in self.config.timeframes[:1]:
                    data = self.db_manager.get_data(ticker, timeframe)
                    if data is None or data.empty:
                        self.logger.warning("Health check: No hay datos recientes para %s %s", ticker, timeframe)
                        return False
            
            self.logger.info("Health check: Todos los componentes funcionando correctamente")
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
            # Check if we have data for first ticker and timeframe
            try:
                tbl = self.db_manager.get_data(self.config.tickers[0], self.config.timeframes[0])
                if tbl is None or tbl.empty:
                    needs_backfill = True
                    self.logger.info("Base de datos existe pero sin datos, iniciando backfill")
            except Exception as e:
                self.logger.warning("Error al verificar datos existentes, iniciando backfill: %s", e)
                needs_backfill = True

        if needs_backfill:
            self.logger.info("Iniciando backfill inicial para todos los timeframes...")
            for t in self.config.tickers:
                for tf in self.config.timeframes:
                    try:
                        await asyncio.to_thread(self.data_manager.backfill_ticker, t, tf)
                        await asyncio.sleep(self.config.request_delay)
                    except Exception as e:
                        self.logger.error("Error en backfill para %s %s: %s", t, tf, e)
                        self.metrics.record_error("backfill")
                        continue
        else:
            self.logger.info("Base de datos existente con datos. Saltando backfill.")

        self.logger.info("Iniciando actualización incremental...")
        for t in self.config.tickers:
            for tf in self.config.timeframes:
                try:
                    await asyncio.to_thread(self.data_manager.incremental_update_ticker, t, tf)
                    await asyncio.sleep(self.config.request_delay)
                except Exception as e:
                    self.logger.error("Error en actualización incremental para %s %s: %s", t, tf, e)
                    self.metrics.record_error("update")
                    continue
    
    async def process_and_notify(self):
        """Procesa y notifica para múltiples timeframes con sistema de batching"""
        human_map = {
            "green_hh": "🟢💪 LIGHT GREEN (Higher High)",
            "green_lh": "🟢 GREEN (Lower High)",
            "red_hl":  "🟠💪 LIGHT RED (Higher Low)",   
            "red_ll":  "🔴 RED (Lower Low)"
        }

        # Procesar todos los tickers y timeframes
        for t in self.config.tickers:
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
                    dbg_name = f"debug_{t}_{tf}.csv"
                    try:
                        df.to_csv(dbg_name)
                        self.logger.info("Datos guardados en %s para debugging", dbg_name)
                    except Exception as csv_error:
                        self.logger.error("Error al guardar datos de debug: %s", csv_error)
                    continue

                try:
                    last_idx = df_calc.index.max()
                    last_date = last_idx.date().isoformat()
                    bx_state = df_calc.loc[last_idx, "bx_state"]
                    bx_val = df_calc.loc[last_idx, "bx_value"] if not pd.isna(df_calc.loc[last_idx, "bx_value"]) else None

                    if bx_state is None:
                        self.logger.info("%s %s %s -> estado indeterminado", t, tf, last_date)
                        continue

                    prev = self.db_manager.get_last_color(t, tf)
                    if prev != bx_state:
                        # Guardar alerta en la base de datos
                        self.db_manager.save_alert(t, tf, last_date, bx_state, bx_val)
                        self.db_manager.set_last_color(t, tf, bx_state)
                        self.metrics.record_alert(t, tf)
                        
                        # Añadir al sistema de batching
                        is_critical = self.notification_manager.batch_manager.add_alert(
                            t, tf, last_date, bx_state, bx_val
                        )
                        
                        # Si es crítico, enviar inmediatamente
                        if is_critical and self.config.batching_enabled:
                            human = human_map.get(bx_state, bx_state)
                            critical_msg = f"🚨 *CRITICAL ALERT* 🚨\n{t} {tf} — BX Trender cambió a {human} el {last_date}\nBX_value={bx_val:.6f}"
                            await self.notification_manager.send_message(critical_msg)
                            self.logger.info("ALERTA CRÍTICA ENVIADA: %s %s %s -> %s", t, tf, last_date, bx_state)
                        
                        self.logger.info("ALERTA DETECTADA: %s %s %s -> %s", t, tf, last_date, bx_state)
                    else:
                        self.logger.debug("%s %s %s -> sin cambio (%s)", t, tf, last_date, bx_state)
                        
                except Exception as e:
                    self.logger.error("Error al procesar notificación para %s %s: %s", t, tf, e)
                    self.metrics.record_error("notification")
                    continue
        
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
                    self.logger.info(f"Batch enviado con {len(batch_manager.pending_alerts)} alertas")
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

            # Bucle continuo para batching si está habilitado
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
                        self.logger.info("Bucle de batching interrumpido por el usuario")
                        break
                    except Exception as e:
                        self.logger.error("Error en bucle de batching: %s", e)
                        self.metrics.record_error("batching_loop")
                        await asyncio.sleep(60)  # Esperar antes de reintentar

            # Health check final
            if self.config.metrics_enabled:
                stats = self.metrics.get_stats()
                self.logger.info("Estadísticas finales: %s", stats)

            self.logger.info("Proceso completado exitosamente.")
            
        except Exception as e:
            self.logger.error("Error crítico en run: %s", e)
            self.metrics.record_error("critical")
            raise

def main():
    """Función principal"""
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

