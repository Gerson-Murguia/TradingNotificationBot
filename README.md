# BX Trender Bot 🤖📈

Un bot de trading automatizado que utiliza el indicador BX Trender para detectar señales de compra y venta en múltiples timeframes. El bot monitorea activos en tiempo real y envía alertas a través de Telegram.

## 📋 Requisitos

- Python 3.8 o superior
- Cuenta de Telegram (para recibir alertas)
- Conexión a internet

## 🚀 Características Implementadas

### 1. Flujos

#### 📈 **Candidatos (Watchlist)**
- **Propósito**: Buscar condiciones de entrada
- **Tickers**: Configurados en `config.yaml` bajo `trading.candidates`
- **Análisis**: Patrones de entrada multi-timeframe con confirmaciones consecutivas
- **Alertas**: ENTRY_HIGH (entrada confirmada), ENTRY_MED (vigilancia)

#### 💼 **Portafolio (Posiciones Abiertas)**
- **Propósito**: Vigilar condiciones de salida/debilidad
- **Tickers**: Configurados en `config.yaml` bajo `trading.portfolio`
- **Análisis**: Patrones de debilidad y salida
- **Alertas**: EXIT (salida urgente cuando daily + weekly en rojo confirmado)

### 2. Confirmación Multi-Timeframe (Hysteresis)

#### 🔍 **Reglas de Confirmación**
- **Daily**: Requiere `confirm_daily` cierres consecutivos (configurable, default: 2)
- **Weekly**: Requiere `confirm_weekly` cierres consecutivos (configurable, default: 1)
- **Monthly**: Requiere `confirm_monthly` cierres consecutivos (configurable, default: 1)

#### 📊 **Filtros de Calidad**
- **BX Value mínimo**: `|bx_value| ≥ bx_value_min_abs` (configurable, default: 0.0)
- **Volumen mínimo**: `volumen_actual ≥ mediana(últimos_20) × volume_min_factor` (configurable, default: 0.5)

### 3. Sistema de Scoring Inteligente

#### 🎯 **Cálculo de Score**
- **Base**: Puntajes por timeframe (Daily: 1.0x, Weekly: 1.2x, Monthly: 1.5x)
- **Estados**: green_hh (3.0), green_lh (1.5), red_hl (-0.5), red_ll (-2.0)
- **Bonus**: Por magnitud de bx_value (saturado en 0.5 por timeframe)
- **Penalizaciones**: Contra-macro, pullback diario, falta de confirmaciones, volumen bajo

#### ⚡ **Umbrales de Acción**
- **ENTRY_HIGH**: Score ≥ 4.0
- **ENTRY_MED**: Score ≥ 1.5
- **NO_ENTRY**: Score < 1.5

### 4. Patrones de Estrategia Implementados

#### 📈 **Patrones de Entrada**
- **Entrada Trending**: Confirmación sostenida en múltiples timeframes
- **Alineación Macro**: Monthly + Weekly + Daily con pesos diferenciados
- **Pullback en Tendencia**: Considera recuperaciones y contra-movimientos

#### 📉 **Patrones de Salida**
- **Debilidad Confirmada**: Daily + Weekly en rojo con confirmaciones consecutivas
- **Salida Urgente**: Cuando ambos timeframes confirman debilidad

### 5. Sistema de Estado y Re-Alerting

#### 💾 **Persistencia de Estado**
- **Base de datos**: Tabla `state` para estado por ticker/timeframe
- **Historial**: Tabla `alerts_v2` para alertas multi-timeframe
- **Re-alerting**: Cooldown configurable por tipo de alerta

#### 🔄 **Hysteresis**
- Contadores de confirmación consecutiva por timeframe
- Evita señales falsas por ruido
- Permite ajuste fino de sensibilidad

## 📁 Estructura de Archivos

```
Trading/
├── bxtrender_bot.py          # Bot principal implementado
├── config.yaml               # Configuración actualizada
├── README_V2.md              # Este archivo
├── prices.db                 # Base de datos (tablas: prices, alerts_v2, state)
└── logs/                     # Logs del sistema
```

## ⚙️ Configuración

### Archivo `config.yaml`

```yaml
# Trading
trading:
  candidates: ["LULU", "DECK", "ARE", "OLN", "CDNA", "NVDA", "AAPL", "MSFT", "TSLA", "AMD"]
  portfolio: ["LULU", "DECK"]  # Tickers en posiciones actuales
  timeframes: ["1d", "1wk", "1mo"]

# Confirmación multi-timeframe
confirmations:
  daily: 2                     # confirm_daily: cierres consecutivos diarios
  weekly: 1                    # confirm_weekly: cierres consecutivos semanales
  monthly: 1                   # confirm_monthly: cierres consecutivos mensuales
  bx_value_min_abs: 0.0       # Umbral mínimo de bx_value
  volume_min_factor: 0.5       # Factor de volumen mínimo

# Alertas
alerts:
  types:
    high: "ALTA - Revisar para entrada"
    medium: "MEDIA - Vigilancia, considerar"
    critical: "CRÍTICA - Considerar salida/reducir"
    noentry: "NO ENTRAR - No cumple requisitos para entrada"
    info: "INFO - Cambios menores, sin acción"
  re_alert_cooldown_hours: 24
```

## 🚀 Uso

### Ejecutar el Bot

```bash
# Activar entorno virtual
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Ejecutar bot
python bxtrender_bot.py
```

## 📊 Ejemplo de Salida

### Alerta ENTRY_HIGH (Entrada)
```
CANDIDATO — NVDA — ENTRY_HIGH
Ticker: NVDA
Acción: ENTRY_HIGH
Prioridad: Alta
Patrón detectado: Monthly=green_hh (2025-01-30), Weekly=green_hh (2025-01-30), Daily=green_hh (2025-01-30)
BX values: monthly=0.023, weekly=0.018, daily=0.015
Razón breve: base_scores: D=3.00, W=3.60, M=4.50 | bx_bonus applied
Sugerencia rápida: Revisar niveles y calcular sizing basados en riesgo.
ID alerta: candidate_entry_NVDA_2025-01-30
```

### Alerta EXIT (Salida)
```
PORTAFOLIO — TSLA — Crítica — EXIT
Ticker: TSLA
Acción: EXIT
Prioridad: Crítica
Estado detectado: Weekly=red_ll (2025-01-30), Daily=red_ll (2025-01-30)
BX values: weekly=-0.025, daily=-0.032
Razón breve: Daily & Weekly red confirmed
Sugerencia rápida: considerar reducción parcial o salida completa según tu gestión de riesgo.
ID alerta: portfolio_exit_TSLA_2025-01-30
```

## 🔧 Personalización

### Ajustar Sensibilidad
- **Más conservador**: Aumentar `confirm_daily`, `confirm_weekly`, `confirm_monthly`
- **Más agresivo**: Reducir confirmaciones, bajar `bx_value_min_abs`

### Modificar Patrones
- Editar método `_pattern_score()` en la clase principal:
  - Ajustar pesos por timeframe
  - Modificar bonus/penalizaciones
  - Cambiar umbrales de score

### Priorización Personalizada
- Modificar `_determine_alert_type()` para cambiar lógica de prioridades
- Ajustar umbrales de score en `_check_candidate_entry()`

## 📈 Métricas y Monitoreo

### Base de Datos
- **prices**: Datos OHLCV por ticker y timeframe
- **alerts_v2**: Historial de alertas multi-timeframe
- **state**: Estado actual por ticker y timeframe

### Logs
- **Archivo**: `logs/trading_bot_YYYYMMDD.log`
- **Nivel**: Configurable por consola y archivo
- **Información**: Estados, alertas, errores, métricas

## 🔄 Características del Sistema

### ✅ **Implementado y Funcional**
- ✅ Flujos bien delimitados (Candidatos/Portafolio)
- ✅ Confirmación multi-timeframe con hysteresis
- ✅ Sistema de scoring inteligente con bonus/penalizaciones
- ✅ Alertas diferenciadas por tipo de acción
- ✅ Sistema de re-alerting con cooldown
- ✅ Patrones de estrategia específicos
- ✅ Filtros de calidad (BX value, volumen)
- ✅ Base de datos mejorada con tablas específicas
- ✅ Logs estructurados y métricas
- ✅ Batching de notificaciones
- ✅ Resumen diario automático

### 🎯 **Beneficios del Sistema**
- **Menos ruido**: Confirmación multi-TF reduce falsos positivos
- **Más acción**: Alertas específicas con sugerencias claras
- **Mejor organización**: Separación clara entre watchlist y portafolio
- **Scoring objetivo**: Sistema numérico para evaluar oportunidades
- **Persistencia**: Estado guardado para análisis histórico

## 🐛 Solución de Problemas

### Error de Conexión a Yahoo Finance
- Verificar conexión a internet
- Revisar si los tickers son válidos
- Aumentar `request_delay` en configuración

### Alertas de Telegram no llegan
- Verificar `TELEGRAM_TOKEN` y `TELEGRAM_CHAT_ID`
- Asegurar que el bot esté iniciado
- Revisar logs para errores de conexión

### Base de datos corrupta
- Eliminar `prices.db` para recrear
- El bot descargará datos históricos automáticamente

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Soporte

Para soporte técnico o preguntas:
- Revisar los logs en `logs/`
- Verificar la configuración en `config.yaml`
- Consultar la documentación del código


**⚠️ Descargo de responsabilidad**: Este software es solo para fines educativos. El trading conlleva riesgos financieros significativos. Los desarrolladores no se hacen responsables de pérdidas financieras.
---

**BX Trender Bot** - Sistema implementado para trading algorítmico con confirmación multi-timeframe, scoring inteligente y alertas accionables.

