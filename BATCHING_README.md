# Sistema de Batching de Notificaciones

## Descripción

El sistema de batching de notificaciones evita el spam y agrupa las alertas del BX Trender Bot para hacer las notificaciones más útiles y menos intrusivas.

## Características

### 🚀 Batching Inteligente
- **Agrupación por timeframe**: Las alertas se agrupan primero por timeframe (1D, 1WK, 1MO) y luego por color
- **Timeout configurable**: Los batches se envían después de un tiempo máximo configurable
- **Límite de alertas**: Máximo número de alertas por mensaje
- **Ordenamiento lógico**: Timeframes ordenados de menor a mayor, colores en orden de intensidad

### 🚨 Alertas Críticas
- **Notificación inmediata**: Los estados críticos (`green_hh`, `red_ll`) se envían inmediatamente
- **Formato destacado**: Las alertas críticas tienen un formato especial con emojis de advertencia

### 📊 Resumen Diario
- **Resumen automático**: Se envía un resumen diario con todas las señales del día
- **Hora configurable**: La hora del resumen se puede configurar (por defecto 18:00)
- **Estadísticas**: Incluye conteos por estado y tickers más activos

## Configuración

### En `config.yaml`:

```yaml
telegram:
  # ... configuración existente ...
  batching:
    enabled: true                    # Habilitar/deshabilitar batching
    max_alerts_per_batch: 5          # Máximo alertas por mensaje
    batch_timeout_seconds: 30        # Tiempo máximo para agrupar (segundos)
    critical_states: ["green_hh", "red_ll"]  # Estados que requieren notificación inmediata
    summary_enabled: true            # Habilitar resumen diario
    summary_time: "18:00"            # Hora del resumen (formato HH:MM)
```

## Comportamiento

### Flujo Normal
1. **Detección de cambio**: El bot detecta un cambio de estado en un ticker/timeframe
2. **Añadir al batch**: La alerta se añade al batch de notificaciones pendientes
3. **Verificar crítico**: Si es un estado crítico, se envía inmediatamente
4. **Envío de batch**: El batch se envía cuando:
   - Se alcanza el máximo de alertas (`max_alerts_per_batch`)
   - Pasa el timeout (`batch_timeout_seconds`)

### Ejemplo de Mensaje de Batch

```
📊 BATCH ALERTS - BX Trender

*1D*
  🟢💪 LIGHT GREEN (1):
    • AAPL - 2024-01-15 (0.1234)
  🟢 GREEN (1):
    • MSFT - 2024-01-15 (0.0567)
  🟠💪 LIGHT RED (1):
    • NVDA - 2024-01-15 (-0.0234)
  🔴 RED (1):
    • TSLA - 2024-01-15 (-0.0891)

*1WK*
  🟢💪 LIGHT GREEN (1):
    • AAPL - 2024-01-15 (0.1456)
  🟢 GREEN (1):
    • MSFT - 2024-01-15 (0.0789)
  🟠💪 LIGHT RED (1):
    • GOOGL - 2024-01-15 (-0.0345)

*1MO*
  🟢💪 LIGHT GREEN (1):
    • AAPL - 2024-01-15 (0.1678)
  🔴 RED (1):
    • NVDA - 2024-01-15 (-0.1234)

_Generated at 14:30:25_
```

### Ejemplo de Resumen Diario

```
📈 DAILY SUMMARY - 2024-01-15
Total signals: 9

*1D*
  🟢💪 LIGHT GREEN: 1
  🟢 GREEN: 1
  🟠💪 LIGHT RED: 1
  🔴 RED: 1

*1WK*
  🟢💪 LIGHT GREEN: 1
  🟢 GREEN: 1
  🟠💪 LIGHT RED: 1

*1MO*
  🟢💪 LIGHT GREEN: 1
  🔴 RED: 1

Most Active Tickers:
• AAPL: 3 signals
• MSFT: 2 signals
• NVDA: 2 signals
```

## Estados Críticos

Los estados críticos (`green_hh` y `red_ll`) representan:
- **`green_hh`**: LIGHT GREEN (Higher High) - Señal muy fuerte de compra
- **`red_ll`**: RED (Lower Low) - Señal muy fuerte de venta

Estos estados se envían inmediatamente con formato especial:

```
🚨 CRITICAL ALERT 🚨
AAPL 1d — BX Trender cambió a 🟢💪 LIGHT GREEN (Higher High) el 2024-01-15
BX_value=0.123400
```

## Ventajas

### ✅ Beneficios del Batching
- **Menos spam**: Reduce el número de mensajes enviados
- **Mejor legibilidad**: Las alertas se agrupan por timeframe y luego por color
- **Organización visual**: Fácil identificación de patrones por timeframe
- **Información contextual**: Incluye estadísticas y resúmenes organizados
- **Flexibilidad**: Configuración adaptable a diferentes necesidades

### ⚡ Rendimiento
- **Menos llamadas a la API**: Reduce el uso de la API de Telegram
- **Mejor experiencia**: Los usuarios reciben información más organizada
- **Escalabilidad**: Funciona bien con muchos tickers y timeframes

## Uso

### Ejecutar con Batching
```bash
python bxtrender_bot.py
```

### Probar el Sistema
```bash
python test_batching_format.py
```

### Deshabilitar Batching
```yaml
telegram:
  batching:
    enabled: false
```

## Monitoreo

El sistema incluye logs detallados:
- `ALERTA DETECTADA`: Cuando se detecta un cambio de estado
- `ALERTA CRÍTICA ENVIADA`: Cuando se envía una alerta crítica
- `Batch enviado con X alertas`: Cuando se envía un batch
- `Resumen diario enviado`: Cuando se envía el resumen diario

## Personalización

### Modificar Estados Críticos
```yaml
critical_states: ["green_hh", "red_ll", "green_lh"]  # Añadir más estados críticos
```

### Ajustar Timeouts
```yaml
batch_timeout_seconds: 60    # Esperar más tiempo antes de enviar
max_alerts_per_batch: 10     # Más alertas por mensaje
```

### Cambiar Hora del Resumen
```yaml
summary_time: "20:00"        # Resumen a las 8 PM
```
