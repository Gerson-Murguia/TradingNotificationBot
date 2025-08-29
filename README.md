# BX Trender Bot 🤖📈

Un bot de trading automatizado que utiliza el indicador BX Trender para detectar señales de compra y venta en múltiples timeframes. El bot monitorea activos en tiempo real y envía alertas a través de Telegram.

## 🚀 Características

- **Indicador BX Trender**: Implementación completa del indicador técnico
- **Múltiples Timeframes**: Soporte para diario, semanal y mensual
- **Alertas en Tiempo Real**: Notificaciones automáticas vía Telegram
- **Base de Datos Local**: Almacenamiento de precios históricos en SQLite
- **Backfill Automático**: Descarga automática de datos históricos
- **Batching Inteligente**: Agrupación de alertas para evitar spam
- **Métricas y Monitoreo**: Seguimiento del rendimiento del bot
- **Configuración Flexible**: Archivo YAML para configuración externa
- **Logging Avanzado**: Sistema de logs con rotación automática

## 📋 Requisitos

- Python 3.8 o superior
- Cuenta de Telegram (para recibir alertas)
- Conexión a internet

## 🛠️ Instalación

1. **Clonar o descargar el proyecto**
   ```bash
   git clone <repository-url>
   cd Trading
   ```

2. **Crear entorno virtual**
   ```bash
   python -m venv venv
   ```

3. **Activar entorno virtual**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

4. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configurar variables de entorno**
   ```bash
   cp env.example .env
   ```
   
   Editar el archivo `.env` con tus credenciales:
   ```env
   TELEGRAM_TOKEN=tu_token_del_bot_aqui
   TELEGRAM_CHAT_ID=tu_chat_id_aqui
   ```

## ⚙️ Configuración

### Configuración del Bot

Edita el archivo `config.yaml` para personalizar el comportamiento del bot:

```yaml
# Tickers a monitorear
trading:
  tickers: ["AAPL", "MSFT", "NVDA"]
  timeframes: ["1d", "1wk", "1mo"]

# Parámetros del indicador
indicators:
  short_l1: 5
  short_l2: 20
  short_l3: 5
  t3_length: 5
  t3_v: 0.7
  long_l1: 20
  long_l2: 5
```

### Configuración de Telegram

1. **Crear un bot en Telegram**:
   - Habla con [@BotFather](https://t.me/botfather)
   - Usa el comando `/newbot`
   - Sigue las instrucciones para crear tu bot

2. **Obtener el Chat ID**:
   - Envía un mensaje a tu bot
   - Visita: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Copia el `chat_id` de la respuesta

3. **Configurar en .env**:
   ```env
   TELEGRAM_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   TELEGRAM_CHAT_ID=123456789
   ```

## 🚀 Uso

### Ejecutar el Bot

```bash
python bxtrender_bot.py
```

### Modos de Operación

El bot puede ejecutarse en diferentes modos:

- **Modo Normal**: Monitoreo continuo con alertas
- **Modo Backfill**: Descarga de datos históricos
- **Modo Test**: Pruebas sin enviar alertas

### Archivos de Log

Los logs se guardan en el directorio `logs/` con rotación automática:
- `trading_bot_YYYYMMDD.log`: Logs diarios
- Configuración de retención en `config.yaml`

## 📊 Indicador BX Trender

El bot implementa el indicador BX Trender que combina:

- **Señales Short Term**: Basadas en promedios móviles de corto plazo
- **Señales Long Term**: Basadas en promedios móviles de largo plazo
- **Filtros T3**: Suavizado de señales para reducir ruido

### Estados del Indicador

- 🟢 **Green HH**: Máximo más alto (señal alcista)
- 🟢 **Green LH**: Mínimo más alto (señal alcista)
- 🔴 **Red LL**: Mínimo más bajo (señal bajista)
- 🔴 **Red HL**: Máximo más bajo (señal bajista)

## 🔧 Personalización

### Agregar Nuevos Tickers

Edita `config.yaml`:
```yaml
trading:
  tickers: ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]
```

### Modificar Parámetros del Indicador

```yaml
indicators:
  short_l1: 3    # Más sensible
  short_l2: 15   # Menos sensible
  t3_v: 0.8      # Más suavizado
```

### Configurar Batching

```yaml
telegram:
  batching:
    enabled: true
    max_alerts_per_batch: 10
    batch_timeout_seconds: 60
```

## 📁 Estructura del Proyecto

```
Trading/
├── bxtrender_bot.py      # Bot principal
├── config.yaml           # Configuración
├── requirements.txt      # Dependencias
├── env.example          # Variables de entorno de ejemplo
├── .env                 # Variables de entorno (crear)
├── prices.db            # Base de datos SQLite
├── state.json           # Estado del bot
├── logs/                # Archivos de log
│   └── trading_bot_*.log
├── venv/                # Entorno virtual
└── README.md           # Este archivo
```

## 🧪 Testing

Ejecutar tests:
```bash
pytest test_*.py
```

Archivos de test disponibles:
- `test_backfill_config.py`: Pruebas de configuración de backfill
- `test_batching_format.py`: Pruebas de formato de batching
- `test_telegram_bot.py`: Pruebas del bot de Telegram

## 📈 Métricas y Monitoreo

El bot incluye un sistema de métricas que registra:
- Alertas enviadas
- Errores ocurridos
- Actualizaciones de datos
- Tiempo de actividad
- Cálculos realizados

### Health Check

El bot realiza verificaciones de salud cada 5 minutos por defecto, configurable en `config.yaml`.

## ⚠️ Advertencias

- **No es consejo financiero**: Este bot es solo para fines educativos
- **Riesgo de pérdida**: El trading conlleva riesgos financieros
- **Paper Trading**: Se recomienda probar primero con datos simulados
- **Monitoreo**: Siempre supervisa el bot durante la operación

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

---

**⚠️ Descargo de responsabilidad**: Este software es solo para fines educativos. El trading conlleva riesgos financieros significativos. Los desarrolladores no se hacen responsables de pérdidas financieras.
