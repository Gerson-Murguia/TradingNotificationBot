# BX Trender Bot - Bot de Trading Inteligente

Un bot de trading automatizado que utiliza el indicador BX Trender para detectar cambios de tendencia en múltiples activos y timeframes, enviando alertas por Telegram.

## 🚀 Características Principales

- **Indicador BX Trender**: Implementación completa del indicador técnico con 4 estados de tendencia
- **Múltiples Timeframes**: Soporte para daily, weekly y monthly
- **Múltiples Activos**: Monitoreo simultáneo de varios tickers
- **Alertas en Tiempo Real**: Notificaciones automáticas por Telegram
- **Base de Datos SQLite**: Almacenamiento persistente de datos históricos
- **Logging Estructurado**: Sistema de logs completo con archivos y consola
- **Configuración Externa**: Parámetros configurables via YAML
- **Manejo de Errores Robusto**: Recuperación automática de fallos
- **Métricas y Monitoreo**: Estadísticas de rendimiento del bot

## 📋 Requisitos

- Python 3.8+
- Cuenta de Telegram con bot configurado
- Conexión a internet para datos de mercado

## 🛠️ Instalación

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd trading-bot
```

2. **Crear entorno virtual**:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**:
```bash
cp .env.example .env
# Editar .env con tus credenciales de Telegram
```

## ⚙️ Configuración

### Variables de Entorno (.env)
```env
TELEGRAM_TOKEN=tu_token_del_bot
TELEGRAM_CHAT_ID=tu_chat_id
```

### Archivo de Configuración (config.yaml)
```yaml
# Configuración de la base de datos
database:
  file: "prices.db"
  backfill_years: 3
  update_period: "60d"
  request_delay: 1.0

# Tickers y timeframes a monitorear
trading:
  tickers: ["AAPL", "MSFT", "NVDA"]
  timeframes: ["1d", "1wk", "1mo"]

# Parámetros del indicador BX Trender
indicators:
  short_l1: 5
  short_l2: 20
  short_l3: 5
  t3_length: 5
  t3_v: 0.7
  long_l1: 20
  long_l2: 5
```

## 🚀 Uso

### Ejecución Básica
```bash
python main.py
```

### Ejecución con Nueva Implementación (Recomendado)
```bash
python bxtrender_bot.py
```

### Ejecución con Configuración Personalizada
```bash
python bxtrender_bot.py --config mi_config.yaml
```

## 📊 Estados del Indicador BX Trender

El bot detecta 4 estados diferentes:

- 🟢💪 **LIGHT GREEN (Higher High)**: Tendencia alcista fuerte
- 🟢 **GREEN (Lower High)**: Tendencia alcista débil
- 🟠💪 **LIGHT RED (Higher Low)**: Tendencia bajista débil
- 🔴 **RED (Lower Low)**: Tendencia bajista fuerte

## 📁 Estructura del Proyecto

```
trading-bot/
├── main.py                 # Implementación original
├── bxtrender_bot.py        # Nueva implementación orientada a objetos
├── config.yaml             # Configuración del bot
├── requirements.txt        # Dependencias
├── .env                    # Variables de entorno
├── prices.db              # Base de datos SQLite
├── logs/                  # Directorio de logs
│   └── trading_bot_YYYYMMDD.log
└── README.md              # Este archivo
```

## 🔧 Arquitectura Mejorada

### Clases Principales

- **`BXTrenderBot`**: Clase principal que coordina todos los componentes
- **`DatabaseManager`**: Manejo de base de datos SQLite
- **`DataManager`**: Obtención y actualización de datos de mercado
- **`IndicatorCalculator`**: Cálculo de indicadores técnicos
- **`NotificationManager`**: Envío de alertas por Telegram
- **`Metrics`**: Recolección de métricas y estadísticas

### Características de la Nueva Implementación

1. **Logging Estructurado**:
   - Logs en archivo y consola
   - Niveles configurables (DEBUG, INFO, WARNING, ERROR)
   - Rotación automática de archivos

2. **Manejo de Errores Robusto**:
   - Try-catch en todas las operaciones críticas
   - Fallbacks para indicadores técnicos
   - Recuperación automática de fallos de red

3. **Configuración Externa**:
   - Parámetros en archivo YAML
   - Variables de entorno para credenciales
   - Fácil modificación sin tocar código

4. **Métricas y Monitoreo**:
   - Estadísticas de alertas enviadas
   - Conteo de errores por tipo
   - Tiempo de actividad
   - Health checks automáticos

## 📈 Logs y Monitoreo

### Niveles de Log
- **DEBUG**: Información detallada para desarrollo
- **INFO**: Información general del funcionamiento
- **WARNING**: Advertencias que no impiden la ejecución
- **ERROR**: Errores que requieren atención

### Archivos de Log
Los logs se guardan en `logs/trading_bot_YYYYMMDD.log` con formato:
```
2024-01-15 10:30:45 - BXTrenderBot - INFO - Telegram Bot inicializado correctamente.
2024-01-15 10:30:46 - BXTrenderBot - INFO - Backfill completado para AAPL 1d: 756 filas guardadas
```

## 🧪 Testing

### Ejecutar Tests
```bash
pytest tests/
```

### Tests Disponibles
- Tests unitarios para indicadores
- Tests de integración para base de datos
- Tests de notificaciones (mock)

## 🔒 Seguridad

- Las credenciales se almacenan en variables de entorno
- No se incluyen tokens en el código
- Conexiones seguras a APIs externas
- Validación de datos de entrada

## 🚨 Troubleshooting

### Problemas Comunes

1. **Error de conexión a Telegram**:
   - Verificar que el token sea válido
   - Confirmar que el chat_id sea correcto
   - Revisar conectividad de red

2. **Datos insuficientes**:
   - Aumentar `backfill_years` en config.yaml
   - Verificar que el ticker exista en Yahoo Finance

3. **Errores de base de datos**:
   - Verificar permisos de escritura
   - Eliminar archivo `prices.db` para reinicializar

### Logs de Debug
Para obtener más información, cambiar el nivel de log a DEBUG en `config.yaml`:
```yaml
logging:
  level: "DEBUG"
```

## 📝 Changelog

### v2.0.0 (Actual)
- ✅ Implementación orientada a objetos
- ✅ Logging estructurado
- ✅ Configuración externa YAML
- ✅ Manejo de errores robusto
- ✅ Métricas y monitoreo
- ✅ Health checks
- ✅ Documentación completa

### v1.0.0 (Original)
- ✅ Indicador BX Trender básico
- ✅ Alertas por Telegram
- ✅ Base de datos SQLite
- ✅ Múltiples timeframes

## 🤝 Contribuciones

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## ⚠️ Disclaimer

Este bot es para fines educativos y de investigación. No garantiza ganancias en trading. El trading conlleva riesgos significativos y puede resultar en pérdidas. Úsalo bajo tu propia responsabilidad.

## 📞 Soporte

Para soporte técnico o preguntas:
- Crear un issue en GitHub
- Revisar la documentación
- Consultar los logs de error

---

**¡Happy Trading! 🚀📈**
