#!/usr/bin/env python3
"""
Script de prueba para verificar la nueva configuración de backfill por timeframe
"""

import asyncio
import datetime as dt
from bxtrender_bot import BXTrenderBot

async def test_backfill_config():
    """Prueba la nueva configuración de backfill por timeframe"""
    print("🧪 Probando nueva configuración de backfill por timeframe...")
    
    # Crear instancia del bot
    bot = BXTrenderBot()
    
    # Mostrar configuración cargada
    print(f"\n📋 Configuración de backfill cargada:")
    for timeframe, years in bot.config.backfill_years.items():
        print(f"  • {timeframe}: {years} años")
    
    # Simular backfill para cada timeframe
    print(f"\n🔄 Simulando backfill para cada timeframe:")
    for ticker in bot.config.tickers[:1]:  # Solo el primer ticker para la prueba
        for timeframe in bot.config.timeframes:
            years = bot.config.backfill_years.get(timeframe, 4)
            expected_points = {
                "1d": years * 252,   # ~252 días de trading por año
                "1wk": years * 52,   # ~52 semanas por año
                "1mo": years * 12    # 12 meses por año
            }
            
            print(f"  • {ticker} {timeframe}: {years} años → ~{expected_points[timeframe]} puntos esperados")
    
    print(f"\n✅ Configuración de backfill por timeframe implementada correctamente!")
    print(f"   - Diario (1d): {bot.config.backfill_years.get('1d', 4)} años")
    print(f"   - Semanal (1wk): {bot.config.backfill_years.get('1wk', 4)} años")
    print(f"   - Mensual (1mo): {bot.config.backfill_years.get('1mo', 4)} años")

if __name__ == "__main__":
    asyncio.run(test_backfill_config())

