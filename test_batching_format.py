#!/usr/bin/env python3
"""
Script de prueba para mostrar el nuevo formato de agrupación por timeframe
"""

import asyncio
import datetime as dt
from bxtrender_bot import BXTrenderBot

async def test_new_format():
    """Prueba el nuevo formato de agrupación por timeframe"""
    print("🚀 Probando nuevo formato de agrupación por timeframe...")
    
    # Crear instancia del bot
    bot = BXTrenderBot()
    
    # Simular alertas con diferentes timeframes
    batch_manager = bot.notification_manager.batch_manager
    
    # Simular alertas variadas
    test_alerts = [
        # Daily alerts
        ("AAPL", "1d", "2024-01-15", "green_hh", 0.1234),
        ("MSFT", "1d", "2024-01-15", "green_lh", 0.0567),
        ("NVDA", "1d", "2024-01-15", "red_hl", -0.0234),
        ("TSLA", "1d", "2024-01-15", "red_ll", -0.0891),
        
        # Weekly alerts
        ("AAPL", "1wk", "2024-01-15", "green_hh", 0.1456),
        ("MSFT", "1wk", "2024-01-15", "green_lh", 0.0789),
        ("GOOGL", "1wk", "2024-01-15", "red_hl", -0.0345),
        
        # Monthly alerts
        ("AAPL", "1mo", "2024-01-15", "green_hh", 0.1678),
        ("NVDA", "1mo", "2024-01-15", "red_ll", -0.1234),
    ]
    
    print(f"📊 Simulando {len(test_alerts)} alertas...")
    
    for ticker, timeframe, date, state, value in test_alerts:
        is_critical = batch_manager.add_alert(ticker, timeframe, date, state, value)
        print(f"  • {ticker} {timeframe} -> {state} {'🚨 (CRÍTICO)' if is_critical else ''}")
    
    print(f"\n📋 Estado del batch:")
    print(f"  • Alertas pendientes: {len(batch_manager.pending_alerts)}")
    print(f"  • Debe enviar batch: {batch_manager.should_send_batch()}")
    
    # Mostrar mensaje del batch con nuevo formato
    batch_msg = batch_manager.get_batch_message()
    print(f"\n📨 Mensaje del batch (NUEVO FORMATO):")
    print("=" * 60)
    print(batch_msg)
    print("=" * 60)
    
    # Mostrar resumen diario con nuevo formato
    summary_msg = batch_manager.get_daily_summary()
    print(f"\n📈 Resumen diario (NUEVO FORMATO):")
    print("=" * 60)
    print(summary_msg)
    print("=" * 60)
    
    print("\n✅ Prueba completada!")
    print("\n💡 Ventajas del nuevo formato:")
    print("  • Agrupación por timeframe (1D, 1WK, 1MO)")
    print("  • Dentro de cada timeframe, ordenado por color")
    print("  • Más fácil de leer y analizar")
    print("  • Mejor organización visual")

if __name__ == "__main__":
    asyncio.run(test_new_format())

