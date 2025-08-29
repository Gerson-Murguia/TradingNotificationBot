import os
import requests
from dotenv import load_dotenv

# Cargar variables del .env
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("✅ Mensaje enviado con éxito")
    else:
        print("❌ Error al enviar mensaje:", response.text)

# --- PRUEBA ---
send_telegram_message("🚀 Test de alerta funcionando!")