#!/usr/bin/env python3
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("TELEGRAM_BOT_TOKEN")

if not token:
    print("❌ Token non trovato nel file .env")
    print("\n📝 Crea/modifica il file .env con:")
    print("TELEGRAM_BOT_TOKEN=il_tuo_token_qui")
else:
    print(f"✅ Token trovato: {token[:15]}...")
    
    # Test connessione
    import requests
    try:
        url = f"https://api.telegram.org/bot{token}/getMe"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('ok'):
                print(f"✅ Bot connesso: @{data['result']['username']}")
                print(f"📛 Nome: {data['result']['first_name']}")
            else:
                print("❌ Token non valido")
        else:
            print(f"❌ Errore API: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Errore connessione: {e}")