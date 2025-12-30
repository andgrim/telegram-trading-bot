# health_server.py
import http.server
import socketserver
import threading
import os
import subprocess
import sys

class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ['/', '/health', '/ping']:
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()
            # Codifica esplicitamente in UTF-8
            response = '🤖 Telegram Trading Bot is running'.encode('utf-8')
            self.wfile.write(response)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def log_message(self, format, *args):
        # Disabilita logging
        pass

def start_health_server():
    """Avvia un server HTTP semplice per health check"""
    port = int(os.environ.get('PORT', 8080))
    
    with socketserver.TCPServer(('0.0.0.0', port), HealthHandler) as httpd:
        print(f"✅ Health server started on port {port}")
        httpd.serve_forever()

if __name__ == "__main__":
    print("🚀 Starting Telegram Trading Bot with health server...")
    
    # Avvia il server HTTP in un thread separato
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    
    # Avvia il bot Telegram (nel thread principale)
    print("🤖 Starting Telegram bot...")
    subprocess.run([sys.executable, "telegram_bot.py"])