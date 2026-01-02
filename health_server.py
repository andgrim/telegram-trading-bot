from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import threading
import os
import time
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ['/', '/health', '/ping', '/status']:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response_data = {
                'status': 'online',
                'service': 'Telegram Trading Bot',
                'version': '4.0.0',
                'timestamp': time.time(),
                'uptime': getattr(self.server, 'start_time', time.time())
            }
            
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            logger.info(f"✅ Health check da {self.client_address[0]}")
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass

def start_health_server():
    """Avvia il server health in un thread separato"""
    port = int(os.environ.get('PORT', 8080))
    
    def run_server():
        try:
            with socketserver.TCPServer(('0.0.0.0', port), HealthHandler) as httpd:
                httpd.start_time = time.time()
                logger.info(f"✅ Health server avviato sulla porta {port}")
                logger.info(f"🔗 Endpoint: http://0.0.0.0:{port}/health")
                httpd.serve_forever()
        except Exception as e:
            logger.error(f"❌ Errore health server: {e}")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    time.sleep(2)
    
    return server_thread

if __name__ == "__main__":
    start_health_server()
    while True:
        time.sleep(1)