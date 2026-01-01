# health_server.py
import http.server
import socketserver
import threading
import os
import sys
import time
from http import HTTPStatus

class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ['/', '/health', '/ping', '/status']:
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()
            response = '🤖 Telegram Trading Bot is running\n'.encode('utf-8')
            self.wfile.write(response)
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def log_message(self, format, *args):
        # Optional: enable for debugging
        # print(f"[Health Server] {args[0]} {args[1]} {args[2]}")
        pass

def start_health_server():
    """Avvia un server HTTP semplice per health check"""
    port = int(os.environ.get('PORT', 8080))
    
    # Tentativi di avvio
    for attempt in range(3):
        try:
            with socketserver.TCPServer(('0.0.0.0', port), HealthHandler) as httpd:
                print(f"✅ Health server started on port {port}")
                print(f"🔗 Local: http://localhost:{port}/health")
                httpd.serve_forever()
        except OSError as e:
            if attempt < 2:
                print(f"⚠️  Port {port} busy, retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"❌ Failed to start health server on port {port}: {e}")
                raise
        except KeyboardInterrupt:
            print("\n🛑 Health server stopped")
            break
        except Exception as e:
            print(f"❌ Health server error: {e}")
            raise

if __name__ == "__main__":
    # Solo per testing diretto
    start_health_server()