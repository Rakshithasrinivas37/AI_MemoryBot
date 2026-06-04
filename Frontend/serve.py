import http.server
import socketserver
import webbrowser
import threading
import os

PORT = 5500
URL  = f"http://localhost:{PORT}"

class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def log_message(self, format, *args):
        pass  # suppress request logs

def open_browser():
    webbrowser.open(URL)

with socketserver.TCPServer(("", PORT), NoCacheHandler) as httpd:
    print(f"✅ Serving on port {PORT} with no cache")
    print(f"🌐 Opening {URL}")

    # Open browser after 1 second delay
    timer = threading.Timer(1.0, open_browser)
    timer.start()

    httpd.serve_forever()