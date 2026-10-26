import errno
import http.server
from functools import partial
from pathlib import Path
import webbrowser

PORT = 5500
FRONTEND_DIR = Path(__file__).resolve().parent


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # suppress request logs


class FrontendServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def create_server(port=PORT):
    handler = partial(NoCacheHandler, directory=str(FRONTEND_DIR))
    try:
        return FrontendServer(("127.0.0.1", port), handler)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        # Let the OS choose a free port without interrupting the existing server.
        server = FrontendServer(("127.0.0.1", 0), handler)
        print(f"Port {port} is in use; using port {server.server_port} instead.")
        return server


def main():
    with create_server() as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Serving MemoryBot at {url}")
        print("Press Ctrl+C to stop.")
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
