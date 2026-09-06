#!/usr/bin/env python3
"""Minimal HTTP server that serves the training status dashboard."""

import argparse
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

ROUTES = {
    "/": ("status.html", "text/html; charset=utf-8"),
    "/status.html": ("status.html", "text/html; charset=utf-8"),
    "/status.svg": ("status.svg", "image/svg+xml"),
    "/status.txt": ("status.txt", "text/plain; charset=utf-8"),
}


class StatusHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        # Strip query-string cache-busters (?t=…) before routing.
        path = self.path.split("?")[0]

        if path not in ROUTES:
            self._send(404, "text/plain; charset=utf-8", b"Not found\n")
            return

        filename, content_type = ROUTES[path]
        if not os.path.exists(filename):
            msg = f"{filename} not found — training has not started yet.\n"
            self._send(503, "text/plain; charset=utf-8", msg.encode())
            return

        with open(filename, "rb") as f:
            body = f.read()
        self._send(200, content_type, body)

    def _send(self, status, content_type, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Serve training status dashboard.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=6005, help="Bind port (default: 6005)")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), StatusHandler)
    print(f"Status dashboard at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
