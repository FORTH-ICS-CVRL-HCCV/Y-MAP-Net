#!/usr/bin/env python3
"""Minimal HTTP server that serves the current training status from status.txt."""

import argparse
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

STATUS_FILE = "status.txt"


class StatusHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "r") as f:
                body = f.read()
            status = 200
        else:
            body = "No status available yet — training has not started or status.txt has not been written.\n"
            status = 503

        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, fmt, *args):
        # Suppress per-request console noise; comment out to re-enable.
        pass


def main():
    parser = argparse.ArgumentParser(description="Serve training status over HTTP.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=6005, help="Bind port (default: 6005)")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), StatusHandler)
    print(f"Status server running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
