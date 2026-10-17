#!/usr/bin/env python3
"""Serve the live demo page (demo/) on 127.0.0.1:5180, or the next free port.

/datacenter.json is read fresh from dashboard/public/datacenter.json on every request.
/three.min.js is served from deck/three.min.js. Python stdlib only.
"""
import http.server
import os
import socket
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo")
ROUTES = {
    "/datacenter.json": (os.path.join(ROOT, "dashboard", "public", "datacenter.json"), "application/json"),
    "/three.min.js": (os.path.join(ROOT, "deck", "three.min.js"), "text/javascript"),
}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DEMO, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ROUTES:
            file_path, ctype = ROUTES[path]
            try:
                with open(file_path, "rb") as fh:
                    body = fh.read()
            except OSError:
                self.send_error(404, "not built yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))


def free_port(start):
    for port in range(start, start + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit("no free port near %d" % start)


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 5180
    port = free_port(start)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("demo serving http://127.0.0.1:%d/" % port, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
