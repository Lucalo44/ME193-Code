"""Stdlib HTTP server for the two pages and their JSON API.

    GET  /            phone page (web/nav.html): GPS-style navigation, record, DAgger
    GET  /dash        laptop page (web/dash.html): raw IMU, movement, live decision view
    GET  /api/state   phone state (instruction, progress, trail, taps)
    GET  /api/live?seq=N&step=M   new raw samples + step records (dashboard)
    GET  /api/meta    feature names, maneuvers, movement labels
    GET  /api/music   the songs in web/music/ (the phone page's radio)
    GET  /music/<file>   one song (supports Range, which phones need to play audio)
    POST /api/start   {"mode": "navigate"|"record"|"dagger", "person": "..."}
    POST /api/tap     {"action": "left"|"right"|"stairs_start"|"stairs_end"|"arrive"|"wrong"}
    POST /api/stop    {"discard": false}
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import config as C
from nav.activity import LABELS
from nav.features import feature_names

PAGES = {"/": "nav.html", "/dash": "dash.html",
         "/map.js": "map.js", "/interior.js": "interior.js"}      # made by tools/make_map.py, make_interior.py


AUDIO = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav", ".ogg": "audio/ogg"}


def _json_default(o):
    try:
        return o.item()            # numpy scalars
    except AttributeError:
        return str(o)


class NavServer:
    def __init__(self, session, port=8080, web_dir=C.WEB_DIR):
        self.session = session
        web_dir = Path(web_dir)
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _songs(self):
                d = web_dir / "music"
                return sorted(f for f in d.iterdir() if f.suffix.lower() in AUDIO) if d.is_dir() else []

            def _song(self, name):
                f = next((s for s in self._songs() if s.name == name), None)     # only files listed in web/music
                if f is None:
                    return self._send(404, b"not found", "text/plain")
                size, start, end = f.stat().st_size, 0, None
                rng = self.headers.get("Range", "")
                try:
                    if rng.startswith("bytes="):
                        a, b = rng[6:].split(",")[0].split("-")
                        start, end = (int(a), int(b) if b else None) if a else (max(0, size - int(b)), None)
                except ValueError:
                    rng = ""
                end = size - 1 if end is None or end >= size else end
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    return self.end_headers()
                self.send_response(206 if rng else 200)
                self.send_header("Content-Type", AUDIO[f.suffix.lower()])
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                if rng:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                with f.open("rb") as fh:
                    fh.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = fh.read(min(65536, left))
                        if not chunk:
                            break
                        try:
                            self.wfile.write(chunk)
                        except (BrokenPipeError, ConnectionResetError):   # the phone skipped to another song
                            return
                        left -= len(chunk)

            def _json(self, obj):
                self._send(200, json.dumps(obj, default=_json_default).encode(), "application/json")

            def do_GET(self):
                u = urlparse(self.path)
                q = parse_qs(u.query)
                if u.path in PAGES:
                    f = web_dir / PAGES[u.path]
                    ctype = "text/javascript" if f.suffix == ".js" else "text/html"
                    return self._send(200, f.read_bytes(), ctype + "; charset=utf-8")
                if u.path == "/api/state":
                    return self._json(srv.session.state())
                if u.path == "/api/music":
                    return self._json([s.name for s in self._songs()])
                if u.path.startswith("/music/"):
                    return self._song(unquote(u.path[len("/music/"):]))
                if u.path == "/api/live":
                    return self._json(srv.session.live(int(q.get("seq", [0])[0]), int(q.get("step", [0])[0])))
                if u.path == "/api/meta":
                    return self._json({"features": feature_names(), "maneuvers": C.MANEUVERS,
                                       "movement": LABELS, "fs": C.FS, "route": C.ROUTE,
                                       "tap_actions": C.TAP_ACTIONS, "people": C.PEOPLE})
                self._send(404, b"not found", "text/plain")

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    body = {}
                s = srv.session
                if self.path == "/api/start":
                    s.start(body.get("mode", "navigate"), body.get("person", C.PEOPLE[0]))
                elif self.path == "/api/tap":
                    s.tap(body.get("action", ""))
                elif self.path == "/api/stop":
                    s.stop(discard=bool(body.get("discard")))
                else:
                    return self._send(404, b"not found", "text/plain")
                self._json(s.state())

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        print(f"[nav] phone page  http://<this device>:{port}/   laptop dashboard  http://<this device>:{port}/dash")

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
