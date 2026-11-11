"""Serves run_safe.py's annotated view (camera + virtual safe panel) as an MJPEG stream, so a
laptop can watch the UNO Q's camera. Stdlib only.

    GET /         a tiny page with the stream and a "press button" link (any browser)
    GET /stream   multipart/x-mixed-replace JPEG stream (view_safe.py reads this)
    GET /press    presses the safe's button, same as the physical one
    GET /state    {"state": "OPEN" | "IDLE" | "ARMED" | "DENIED" | "LOCKOUT", "status": ..., "steps": n}

The view is drawn and JPEG-encoded on the streamer's own thread, from the newest camera
frame and the safe's latest state (a render() callback from run_safe.py). So the stream runs
at camera speed instead of waiting on the nets, and the nets never wait on the stream.
Nothing is drawn or encoded while no viewer is connected.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

STREAM_FPS = 15
JPEG_QUALITY = 75
BOUNDARY = b"safeframe"
PAGE = b"""<!doctype html><title>Hand Safe</title>
<body style="background:#111;color:#ddd;font-family:sans-serif;text-align:center">
<img src="/stream" style="max-width:100%"><p><a href="/press" target="p" style="color:#9cf">press button</a>
<iframe name="p" style="display:none"></iframe></body>"""


class FrameStreamer:
    def __init__(self, port, render, on_press=None, state=None):
        self.render, self.on_press, self.state = render, on_press, state
        self.running = True
        self.jpeg, self.seq = None, 0
        self.cond = threading.Condition()
        self.clients = 0
        self.last_publish = 0.0
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):          # keep the app log clean
                pass

            def do_GET(self):
                if self.path.startswith("/stream"):
                    streamer._serve_stream(self)
                elif self.path.startswith("/state"):
                    info = streamer.state() if streamer.state else {}
                    self._reply(json.dumps(info).encode(), "application/json")
                elif self.path.startswith("/press"):
                    if streamer.on_press:
                        streamer.on_press()
                    print("[stream] button pressed from viewer")
                    self._reply(b"pressed", "text/plain")
                else:
                    self._reply(PAGE, "text/html")

            def _reply(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        threading.Thread(target=self._render_loop, daemon=True).start()
        print(f"[stream] serving the safe's view on port {port}")

    def _render_loop(self):
        while self.running:
            t0 = time.time()
            if self.clients > 0:
                try:
                    img = self.render()
                    if img is not None:
                        self.publish(img)
                except Exception as e:             # never let a drawing bug kill the stream
                    print("[stream] render error:", e)
            time.sleep(max(0.005, 1.0 / STREAM_FPS - (time.time() - t0)))

    def publish(self, img):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        self.last_publish = time.time()
        with self.cond:
            self.jpeg, self.seq = buf.tobytes(), self.seq + 1
            self.cond.notify_all()

    def _serve_stream(self, h):
        h.send_response(200)
        h.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + BOUNDARY.decode())
        h.send_header("Cache-Control", "no-cache")
        h.end_headers()
        with self.cond:
            self.clients += 1
        print(f"[stream] viewer connected ({self.clients} watching)")
        seen = -1
        try:
            while True:
                with self.cond:
                    self.cond.wait_for(lambda: self.seq != seen, timeout=5)
                    jpeg, seen = self.jpeg, self.seq
                if jpeg is None:
                    continue
                h.wfile.write(b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n"
                              + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with self.cond:
                self.clients -= 1
            print(f"[stream] viewer left ({self.clients} watching)")

    def close(self):
        self.running = False
        self.server.shutdown()
        self.server.server_close()
