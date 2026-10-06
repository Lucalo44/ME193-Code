"""
server/ws_server.py -- serves web/ over HTTP and streams game state to the
browser over a WebSocket. Both run in background threads; the asyncio loop
lives entirely inside its own thread.

Thread-safe entry points for the rest of the program:
    publish_state(json_str)   latest snapshot; sent at STATE_BROADCAST_HZ
    publish_event(json_str)   sent immediately, in order
    publish_frame(jpeg_b64)   latest camera preview frame
"""

from __future__ import annotations

import asyncio
import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional, Set

from websockets.asyncio.server import broadcast, serve

import config as C
from server.protocol import camera_frame_message, parse_client_message


class _StaticHandler(SimpleHTTPRequestHandler):
    runtime: dict = {}

    def do_GET(self):  # noqa: N802 -- http.server naming
        if self.path.split("?")[0] == "/runtime.json":
            body = json.dumps(self.runtime).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):  # keep the terminal quiet
        pass


class GameServer:
    def __init__(self, web_dir: str, on_message: Callable[[dict], None],
                 http_port: int = C.HTTP_PORT, ws_port: int = C.WS_PORT, runtime: Optional[dict] = None):
        self.web_dir = web_dir
        self.on_message = on_message
        self.http_port = http_port
        self.ws_port = ws_port
        self.runtime = {"ws_port": ws_port, **(runtime or {})}
        self.clients: Set = set()
        self._state: Optional[str] = None
        self._state_dirty = False
        self._frame: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop: Optional[asyncio.Event] = None
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._ready = threading.Event()
        self.error: Optional[str] = None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.http_port}/"

    def start(self) -> None:
        handler = type("Handler", (_StaticHandler,), {"runtime": self.runtime})
        self._httpd = ThreadingHTTPServer(("localhost", self.http_port),
                                          functools.partial(handler, directory=self.web_dir))
        threading.Thread(target=self._httpd.serve_forever, daemon=True, name="http").start()
        threading.Thread(target=self._run_ws, daemon=True, name="websocket").start()
        self._ready.wait(timeout=5)
        if self.error:
            raise RuntimeError(self.error)

    def stop(self) -> None:
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._httpd:
            self._httpd.shutdown()

    # -- thread-safe publishing -------------------------------------------
    def publish_state(self, msg: str) -> None:
        self._state = msg
        self._state_dirty = True

    def publish_event(self, msg: str) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(self._send_all, msg)

    def publish_frame(self, jpeg_b64: str) -> None:
        if jpeg_b64:
            self._frame = camera_frame_message(jpeg_b64)

    # -- asyncio side -----------------------------------------------------
    def _send_all(self, msg: str) -> None:
        if self.clients:
            broadcast(self.clients, msg)

    def _run_ws(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:  # noqa: BLE001
            self.error = f"WebSocket server failed: {exc}"
            self._ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with serve(self._handler, "localhost", self.ws_port, max_size=2**20):
            self._ready.set()
            period = 1.0 / C.STATE_BROADCAST_HZ
            while not self._stop.is_set():
                if self._state_dirty and self._state:
                    self._state_dirty = False
                    self._send_all(self._state)
                if self._frame:
                    frame, self._frame = self._frame, None
                    self._send_all(frame)
                await asyncio.sleep(period)

    async def _handler(self, ws) -> None:
        self.clients.add(ws)
        try:
            async for raw in ws:
                msg = parse_client_message(raw)
                if msg:
                    self.on_message(msg)
        except Exception:
            pass
        finally:
            self.clients.discard(ws)
