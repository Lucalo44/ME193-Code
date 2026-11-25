#!/usr/bin/env python3
"""Run Ground Control on a laptop: simulated board, same dashboard.

    venv/bin/python python/devserver.py                    # FakeLLM, no MQTT
    venv/bin/python python/devserver.py --llm http://127.0.0.1:9999 --mqtt
        (real model on the board via `adb forward tcp:9999 tcp:9999`)

Then open http://localhost:7070. SimWebUI mimics the WebUI brick's API
(on_message / send_message / "<event>_response") on python-socketio, so
core.py can't tell the difference.
"""
import argparse
import asyncio
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("GC_SIM", "1")

import socketio  # noqa: E402
import uvicorn  # noqa: E402


class SimWebUI:
    def __init__(self, assets, port):
        self.port = port
        self.sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        assets = str(assets)
        static = {"/": f"{assets}/index.html"}
        for p in Path(assets).rglob("*"):
            if p.is_file():
                static["/" + str(p.relative_to(assets))] = str(p)
        self.app = socketio.ASGIApp(self.sio, static_files=static)
        self.loop = None

    def on_message(self, event, fn):
        @self.sio.on(event)
        async def _h(sid, data=None):
            res = await asyncio.get_running_loop().run_in_executor(None, fn, sid, data)
            if res is not None:
                await self.sio.emit(f"{event}_response", res, to=sid)

    def send_message(self, event, payload, room=None):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.sio.emit(event, payload, to=room), self.loop)

    def serve(self):
        config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port, log_level="warning")
        server = uvicorn.Server(config)

        async def run():
            self.loop = asyncio.get_running_loop()
            await server.serve()
        threading.Thread(target=lambda: asyncio.run(run()), daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7070)
    ap.add_argument("--llm", default="", help="llama.cpp URL; default FakeLLM")
    ap.add_argument("--model", default="")
    ap.add_argument("--mqtt", action="store_true")
    ap.add_argument("--lego-host", default="", help="LEGO service host (e.g. the board's IP)")
    a = ap.parse_args()

    from core import GroundControl
    from hal.sim_hal import SimHal
    from planner.fake_llm import FakeLLM
    from planner.llm_client import LlmClient

    llm = LlmClient(a.llm, a.model or None) if a.llm else FakeLLM(delay_s=0.18)
    if a.lego_host:
        os.environ["GC_LEGO_HOST"] = a.lego_host
    ui = SimWebUI(HERE.parent / "assets", a.port)
    ui.serve()
    gc = GroundControl(SimHal(), ui=ui, llm=llm, lego=bool(a.lego_host), mqtt=a.mqtt)
    gc.start(warm=bool(a.llm))
    print(f"Ground Control (sim) on http://localhost:{a.port}  llm={'fake' if not a.llm else a.llm}", flush=True)
    try:
        while True:
            gc.heartbeat()
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
