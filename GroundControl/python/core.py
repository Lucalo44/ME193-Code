"""Ground Control wiring. main.py (board), devserver.py (laptop) and the tests
all build this same object; only the HAL, UI and LLM client differ.

Threads
  main      App.run heartbeat loop (board) -- only feeds the MCU watchdog
  paho      MQTT in/out; answers /status and stop inline
  worker    one job at a time: plan (LLM) -> execute
  status    retained status every STATUS_PERIOD_S
  lego      connects to the LEGO BLE service, retries, polls its state
  viz       forwards events to the dashboard
  matrix    animates the LED matrix
"""
import threading
import time

import config
import skills  # noqa: F401  (registers skills)
from dance.conductor import Conductor
from executor import Executor
from hal.performers import CarPerformer, LegoPerformer, SimLegoPerformer
from jobs import Job, JobQueue, QueueFull
from mqtt_io import MqttIO
from planner.llm_client import LlmClient, LlmError
from planner.planner import Planner
from skills.context import Ctx
from skills.matrix import Matrix
from status import Status
from viz.bus import VizBus


class GroundControl:
    def __init__(self, hal, ui=None, llm=None, lego=config.LEGO_ENABLED, mqtt=True):
        self.status = Status()
        self.viz = VizBus(ui)
        self.ui = ui
        self.hal = hal
        hal.on_frame = lambda f: self.viz.emit("matrix", {"frame": f})
        hal.on_drive = lambda l, r: self.viz.emit("drive", {"l": l, "r": r, "pose": hal.pose()})
        self.matrix = Matrix(hal)
        self.queue = JobQueue()
        self.use_mqtt = mqtt
        self.mqtt = MqttIO(self.status, self.submit, self.estop)
        self.mqtt.on_publish_hook = self._on_publish
        self.llm = llm or LlmClient()
        self.planner = Planner(self.llm, emit=self.viz.emit)
        self.car = CarPerformer(hal, self.matrix)
        self.sim_lego = SimLegoPerformer()
        self.real_lego = LegoPerformer() if lego else None
        self.performers = {"car": self.car, "lego": self.sim_lego}
        self.conductor = Conductor(self.performers, emit=self.viz.emit, status=self.status)
        self.executor = Executor(self.make_ctx, emit=self.viz.emit, status=self.status)
        self.status.on_change(self._on_status)
        self.status.update(car={"connected": hal.connected, "left": 0, "right": 0},
                           imu=hal.imu(), lego={"connected": False, "state": "sim"})
        self._stop = threading.Event()

    # ---------------------------------------------------------------- lifecycle
    def start(self, warm=True):
        if self.use_mqtt:
            self.mqtt.start()
        for target, name in [(self._worker, "worker"), (self._status_ticker, "status")]:
            threading.Thread(target=target, daemon=True, name=name).start()
        if self.real_lego is not None:
            threading.Thread(target=self._lego_loop, daemon=True, name="lego").start()
        if warm:
            threading.Thread(target=self._warm, daemon=True, name="warm").start()
        else:
            self.status.update(llm_ready=True)
        if self.ui is not None:
            self._wire_ui()
        self.matrix.icon("smile")

    def heartbeat(self):
        """Call from the main loop every HEARTBEAT_S."""
        try:
            self.hal.heartbeat()
        except Exception as e:  # noqa: BLE001
            print(f"[hb] {e!r}", flush=True)

    def _warm(self):
        self.status.set_state("PLANNING", "Warming up the model (prefilling the fixed prompt)")
        self.matrix.thinking()
        for attempt in range(30):
            try:
                r = self.planner.warm()
                print(f"[llm] warm-up done in {r['ms']} ms, timings {r['timings']}", flush=True)
                self.viz.emit("llm.warm", r)
                self.status.update(llm_ready=True)
                break
            except (LlmError, OSError) as e:
                print(f"[llm] warm-up attempt {attempt}: {e}", flush=True)
                time.sleep(5)
        self.status.set_state("IDLE", "Waiting for a command")
        self.matrix.icon("smile")

    # ---------------------------------------------------------------- inputs
    def submit(self, text, source="mqtt", reply_to=None, plan=None):
        text = (text or "").strip()[: config.MAX_CMD_CHARS]
        if not text:
            return False, "empty command"
        job = Job(text=text, source=source, reply_to=reply_to or config.TOPIC_REPLY, plan=plan)
        try:
            self.queue.put(job)
        except QueueFull as e:
            return False, str(e)
        self.status.update(queue=len(self.queue))
        self.viz.emit("job.queued", {"id": job.id, "text": text, "source": source})
        return True, job.id

    def estop(self, source="mqtt"):
        """Halt everything NOW. Called from the MQTT thread, the UI, or a reflex."""
        t0 = time.time()
        self.conductor.stop_all()
        try:
            self.hal.stop()
        except Exception as e:  # noqa: BLE001
            print(f"[estop] hal.stop failed: {e!r}", flush=True)
        cur, dropped = self.queue.cancel_all()
        self.matrix.icon("x")
        ms = round((time.time() - t0) * 1000, 1)
        self.status.set_state("ESTOP", f"Stopped by {source} ({ms} ms)", queue=0)
        self.viz.emit("reflex.action", {"word": "stop", "source": source, "latency_ms": ms,
                                        "cancelled": cur.id if cur else None, "dropped": len(dropped)})
        threading.Timer(2.0, self._clear_estop).start()
        return ms

    def _clear_estop(self):
        if self.status.get("state") == "ESTOP" and self.queue.current is None:
            self.status.set_state("IDLE", "Waiting for a command")

    # ---------------------------------------------------------------- worker
    def make_ctx(self, job):
        return Ctx(hal=self.hal, matrix=self.matrix, publish=self.mqtt.publish, status=self.status,
                   conductor=self.conductor, reply_to=job.reply_to, cancel=job.cancel)

    def _worker(self):
        while not self._stop.is_set():
            job = self.queue.get(timeout=0.5)
            if job is None:
                continue
            try:
                self.run_job(job)
            except Exception as e:  # noqa: BLE001
                print(f"[worker] job {job.id} crashed: {e!r}", flush=True)
                self.status.set_state("ERROR", f"Job {job.id} crashed: {e}")
            finally:
                self.queue.done()
                self.status.update(queue=len(self.queue), job=None)

    def run_job(self, job):
        t_start = time.time()
        job.mark("queue", job.created, t_start)
        self.viz.emit("job.new", {"id": job.id, "text": job.text, "source": job.source,
                                  "reply_to": job.reply_to, "t": t_start})
        self.status.set_state("PLANNING", f"Planning: '{job.text[:60]}'",
                              job={"id": job.id, "step": 0, "of": 0, "skill": None},
                              queue=len(self.queue))
        if job.plan is not None:                      # hand-written plan: no LLM
            plan, info = job.plan, {}
            self.viz.emit("llm.done", {"id": job.id, "plan": plan, "raw_text": "(hand-written plan)",
                                       "timings": {}, "decisions": []})
        else:
            self.matrix.thinking()
            try:
                plan, info = self.planner.plan(job)
            except (LlmError, OSError) as e:
                plan, info = None, {"error": f"LLM unavailable: {e}"}
        if self.planner.last_tok_s:
            self.status.update(llm_tok_s=self.planner.last_tok_s)
        llm_t = info.get("timings") or {}
        if job.cancel.is_set():
            self._finish(job, "stopped", t_start, llm_t, [])
            return
        if plan is None:
            reply = f"Sorry, I can't do that: {info.get('error', 'no plan')}"
            self.mqtt.publish(job.reply_to, reply)
            self.matrix.icon("x")
            self._finish(job, reply, t_start, llm_t, [])
            return
        job.plan = plan
        self.status.set_state("EXECUTING", f"Executing {len(plan.get('steps', []))} step(s)")
        res = self.executor.run(job, plan)
        if self.matrix.content.get("kind") == "thinking" and not job.cancel.is_set():
            self.matrix.icon("check" if not res["errors"] else "x")   # plan didn't use the matrix
        self._finish(job, res["reply"], t_start, llm_t, res.get("steps", []), res)

    def _finish(self, job, reply, t_start, llm_t, steps, res=None):
        job.reply = reply
        t_end = time.time()
        # Latency Gantt segments (seconds relative to job.created)
        segs = [{"name": "queue", "t0": 0, "t1": round(t_start - job.created, 3)}]
        if llm_t:
            pre, dec = llm_t.get("prefill_ms", 0) / 1000, llm_t.get("decode_ms", 0) / 1000
            base = t_start - job.created
            segs.append({"name": "prefill", "t0": round(base, 3), "t1": round(base + pre, 3)})
            segs.append({"name": "decode", "t0": round(base + pre, 3), "t1": round(base + pre + dec, 3)})
        for s in steps:
            segs.append({"name": s["skill"], "t0": round(s["t0"] - job.created, 3),
                         "t1": round(s["t1"] - job.created, 3)})
        payload = {"id": job.id, "text": job.text, "source": job.source, "reply": reply,
                   "total_ms": round((t_end - job.created) * 1000), "segments": segs,
                   "cache_n": llm_t.get("cache_n"), "prompt_n": llm_t.get("prompt_n"),
                   "tok_s": llm_t.get("tok_s"), "errors": (res or {}).get("errors", [])}
        self.viz.emit("job.done", payload)
        self.mqtt.publish_event("job.done", {k: payload[k] for k in ("id", "text", "reply", "total_ms")})
        if self.status.get("state") != "ESTOP":
            self.status.set_state("IDLE", f"Done: {reply[:60]}", last_reply=reply)

    # ---------------------------------------------------------------- background
    def _status_ticker(self):
        while not self._stop.wait(config.STATUS_PERIOD_S):
            self.status.update(notify=False, imu=self.hal.imu(),
                               car={"connected": self.hal.connected, "left": self.hal.left,
                                    "right": self.hal.right})
            snap = self.status.snapshot()
            self.mqtt.publish_status(snap)
            self.viz.emit("status", snap)

    def _lego_loop(self):
        lego = self.real_lego
        while not self._stop.is_set():
            if not lego.connected:
                ok = lego.connect()
                if ok:
                    self.performers["lego"] = lego
                    print(f"[lego] connected via {lego.host}:{lego.port}, BLE rtt {lego.latency_ms:.0f} ms",
                          flush=True)
                else:
                    self.performers["lego"] = self.sim_lego
                self.status.update(lego={"connected": ok, "state": "ble" if ok else "sim",
                                         "error": None if ok else lego.error,
                                         "rtt_ms": round(lego.latency_ms, 1) if ok else None})
                self._stop.wait(5 if ok else 15)
                continue
            st = lego.state()
            self.viz.emit("lego.state", st)
            self._stop.wait(1.0)

    def _on_status(self, snap):
        self.mqtt.publish_status(snap)
        self.viz.emit("status", snap)

    def _on_publish(self, topic, payload):
        self.viz.emit("mqtt.out", {"topic": topic, "payload": payload[:300], "t": time.time()})

    # ---------------------------------------------------------------- dashboard
    def _wire_ui(self):
        ui = self.ui

        def send_command(sid, data):
            text = (data or {}).get("text", "")
            d = {"text": text, "source": (data or {}).get("source", "ui"),
                 "reply_to": (data or {}).get("reply_to")}
            return {"result": self.mqtt.handle(d)}

        def get_state(sid, data=None):
            return {**self.viz.snapshot(), "status": self.status.snapshot(),
                    "matrix": self.hal.frame, "models": config.LLM_MODELS,
                    "model": self.llm.model, "think_chars": self.planner.think_chars}

        def estop(sid, data=None):
            return {"ms": self.estop(source="ui")}

        def set_model(sid, data):
            m = (data or {}).get("model")
            if m in config.LLM_MODELS:
                self.llm.model = m
                self.status.update(model=m)
                threading.Thread(target=self._warm, daemon=True).start()
            return {"model": self.llm.model}

        def set_think(sid, data):
            self.planner.think_chars = int(max(0, min(120, int((data or {}).get("chars", 60)))))
            threading.Thread(target=self._warm, daemon=True).start()
            return {"think_chars": self.planner.think_chars}

        ui.on_message("send_command", send_command)
        ui.on_message("get_state", get_state)
        ui.on_message("estop", estop)
        ui.on_message("set_model", set_model)
        ui.on_message("set_think", set_think)
