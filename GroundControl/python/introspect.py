"""Self-knowledge for the planner: what board am I, what's plugged in, what's running.

The LLM never states these facts itself (a 1.7B model would make them up). It
picks a topic with the self_report skill, and this module gathers real facts
and phrases them:

  board      what the hardware is
  pins       what the sketch drives + a live probe of the free header pins + I2C scan
  devices    USB / camera / audio / Bluetooth / I2C / the LEGO bot
  software   the app, the model, the other App Lab apps and services
  health     RAM, disk, CPU load, my own temperature, uptime
  network    IP address, MQTT broker and topics, dashboard URL
  all        a short summary of everything

Host-level facts come from the host service (lego_service/lego_server.py, op
"sysinfo"), because the app container has its own OS image and only sees its own
app folder. Results are cached for HOST_TTL_S seconds; pins and I2C are probed
live each time they're asked about.
"""
import json
import os
import socket
import time

import config
from hal.performers import docker_host_gateway

HOST_TTL_S = 30

# What sketch/sketch.ino does with the header (configured, not sensed).
CONFIGURED_PINS = [
    ("D5/D6", "left wheel motor (PWM to the motor driver)"),
    ("D9/D10", "right wheel motor (PWM to the motor driver)"),
]
I2C_BUS_NAMES = {"Wire": "header SDA/SCL (D20/D21)", "Wire1": "Qwiic connector", "Wire2": "A4/A5"}
# Common addresses. "likely", because different parts can share an address.
I2C_KNOWN = {
    "1e": "Modulino Buzzer", "29": "a distance sensor (e.g. Modulino Distance, VL53L4CD)",
    "3c": "an SSD1306 OLED display", "44": "Modulino Thermo (HS3003)",
    "48": "an ADC or temperature sensor (ADS1115/TMP102)", "68": "an IMU or RTC (MPU6050/DS3231)",
    "6a": "Modulino Movement (LSM6DSOX IMU)", "6b": "Modulino Movement (LSM6DSOX IMU)",
    "6c": "Modulino Pixels", "76": "a BME280/BMP280 pressure sensor", "77": "a BME280/BMP280 pressure sensor",
    "7c": "Modulino Buttons",
}


def _human_s(s):
    s = int(s)
    if s < 120:
        return f"{s} s"
    if s < 7200:
        return f"{s // 60} min"
    return f"{s // 3600} h {s % 3600 // 60} min"


def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


class SelfModel:
    def __init__(self, hal, status, host=None, port=None, t0=None):
        self.hal = hal
        self.status = status
        self.host_addr = host or config.LEGO_HOST or docker_host_gateway()
        self.port = port or config.LEGO_PORT
        self.t0 = t0 or time.time()
        self._host_cache = (0.0, None)

    # ---------------------------------------------------------------- facts
    def host(self):
        t, info = self._host_cache
        if info is not None and time.time() - t < HOST_TTL_S:
            return info
        try:
            with socket.create_connection((self.host_addr, self.port), timeout=3) as s:
                s.settimeout(8)
                s.sendall(b'{"op":"sysinfo","rid":1}\n')
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
            info = json.loads(buf)
        except (OSError, ValueError):
            info = self._local_fallback()
        self._host_cache = (time.time(), info)
        return info

    def _local_fallback(self):
        """What the container can see on its own if the host service is down."""
        info = {"from": "container"}
        try:
            info["model"] = open("/sys/firmware/devicetree/base/model").read().strip("\x00 \n")
        except OSError:
            info["model"] = None
        info["cpus"] = os.cpu_count()
        try:
            mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
            info["mem_total_mb"] = mem["MemTotal"] // 1024
            info["mem_avail_mb"] = mem["MemAvailable"] // 1024
            info["uptime_s"] = int(float(open("/proc/uptime").read().split()[0]))
            info["load"] = open("/proc/loadavg").read().split()[:3]
        except (OSError, KeyError, ValueError):
            pass
        return info

    # ---------------------------------------------------------------- topics
    def report(self, topic):
        fn = getattr(self, f"_t_{topic}", None)
        if fn is None:
            return f"I don't know how to describe '{topic}'."
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  a broken probe shouldn't kill the job
            return f"I tried to check my {topic} but something failed ({e})."

    @staticmethod
    def _board_name(h):
        m = (h.get("model") or "").lower()
        if "uno" in m:
            return "an Arduino UNO Q"
        return h.get("model") or None

    def _t_board(self):
        h = self.host()
        board = self._board_name(h)
        if board is None:
            return "I couldn't identify my board (my host service isn't answering)."
        soc = h.get("soc") or []
        soc = soc[0].replace("qcom,", "").upper() if soc else None
        ram = f" and {h['mem_total_mb'] / 1024:.1f} GB of RAM" if h.get("mem_total_mb") else ""
        parts = [f"I'm {board}" + (f" named {h['hostname']}" if h.get("hostname") else "") + "."]
        parts.append(f"My Linux side is {'a Qualcomm ' + soc if soc else 'a Qualcomm processor'} with "
                     f"{h.get('cpus', '?')} Arm cores{ram}"
                     + (f", running {h['os']} (kernel {h['kernel']})" if h.get("os") else "") + ".")
        parts.append("An STM32U585 microcontroller handles the real-time side: the 8x13 LED matrix, "
                     "the header pins and the motors, and talks to Linux over the Bridge.")
        return " ".join(parts)

    def _pins_text(self):
        cfg = "; ".join(f"{p} drive the {what}" for p, what in CONFIGURED_PINS)
        out = [f"Configured in my sketch: {cfg}. Those are outputs, so I can't sense what's attached to them."]
        probe = self.hal.probe_pins()
        if probe is None:
            out.append("I couldn't probe the other pins.")
        else:
            driven = [f"{p} (held {'high' if v == 'H' else 'low'})" for p, v in probe.items() if v in "HL"]
            odd = [p for p, v in probe.items() if v == "?"]
            floating = [p for p, v in probe.items() if v == "f"]
            if driven:
                out.append(f"Probing the free pins just now, something is driving {_join(driven)}.")
            else:
                out.append("Probing the free pins just now, none of them is being driven by anything.")
            if floating:
                out.append(f"{len(floating)} pins are floating ({_join(floating)}): nothing attached, "
                           "or something passive like a switch without a pull resistor.")
            if odd:
                out.append(f"{_join(odd)} gave inconsistent readings.")
        out.append(self._i2c_text())
        return " ".join(out)

    def _i2c_text(self):
        scan = self.hal.i2c_scan()
        if scan is None:
            return "I couldn't scan the I2C buses."
        found = []
        for bus, addrs in scan.items():
            for a in addrs:
                what = I2C_KNOWN.get(a.lower(), "an unknown device")
                found.append(f"0x{a} on the {I2C_BUS_NAMES.get(bus, bus)} (likely {what})"
                             if a.lower() in I2C_KNOWN else f"{what} at 0x{a} on the {I2C_BUS_NAMES.get(bus, bus)}")
        if not found:
            return "Nothing answers on any of my three I2C buses (Qwiic, header SDA/SCL, A4/A5)."
        return f"On I2C I found {_join(found)}."

    def _t_pins(self):
        return self._pins_text()

    def _t_devices(self):
        h = self.host()
        out = []
        usb = h.get("usb") or []
        out.append(f"USB: {_join(usb)}." if usb else "Nothing is plugged into my USB port right now.")
        cams = [v for v in (h.get("video") or []) if "venus" not in v.lower()]
        out.append(f"Camera: {_join(cams)}." if cams else "No camera attached (the video devices I have are the "
                   "built-in video codec).")
        snd = h.get("sound_cards") or []
        out.append(f"Audio: {_join(snd)}." if snd else "")
        if h.get("bluetooth"):
            lego = self.status.get("lego") or {}
            out.append("Bluetooth is on" + (", and the LEGO Double Motor is connected."
                                            if lego.get("connected") else
                                            ", but the LEGO Double Motor isn't connected, so the LEGO bot is simulated."))
        out.append(self._i2c_text())
        return " ".join(x for x in out if x)

    def _t_software(self):
        h = self.host()
        s = self.status.snapshot()
        model = s.get("model", "?").split("/")[-1]
        out = [f"I'm running the App Lab app Ground Control, up {_human_s(time.time() - self.t0)}. "
               f"My planner is {model} on llama.cpp"
               + (f" at {s['llm_tok_s']} tokens/s" if s.get("llm_tok_s") else "") + "."]
        others = [m.split("/")[-1] for m in (h.get("models") or []) if m.split("/")[-1] != model]
        if others:
            out.append(f"Also installed: {_join(others)}.")
        apps = [a for a in (h.get("apps_installed") or []) if a != "Ground Control"]
        if apps:
            out.append(f"Other App Lab apps on me (not running): {_join(apps)}.")
        out.append("A host service runs beside the app for Bluetooth and system info.")
        out.append(f"I'm {'connected to' if s.get('mqtt') else 'not connected to'} the MQTT broker "
                   f"{config.BROKER_HOST} and currently {s.get('state', '?').lower()}.")
        if h.get("app_cli"):
            out.append(f"({h['app_cli']}.)")
        return " ".join(out)

    def _t_health(self):
        h = self.host()
        out = []
        if h.get("mem_total_mb"):
            out.append(f"{h['mem_avail_mb'] / 1024:.1f} of {h['mem_total_mb'] / 1024:.1f} GB RAM free")
        if h.get("disk_total_mb"):
            out.append(f"{h['disk_free_mb'] / 1024:.1f} of {h['disk_total_mb'] / 1024:.1f} GB disk free")
        if h.get("load"):
            out.append(f"CPU load {h['load'][0]} (1 min) on {h.get('cpus', '?')} cores")
        if h.get("temp_c") is not None:
            out.append(f"{h['temp_c']} °C")
        if h.get("uptime_s"):
            out.append(f"up {_human_s(h['uptime_s'])}")
        return ("Right now: " + _join(out) + ".") if out else "I couldn't read my resource usage."

    def _t_network(self):
        h = self.host()
        ips = h.get("ips") or []
        out = [f"My IP address is {_join(ips)}" + (" on Wi-Fi" if ips else "") + "." if ips
               else "I couldn't read my IP address."]
        if ips:
            out.append(f"My dashboard is at http{'s' if config.WEB_TLS else ''}://{ips[0]}:7000.")
        out.append(f"I take commands on {config.TOPIC_CMD} via {config.BROKER_HOST} (MQTT over WebSockets) "
                   f"and reply on {config.TOPIC_REPLY}.")
        return " ".join(out)

    def _t_all(self):
        h = self.host()
        s = self.status.snapshot()
        ram = f", {h['mem_total_mb'] / 1024:.1f} GB RAM" if h.get("mem_total_mb") else ""
        model = s.get("model", "?").split("/")[-1]
        scan = self.hal.i2c_scan() or {}
        n_i2c = sum(len(v) for v in scan.values())
        lego = "connected" if (s.get("lego") or {}).get("connected") else "simulated"
        board = self._board_name(h) or "a board I couldn't identify"
        i2c = "no I2C devices answer" if n_i2c == 0 else ("1 I2C device answers" if n_i2c == 1
                                                          else f"{n_i2c} I2C devices answer")
        return (f"I'm {board} ({h.get('cpus', '?')} cores{ram}) running Ground Control with "
                f"{model}. My sketch drives two wheel motors on D5/D6 and D9/D10, "
                f"{i2c} on my buses, the LEGO bot is "
                f"{lego}, and I'm {s.get('state', '?').lower()}. Ask about my board, pins, devices, software, "
                "health or network for details.")


    # ---------------------------------------------------------------- compact facts
    # Short "key: value" facts for the answer composer (the LLM writes the reply
    # to the exact question from these). Compact on purpose: every token is
    # prefill time on the board.
    # Words that mean a question needs a topic's facts, whatever topic the model picked.
    KEYWORDS = {
        "health": ("disk", "storage", "space", "ram", "memory", "temperature", "hot", "warm", "cool", "load", "uptime", "how long"),
        "pins": ("pin", "plugged", "attached", "wired", "header", "gpio", "i2c", "qwiic", "connected to"),
        "network": ("ip", "address", "network", "wifi", "wi-fi", "url", "dashboard", "broker", "mqtt", "topic"),
        "software": ("app", "running", "model", "program", "software", "version", "llm"),
        "board": ("board", "processor", "cpu", "chip", "microcontroller", "made of", "what are you"),
        "devices": ("usb", "camera", "microphone", "audio", "speaker", "bluetooth", "lego", "device"),
    }

    def facts(self, topic, question=""):
        """Facts for the chosen topic, plus any topic the question's words point to
        (a 1.7B model sometimes picks a near-miss topic, e.g. 'devices' for disk space)."""
        q = f" {question.lower()} "
        topics = [topic] + [t for t, words in self.KEYWORDS.items()
                            if t != topic and any(w in q for w in words)][:2]
        out = []
        for t in topics:
            try:
                out.append(getattr(self, f"_f_{t}")())
            except Exception as e:  # noqa: BLE001
                out.append(f"error reading {t}: {e}")
        return " ".join(out)

    def _f_board(self):
        h = self.host()
        soc = (h.get("soc") or ["?"])[0].replace("qcom,", "Qualcomm ").replace("qrb", "QRB")
        return (f"board: {self._board_name(h) or 'unknown'} (hostname {h.get('hostname', '?')}); "
                f"main processor (runs Linux, Python and this LLM): {soc}, {h.get('cpus', '?')} Arm cores, "
                f"{h.get('mem_total_mb', 0) / 1024:.1f} GB RAM, {h.get('os', '?')}; "
                "second chip, a real-time microcontroller: STM32U585 (runs the 8x13 LED matrix, header pins, motors)")

    def _f_pins(self):
        probe = self.hal.probe_pins() or {}
        floating = [p for p, v in probe.items() if v == "f"]
        driven = [f"{p} held {'high' if v == 'H' else 'low'}" for p, v in probe.items() if v in "HL"]
        scan = self.hal.i2c_scan() or {}
        i2c = [f"0x{a} on {I2C_BUS_NAMES.get(b, b)} (likely {I2C_KNOWN.get(a.lower(), 'unknown device')})"
               for b, addrs in scan.items() for a in addrs]
        summary = (f"Something is plugged into {', '.join(d.split()[0] for d in driven)}. " if driven
                   else f"Nothing is plugged into any of my {len(probe)} free pins. ")
        summary += ("An I2C device is connected. " if i2c else "No I2C device is connected. ")
        return (summary + f"open pins (nothing attached): {len(floating)} of the {len(probe)} free header pins I just probed "
                f"({', '.join(floating) or 'none'}); pins with something attached and driving them: "
                f"{len(driven)} ({', '.join(driven) or 'none'}); "
                f"motor pins (outputs, cannot be sensed): D5, D6 left wheel; D9, D10 right wheel. "
                f"I2C devices: {', '.join(i2c) or 'none on Qwiic, header SDA/SCL or A4/A5'}. "
                "Not probed: D0/D1 (serial), D20/D21 (I2C).")

    def _f_devices(self):
        h = self.host()
        lego = (self.status.get("lego") or {}).get("connected")
        cams = [v for v in (h.get("video") or []) if "venus" not in v.lower()]
        scan = self.hal.i2c_scan() or {}
        n = sum(len(v) for v in scan.values())
        return (f"USB devices: {', '.join(h.get('usb') or []) or 'none'}; camera: {', '.join(cams) or 'none'}; "
                f"audio: {', '.join(h.get('sound_cards') or []) or 'none'}; bluetooth: {'on' if h.get('bluetooth') else 'off'}; "
                f"LEGO Double Motor: {'connected' if lego else 'not connected (simulated)'}; I2C devices: {n}")

    def _f_software(self):
        h = self.host()
        s = self.status.snapshot()
        model = s.get("model", "?").split("/")[-1]
        apps = [a for a in (h.get("apps_installed") or []) if a != "Ground Control"]
        others = [m.split("/")[-1] for m in (h.get("models") or []) if m.split("/")[-1] != model]
        return (f"running app: Ground Control (up {_human_s(time.time() - self.t0)}); planner model: {model} on llama.cpp"
                + (f" ({s['llm_tok_s']} tokens/s)" if s.get("llm_tok_s") else "")
                + f"; other models installed: {', '.join(others) or 'none'}; other App Lab apps installed, not running: "
                f"{', '.join(apps) or 'none'}; MQTT: {'connected' if s.get('mqtt') else 'disconnected'} ({config.BROKER_HOST}); "
                f"current state: {s.get('state', '?').lower()}")

    def _f_health(self):
        h = self.host()
        return (f"RAM free: {h.get('mem_avail_mb', 0) / 1024:.1f} of {h.get('mem_total_mb', 0) / 1024:.1f} GB; "
                f"disk free: {h.get('disk_free_mb', 0) / 1024:.1f} of {h.get('disk_total_mb', 0) / 1024:.1f} GB; "
                f"CPU load (1 min): {(h.get('load') or ['?'])[0]} on {h.get('cpus', '?')} cores; "
                f"my own processor temperature (inside me, not the weather): {h.get('temp_c', '?')} °C; "
                f"uptime: {_human_s(h.get('uptime_s', 0))}")

    def _f_network(self):
        h = self.host()
        ips = h.get("ips") or []
        return (f"IP address: {', '.join(ips) or 'unknown'} (Wi-Fi); dashboard: "
                f"http{'s' if config.WEB_TLS else ''}://{ips[0] if ips else '?'}:7000; "
                f"commands topic: {config.TOPIC_CMD}; replies topic: {config.TOPIC_REPLY}; broker: {config.BROKER_HOST}")

    def _f_all(self):
        return self._f_board() + ". " + self._f_software() + ". " + self._f_devices()


TOPICS = ["board", "pins", "devices", "software", "health", "network", "all"]
