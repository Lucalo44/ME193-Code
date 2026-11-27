"""Real-world and local facts the model can't know on its own.

    time / date    the board's clock in TIMEZONE (the board itself runs on UTC)
    weather        current conditions, from the SAME source and settings as the
                   installed weather app (Boston Weather Display reads Open-Meteo for
                   Boston). App Lab runs one app at a time, so we don't start that
                   app (it would stop this one); we read its settings from its code
                   via the host service and call the same API.
    forecast       today's and tomorrow's high/low from the same source
    apps           the App Lab apps installed on the board
    app_info       what one app does (its app.yaml description + README), by name

The host service ("apps", "app_config") is needed for anything outside this
app's own folder; without it, weather falls back to Boston defaults and says so.
"""
import datetime
import difflib
import json
import re
import socket
import time
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

import config
from hal.performers import docker_host_gateway

TIMEZONE = "America/New_York"
WEATHER_TTL_S = 300
DEFAULT_PLACE = ("Boston", 42.3601, -71.0589, "fahrenheit")
WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy",
       51: "drizzling lightly", 53: "drizzling", 55: "drizzling heavily", 56: "freezing drizzle",
       57: "freezing drizzle", 61: "raining lightly", 63: "raining", 65: "raining heavily",
       66: "freezing rain", 67: "freezing rain", 71: "snowing lightly", 73: "snowing", 75: "snowing heavily",
       77: "snow grains", 80: "rain showers", 81: "rain showers", 82: "heavy rain showers",
       85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms", 96: "thunderstorms with hail",
       99: "thunderstorms with hail"}
WHAT = ["time", "date", "weather", "forecast", "apps", "app_info"]


def ground_app_questions(plan, text, names):
    """Grounding guard. If the command names one of my installed apps and the model
    planned to `answer` from its own head, look the app up instead. A 1.7B model
    confidently invents what "GaitID" does. Returns (plan, note or None)."""
    if not plan or not names:
        return plan, None
    low = text.lower()
    hit = next((n for n in sorted(names, key=len, reverse=True) if n.lower() in low), None)
    if hit is None:
        return plan, None
    steps = plan.get("steps") or []
    if not any(st.get("skill") == "answer" for st in steps):
        return plan, None
    new = [{"skill": "lookup", "args": {"what": "app_info", "about": hit}} if st.get("skill") == "answer" else st
           for st in steps]
    return {**plan, "steps": new, "grounded": True}, f"'{hit}' is one of my apps, so I looked it up instead of guessing"


def describe_apps_in_catalog(lookup, patch=True):
    """Name the installed apps in lookup's catalog line (once, at startup; stays
    cache-friendly). Without it the model 'knows' what GaitID is and answers from
    imagination instead of looking it up. Returns the names (empty if unreachable)."""
    from skills.registry import REGISTRY
    names = lookup.app_names()
    if patch and names and "lookup" in REGISTRY:
        base = REGISTRY["lookup"].desc.split("; installed apps:")[0]
        REGISTRY["lookup"].desc = base + "; installed apps: " + ", ".join(names)
    return names


class Lookup:
    def __init__(self, host=None, port=None, fetch=None):
        self.host_addr = host or config.LEGO_HOST or docker_host_gateway()
        self.port = port or config.LEGO_PORT
        self.fetch = fetch or self._http_json          # injectable for tests
        self._weather = (0.0, None, None)

    # ---------------------------------------------------------------- plumbing
    def _host(self, op, timeout=8, **kw):
        with socket.create_connection((self.host_addr, self.port), timeout=3) as s:
            s.settimeout(timeout)
            s.sendall((json.dumps({"op": op, "rid": 1, **kw}) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        return json.loads(buf)

    def model_hash(self, model):
        """sha256 of the model's GGUF (host service; ~15 s the first time, then cached)."""
        try:
            return self._host("model_hash", timeout=60, model=model).get("sha256")
        except (OSError, ValueError):
            return None

    @staticmethod
    def _http_json(url):
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())

    def _apps(self):
        try:
            return self._host("apps").get("apps", [])
        except (OSError, ValueError):
            return None

    # ---------------------------------------------------------------- answers
    def answer(self, what, about=""):
        fn = getattr(self, f"_q_{what}", None)
        if fn is None:
            return f"I can't look up '{what}'."
        try:
            return fn((about or "").strip())
        except Exception as e:  # noqa: BLE001  a dead network shouldn't kill the job
            return f"I tried to look up the {what.replace('_', ' ')} but it failed ({e})."

    def _now(self):
        return datetime.datetime.now(ZoneInfo(TIMEZONE))

    def _q_time(self, about):
        n = self._now()
        return f"It's {n.strftime('%-I:%M %p')} ({n.tzname()}, Boston time)."

    def _q_date(self, about):
        n = self._now()
        return f"Today is {n.strftime('%A, %B %-d, %Y')}."

    def _weather_source(self):
        """(place, lat, lon, unit, app_name) from the installed weather app's own code."""
        apps = self._apps()
        if apps:
            for a in apps:
                if "weather" in (a["name"] + " " + a["description"]).lower():
                    try:
                        c = self._host("app_config", folder=a["folder"]).get("constants", {})
                    except (OSError, ValueError):
                        c = {}
                    lat, lon = c.get("LATITUDE"), c.get("LONGITUDE")
                    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                        url = next((v for v in c.values() if isinstance(v, str) and "temperature_unit=" in v), "")
                        unit = "fahrenheit" if "temperature_unit=fahrenheit" in url else "celsius"
                        place = re.split(r"\s+weather", a["name"], flags=re.I)[0].strip() or "here"
                        return place, lat, lon, unit, a["name"]
        place, lat, lon, unit = DEFAULT_PLACE
        return place, lat, lon, unit, None

    def _get_weather(self):
        t, data, src = self._weather
        if data is not None and time.time() - t < WEATHER_TTL_S:
            return data, src
        place, lat, lon, unit, app = self._weather_source()
        q = urllib.parse.urlencode({
            "latitude": lat, "longitude": lon, "temperature_unit": unit, "wind_speed_unit": "mph",
            "timezone": TIMEZONE, "forecast_days": 2,
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max"})
        data = self.fetch(f"https://api.open-meteo.com/v1/forecast?{q}")
        src = {"place": place, "unit": "°F" if unit == "fahrenheit" else "°C", "app": app}
        self._weather = (time.time(), data, src)
        return data, src

    def _source_note(self, src):
        if src["app"]:
            return f"(from Open-Meteo, the same source and settings as your {src['app']} app)"
        return "(from Open-Meteo; I couldn't reach my host service, so I used Boston)"

    def _q_weather(self, about):
        d, src = self._get_weather()
        c = d["current"]
        sky = WMO.get(c.get("weather_code"), "")
        u = src["unit"]
        dl = d.get("daily") or {}
        tomorrow = ""
        if len(dl.get("time", [])) > 1:
            tsky = WMO.get(dl["weather_code"][1], "")
            tomorrow = (f" Tomorrow: high {round(dl['temperature_2m_max'][1])}{u}, "
                        f"low {round(dl['temperature_2m_min'][1])}{u}" + (f", {tsky}" if tsky else "") + ".")
        return (f"It's {round(c['temperature_2m'])}{u} in {src['place']} right now"
                + (f" and {sky}" if sky else "")
                + f", feels like {round(c['apparent_temperature'])}{u}, wind {round(c['wind_speed_10m'])} mph "
                + self._source_note(src) + "." + tomorrow)

    def _q_forecast(self, about):
        d, src = self._get_weather()
        dl = d["daily"]
        u = src["unit"]
        parts = []
        for i, label in enumerate(["Today", "Tomorrow"]):
            if i < len(dl["time"]):
                sky = WMO.get(dl["weather_code"][i], "")
                rain = dl.get("precipitation_probability_max", [None, None])[i]
                parts.append(f"{label} in {src['place']}: high {round(dl['temperature_2m_max'][i])}{u}, "
                             f"low {round(dl['temperature_2m_min'][i])}{u}"
                             + (f", {sky}" if sky else "") + (f", {rain}% chance of rain" if rain is not None else ""))
        return ". ".join(parts) + " " + self._source_note(src) + "."

    def app_names(self):
        """Installed app names, for the catalog line (so 'the GaitID app' reads as an app)."""
        apps = self._apps()
        return [a["name"] for a in apps if a["folder"] != "ground-control"] if apps else []

    def _q_apps(self, about):
        apps = self._apps()
        if apps is None:
            return "I couldn't reach my host service to list my apps."
        names = [a["name"] + (" (running)" if a["running"] else "") for a in apps]
        return f"I have {len(apps)} App Lab apps: " + ", ".join(names) + "."

    def _q_app_info(self, about):
        apps = self._apps()
        if apps is None:
            return "I couldn't reach my host service to read my apps."
        if not about:
            return self._q_apps(about)
        key = about.lower().replace(" app", "").strip()
        names = {a["name"].lower(): a for a in apps}
        names.update({a["folder"].lower(): a for a in apps})
        hit = names.get(key)
        if hit is None:
            close = difflib.get_close_matches(key, list(names), n=1, cutoff=0.4)
            hit = names[close[0]] if close else next((a for n, a in names.items() if key in n or n in key), None)
        if hit is None:
            return f"I don't have an app called '{about}'. " + self._q_apps("")
        desc = (hit["description"] or hit["readme"] or "It has no description").rstrip(". ") + "."
        extra = f" {hit['readme']}" if hit["description"] and hit["readme"] else ""
        state = "It's running right now." if hit["running"] else "It isn't running right now."
        return f"{hit['name']}: {desc}{extra[:300]} {state}"
