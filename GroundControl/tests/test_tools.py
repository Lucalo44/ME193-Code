import json
import time

import pytest

import config
from compute import ComputeError, compute
from lookup import Lookup
from memory_store import Memory


def wait_for(cond, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------- compute
@pytest.mark.parametrize("expr,result", [
    ("37 times 89", "3293"), ("square root of 144", "12"), ("2^10", "1024"), ("12 x 7", "84"),
    ("(3 + 4) * 2", "14"), ("10 / 4", "2.5"), ("what is 15 minus 20?", "-5")])
def test_math(expr, result):
    assert compute("math", expr).endswith(f"= {result}.")


@pytest.mark.parametrize("bad", ['__import__("os")', "open('x')", "2 ** 100000", "1/0", "abc"])
def test_math_refuses_non_arithmetic(bad):
    with pytest.raises(ComputeError):
        compute("math", bad)


def test_letters():
    assert compute("count", "strawberry", "e") == "There is 1 'e' in 'strawberry'."
    assert compute("count", "the word 'strawberry'", "e's") == "There is 1 'e' in 'strawberry'."
    assert compute("count", "strawberry", "r").startswith("There are 3")
    assert compute("length", "dance") == "'dance' has 5 letters."
    assert compute("spell", "robot") == "robot: R-O-B-O-T."
    assert compute("reverse", "hello") == "'hello' reversed is 'olleh'."


# ---------------------------------------------------------------- memory
def test_memory_roundtrip(tmp_path):
    m = Memory(tmp_path / "m.json")
    assert "remember" in m.do("save", "my favorite color", "green")
    assert m.do("recall", "favorite color") == "Your favorite color is green."
    assert m.do("recall", "color") == "Your favorite color is green."          # fuzzy key
    assert Memory(tmp_path / "m.json").do("recall", "favorite color").endswith("green.")   # persisted
    assert "forgot" in m.do("forget", "favorite color")
    assert "don't know" in m.do("recall", "favorite color")


def test_memory_caps(tmp_path):
    m = Memory(tmp_path / "m.json")
    m.do("save", "k" * 99, "v" * 999)
    d = json.loads((tmp_path / "m.json").read_text())
    (k, v), = d.items()
    assert len(k) <= 30 and len(v) <= 120


# ---------------------------------------------------------------- lookup
APPS = [{"folder": "boston-weather-display", "name": "Boston Weather Display",
         "description": "Fetches the latest Boston weather", "readme": "", "running": False},
        {"folder": "gaitid", "name": "GaitID", "description": "Wearable activity classifier: still / walk / run / bike",
         "readme": "", "running": False}]
CONSTS = {"LATITUDE": 42.36, "LONGITUDE": -71.06,
          "WEATHER_URL": "https://api.open-meteo.com/v1/forecast?latitude={...}&temperature_unit=fahrenheit"}
WEATHER = {"current": {"temperature_2m": 41.2, "apparent_temperature": 35.9, "weather_code": 3, "wind_speed_10m": 9.4},
           "daily": {"time": ["2026-11-25", "2026-11-26"], "temperature_2m_max": [45.1, 38.0],
                     "temperature_2m_min": [33.0, 29.4], "weather_code": [3, 71],
                     "precipitation_probability_max": [10, 60]}}


def fake_lookup():
    urls = []
    lk = Lookup(fetch=lambda url: (urls.append(url), WEATHER)[1])
    lk._host = lambda op, **kw: {"apps": {"apps": APPS}, "app_config": {"constants": CONSTS}}[op]
    return lk, urls


def test_weather_uses_the_weather_apps_settings():
    lk, urls = fake_lookup()
    t = lk.answer("weather")
    assert t.startswith("It's 41°F in Boston right now and overcast")
    assert "Boston Weather Display app" in t
    assert "Tomorrow: high 38°F, low 29°F, snowing lightly." in t
    assert "latitude=42.36" in urls[0] and "temperature_unit=fahrenheit" in urls[0]
    f = lk.answer("forecast")
    assert "Tomorrow in Boston: high 38°F, low 29°F, snowing lightly, 60% chance of rain" in f


def test_apps_and_app_info():
    lk, _ = fake_lookup()
    assert lk.answer("apps") == "I have 2 App Lab apps: Boston Weather Display, GaitID."
    assert lk.answer("app_info", "gait id").startswith("GaitID: Wearable activity classifier")


def test_time_and_date_are_boston_time():
    lk = Lookup()
    assert lk.answer("time").startswith("It's ") and ("EST" in lk.answer("time") or "EDT" in lk.answer("time"))
    assert lk.answer("date").startswith("Today is ")


def test_lookup_without_host_service_falls_back():
    lk = Lookup(host="127.0.0.1", port=1, fetch=lambda url: WEATHER)
    assert "I couldn't reach my host service, so I used Boston" in lk.answer("weather")


# ---------------------------------------------------------------- end to end through the planner
def test_tools_end_to_end(gc):
    gc.lookup, _ = fake_lookup()
    gc.submit("how many e's are in strawberry?")
    assert wait_for(lambda: (config.TOPIC_REPLY, "There is 1 'e' in 'strawberry'.") in gc.sent)
    gc.submit("remember that my favorite color is green")
    assert wait_for(lambda: any("I'll remember your favorite color: green" in p for _, p in gc.sent))
    gc.submit("what is my favorite color?")
    assert wait_for(lambda: (config.TOPIC_REPLY, "Your favorite color is green.") in gc.sent)
    gc.submit("what's the temperature outside?")
    assert wait_for(lambda: any(p.startswith("It's 41°F in Boston") for _, p in gc.sent))


def test_grounding_guard_replaces_imagined_app_answers():
    from lookup import ground_app_questions
    plan = {"think": "answer", "steps": [{"skill": "answer", "args": {"text": "GaitID analyzes gait abnormalities."}}]}
    new, note = ground_app_questions(plan, "what does the GaitID app do?", ["Boston Weather Display", "GaitID"])
    assert new["steps"] == [{"skill": "lookup", "args": {"what": "app_info", "about": "GaitID"}}] and note
    same, note2 = ground_app_questions(plan, "what is gait analysis?", ["GaitID"])
    assert same is plan and note2 is None
