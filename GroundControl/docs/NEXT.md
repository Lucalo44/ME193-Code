# Next session: wider skill set

Agreed on 2026-11-25: build all three skills below. Same pattern as `self_report`: the model picks an option and copies inputs, and code produces the facts, so nothing is hallucinated. Grouping keeps the catalog short (each new skill costs a 1.7B model some accuracy).

## 1. `lookup(what, about)`

`what`: `time | date | weather | forecast | apps | app_info`

- **time/date**: the board's clock (America/New_York). Today the model just guesses the time.
- **weather/forecast**: reuse the *Boston Weather Display* app's own data source rather than running the app. App Lab runs one app at a time, so starting it would stop Ground Control. The app (`/home/arduino/ArduinoApps/boston-weather-display/python/main.py`) calls Open-Meteo with `LATITUDE = 42.3601`, `LONGITUDE = -71.0589`, `temperature_unit=fahrenheit`, `timezone=America/New_York`. The container can't see other app folders, so add a host-service op (`lego_service/lego_server.py`, e.g. `app_config`) that reads those constants from the app's file. Ground Control then calls the same URL (verified reachable from the board). If the app's city changes, the robot follows.
- **apps / app_info**: installed App Lab apps with their `app.yaml` descriptions (and README first paragraph). Host service already lists app names in `sysinfo`; extend it with descriptions.
- Reply text is phrased in code, e.g. "It's 41 °F in Boston right now (from the Boston Weather Display app's source, Open-Meteo)."

## 2. `compute(op, text, of)`

`op`: `math | count | length | spell | reverse`

- Fixes the tokenization failure seen on the dashboard: "how many e's in strawberry?" got "3", because the model sees `str|aw|berry`, not letters. The correct answer is 1.
- `math`: safe `ast`-based evaluator (+ − × ÷ ** %, parentheses, sqrt/round). No `eval`.
- `count`: letter or word occurrences in `text` (`of` = what to count).

## 3. `memory(op, key, value)`

`op`: `save | recall | forget`

- Small persistent key-value store in the app folder (`/app/memory.json`, survives restarts). "remember my name is Luca" → save name=Luca; "what's my name?" → recall name.
- Cap sizes (keys ≤ 30 chars, values ≤ 120, ≤ 100 entries). It's public-broker input, so treat it as untrusted text and never execute it.

## Also

- Latency chart: the time-axis labels smear together when one job is very long (seen with a 227 s job). Pick the tick step from the axis width, not just tmax.
- Prompt: add one few-shot example covering a new skill (e.g. a weather question). Keep the examples balanced (see EVAL.md history: examples get copied).
- Eval: add held-out categories `lookup`, `compute`, `memory` (4–5 commands each, never tuned on), then re-run on Qwen3-1.7B and update EVAL.md/README.
- Tests: each skill against the sim HAL; the weather call is mocked.

## State at handoff

- Board runs the latest commit (Qwen3-1.7B, `answer` + `self_report`, LLM lock, ctx 4096, https dashboard).
- The held-out re-run with the `self` category crashed after 26 of 48 commands (21/26 correct so far; traceback not captured, probably an LLM timeout while dashboard commands were sharing the model). First make `run_eval.py` catch per-command exceptions (record as `did_nothing` with the error) and save partial results, then re-run: `venv/bin/python tools/eval/run_eval.py --tag self` (about 25 min, needs `adb forward tcp:9999 tcp:9999`; don't use the dashboard meanwhile).
