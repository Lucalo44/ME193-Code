# Phase 0: board survey and spikes

Measured on 2026-11-25 over USB (adb) with the board on home Wi-Fi.

## Board

| | |
|---|---|
| Board | Arduino UNO Q, hostname `Luca`, USB product "UNO Q _ Luca" (serial 1445925855) |
| RAM | **4 GB** (3.6 GiB usable), 3.0 GiB available idle; 1.8 GiB swap |
| CPU | 4 cores (Qualcomm, `CPU part 0x801`) |
| OS | Debian 13 (trixie), kernel 7.0.0 |
| Disk | 9.8 GB root; 1.4 GB free after both models were installed |
| App Lab | `arduino-app-cli` 0.14.1, brick images 0.13.2 |
| App container | Python 3.13.18; paho-mqtt 2.1.0, numpy 2.5.4, httpx, openai, langchain preinstalled |
| Wi-Fi | 192.168.1.58 on the home network (not yet in `ArduinoUNOq/tools/board_secrets.py` HOSTS; adb works regardless) |

## LLM models (llama.cpp runner)

`arduino-app-cli model list` offers two llama.cpp models for the UNO Q (Gemma 4 E2B/E4B are VENTUNO Q only):

| Model | Size | Installed |
|---|---|---|
| `llamacpp:Qwen3.5-0.8B-Q4_0` (default) | 507 MB | yes |
| `llamacpp:gemma-3-1b-it-Q4_0` | 722 MB | yes |

The runner is a separate container (`llamacpp-models-runner:9999` inside the app's network, `127.0.0.1:9999` on the host) in **router mode** (`LLAMA_ARG_MODELS_DIR`, `MODELS_MAX=1`): the request's `model` field picks the model, and switching models unloads the other one. Container memory limit 2500 MB, ctx 16384, 4 threads, flash attention on, reasoning off.

## Spike 1: MQTT over WebSockets (`tools/spikes/spike_mqtt.py`, run inside the app container)

| Path | Result | connect+sub+pub | publish → message |
|---|---|---|---|
| `/mqtt` | PASS | 545–865 ms | 89–99 ms |
| `/` | PASS | 435–439 ms | 80–89 ms |

## Spike 2: llama.cpp features (`tools/spikes/spike_llm.py`, Qwen3.5-0.8B)

| Feature | Result | Notes |
|---|---|---|
| `GET /v1/models` | PASS | both models listed |
| chat completion, cold | PASS | 21.6 s total: prefill 82 tok in 7.2 s (**11.4 tok/s**), decode 40 tok in 9.0 s (**4.33 tok/s**) |
| prompt cache (`timings.cache_n`) | PASS | same prompt again: 78 of 82 tokens reused, prefill 0.54 s |
| `response_format: json_schema` | PASS | valid JSON every time |
| `logprobs` + `top_logprobs: 5` (chat) | PASS | OpenAI-style per-token top-5 |
| streaming SSE | PASS | first token 474 ms with warm cache |
| **stream + json_schema + logprobs together** | PASS | 43 tokens, top-5 on each, final chunk carries `timings` |
| `/completion` `n_probs` + `post_sampling_probs` | PASS | returns `top_probs` with probabilities |
| `/tokenize` `with_pieces` | PASS | `[{id: 900, piece: "Type"}, {id: 20375, piece: " Hi"}, ...]` |
| `/apply-template` | PASS | Qwen chat template (`<|im_start|>`) |

Found later: `top_logprobs` without `logprobs: true` is a 400; `"items": false` in a schema is rejected (tuple length comes from `prefixItems` + `minItems/maxItems` instead). `prefixItems`, `const` (including arrays) and `anyOf` all work.

The top-k probabilities are the model's **raw** distribution (before the grammar mask). At temperature 0, any generated token that isn't the raw top-1 was forced by the grammar. That gives the dashboard's ⚑ marker for free.

## Spike 3: Bridge round trip

`Bridge.call("ping", 41)` → 42 in **8 ms**. PASS. The sketch (frames, wheel PWM, 1 s heartbeat watchdog) compiled and flashed through `arduino-app-cli app start`.

## Spike 4: peripherals

| | Result |
|---|---|
| USB camera | **none attached** (`/dev/video0/1` are the Qualcomm Venus codec, not cameras) |
| USB mic | **none attached** (only the board's own audio codec) |
| IMU | **none on the board**; the LEGO Double Motor has one (yaw/pitch/roll); a Modulino Movement would need Qwiic |
| I2C | `/dev/i2c-0..2` present |
| Bluetooth | `hci0` up, BlueZ active, user `arduino` is in the `bluetooth` group |

## Spike 5: BLE to the LEGO Double Motor (`tools/spikes/spike_ble.py`)

- **The app container cannot do BLE**: no D-Bus socket mounted, bridge network, not privileged. So the LEGO driver runs as a small service on the board **host** (`lego_service/lego_server.py`). The app reaches it over TCP via the Docker bridge gateway (both share one kernel clock, which the duet's timing relies on).
- Host Python lacks `ensurepip` and sudo needs a password, so the venv was bootstrapped with `python3 -m venv --without-pip` + `get-pip.py`: `/home/arduino/gc-lego-venv` with `legoeducation==1.1.1`, `bleak==3.0.2` (same versions as the sumo robot).
- BLE scan from the host: PASS (49 devices visible). **0 LEGO devices**: the Double Motor wasn't available. Connect + latency calibration is still to do: `spike_ble.py --connect`.
- **Decision (2026-11-25):** the LEGO link is assumed OK because the radio, `legoeducation` 1.1.1 and the service all work on the board, and the same package and card worked from the laptop for the sumo robot. Until the motor is connected, the app falls back to `SimLegoPerformer` (dashboard shows "LEGO: simulated"). To verify with the motor: the BLE round trip, the B-arm sign/zero (`GC_LEGO_B_SIGN`, `GC_LEGO_ZERO_A/B`), the IMU units, and the 3° motion threshold.

## Open questions this didn't answer

See the "Questions for Luca" list in the README.
