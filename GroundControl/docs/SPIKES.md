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

**Later the same day:** installed `llamacpp:unsloth/Qwen3-1.7B-GGUF:Q4_0` (1.06 GB) from Hugging Face with `arduino-app-cli model install`, and deleted Gemma 3 1B (`arduino-app-cli model delete`) to free disk: 1022 MB free afterwards. The server picks up new models only after the runner container restarts. Qwen3-1.7B needs `LLAMA_ARG_CTX_SIZE=2048` (see below) and runs at 4.8–5.2 tok/s prefill and 3.0–3.1 tok/s decode, using 948 MiB. It is now the default (80% vs 62% on the held-out eval).

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

### Context size vs. memory cap

With the default `LLAMA_ARG_CTX_SIZE=16384`, Qwen3-1.7B decoded at **0.3 tok/s** while the board's free RAM sank to 104 MB. Its KV cache is about 115 KB/token (28 layers × 8 KV heads × 128 × 2 × fp16), so about 1.8 GB at 16k. With the weights, that exceeds the runner's 2.44 GiB cgroup cap, and the mmapped weights get evicted and re-read from eMMC on every token. Qwen3.5-0.8B (a hybrid model) has a much smaller KV cache and never hit this. Fix: `arduino-app-cli app brick config <app> arduino:llm LLAMA_ARG_CTX_SIZE=2048`, which writes `variables:` under the brick in `app.yaml` (and drops the `model:` line, so keep the repo's copy authoritative).

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

# Part 2 spikes (2026-11-27)

## A1: saving the prompt cache to disk

The 3-minute warm-up was only llama.cpp re-reading the same 867-token prefix, so I wanted to save that state once and load it at boot.

1. **The runner takes the flag.** App Lab's runner starts `llama-server` from `/run-model-router.sh` with only `--models-preset`, but the binary reads `LLAMA_ARG_SLOT_SAVE_PATH` from the environment, and App Lab passes every `variables:` entry in `app.yaml` through, the same way `LLAMA_ARG_CTX_SIZE` already works. So no sidecar server was needed. The files go to `/models/slots`, which is the host's `models/llamacpp/slots` folder, so they survive restarts.
2. **The router passes `/slots` through,** but only with the model named: `?model=` on a GET, `"model"` in the body of a POST. There is exactly one slot (id 0) at ctx 2048.
3. **The numbers.** Saving 904 tokens wrote 103.7 MB in 0.19 s, close to the estimated 112 KB per token. Restoring took 0.12 s from the file cache.
4. **Cold runner.** After `docker restart` of the runner: the router answered in 1.0 s, the restore (including loading the model) took 3.6 s, and the first plan took 14.1 s with 855 tokens reused. That is 18.7 s from restart to a planned command, down from 3–7 minutes. `docker restart` keeps the GGUF in the kernel's file cache, so a real power-on also pays to read 1 GB from eMMC.
5. **In the app.** The first boot finds no file, warms up cold (184 s) and saves. Every boot after that restores: **warm-up done in 4.0 s**, with 866 of 867 tokens reused by the check request.

The file name is a hash of everything that changes the cached state: the GGUF's sha256 (from the host service, cached by size and mtime, since the app container can't read the models folder), the exact prefix token ids, the context size, the llama.cpp build and the runner's `variables:` block. A mismatch just means a cold warm-up and a new save. A stale restore would give silently wrong plans, so nothing is ever restored without a matching key, and a restore only counts if the check request reuses the prefix.

The same file fixes a slowdown I hadn't noticed. Answers and Wikipedia choreography use other prompts, and with one slot they pushed the planner's prefix out, so the command after a question re-read all 867 tokens. Now the prefix is restored in the background right after those calls.

## A0: what the planner's time goes to

[tools/bench/latency.py](../tools/bench/latency.py) plans a fixed 20-command subset against the board and splits every command into prefill, think tokens, step tokens and everything else. The v2 baseline ([bench/2026-11-26_baseline-v2.json](bench/2026-11-26_baseline-v2.json), with a chart next to it):

1. **20/20 correct, mean 20.8 s** (p50 19.9 s, p90 31.0 s). No repair passes.
2. **Prefill is 30%** (6.2 s): the command's 11–31 new tokens at about 3 tokens per second, with 855 tokens reused from the cache every time.
3. **Decode is 70%** (14.5 s): 32 tokens on average at 2.18 tokens per second. Only 2 of those are the think. The other 28 are the steps' JSON, so a shorter plan encoding (A5) is where the time is.

## A0: sharing the CPU with perception

The board has 4 Cortex-A53 cores, and llama.cpp uses all of them. Keyword spotting, pose and beat tracking don't exist yet, so [tools/bench/contention.py](../tools/bench/contention.py) stands in for one with a numpy matrix-multiply loop in the app container, and plans the same 4 commands under each condition.

| Load | llama.cpp on 4 threads | llama.cpp on 3 threads |
|---|---|---|
| none | **2.21 tok/s** (22.5 s per command) | 1.68 tok/s (29.8 s) |
| 1 load, numpy's default (BLAS on every core) | 0.83 tok/s (57.3 s) | – |
| 1 load, 1 thread, unpinned | 1.35 tok/s (36.1 s) | – |
| 1 load, 1 thread, pinned to core 3 | 1.39 tok/s (34.9 s) | **1.53 tok/s** (32.9 s) |
| 1 load at 30% duty, 1 thread, pinned | **1.91 tok/s** (26.1 s) | 1.67 tok/s (30.2 s) |

What this means for every perception model:

1. **One thread, always.** numpy's BLAS quietly spreads one matrix multiply over all four cores, and that alone made the planner 2.7 times slower. Perception code sets `OPENBLAS_NUM_THREADS=1` (and onnxruntime's intra-op threads to 1), and pins itself to core 3 *before* importing numpy, because BLAS starts its threads at import.
2. **Pinning alone barely helps** (1.35 → 1.39). llama.cpp's four threads wait for each other at every step, so one busy core holds up all of them.
3. **Giving llama.cpp only 3 threads is worse** unless a perception model runs flat out. With nothing else running it decodes 24% slower, so it stays on 4.
4. **Duty cycle is what matters.** At 30% duty, the planner keeps 86% of its speed. That's what [python/perception/budget.py](../python/perception/budget.py) enforces: while the LLM is decoding, pose drops to 2 fps and keyword spotting (small and essential) keeps running.

I'll measure each real model against this table when it lands.

## A0: the freeze

On 2026-11-26 the board stopped answering during testing. The dashboard went down, `adb shell` hung, then adb reported the device offline, and only a power cycle brought it back. This is what I found:

1. **It isn't memory.** The journal is persistent, so the boot that froze is still readable (`journalctl -b -1`). It has no OOM kills, no hung-task or RCU-stall warnings and no I/O errors, and systemd kept running its scheduled jobs (man-db, fwupd) until the power cycle at 00:43:21 UTC. The kernel was alive; the app and adb's shell were not.
2. **I couldn't reproduce it.** [tools/bench/freeze_watch.sh](../tools/bench/freeze_watch.sh) restarts the app and its model runner the way a boot does and logs memory, swap, temperature, load and new kernel messages every second. 20 rounds in a row came back clean: ready in 35–37 s every time, at most 1.07 GB used with 2.6 GB available, zram swap unused, under 64 °C, no kernel warnings. Four cold warm-ups during the day's benchmarks (184–236 s) were clean too.
3. **The board already has the safety net I was going to add:** a 1.8 GB zram swap (`/dev/zram0`, priority 100). Its kernel has no `/proc/pressure`, so memory pressure has to be read from `free` instead.
4. **What changed since then.** The 3-minute warm-up, the longest stretch of all four cores at 100%, is down to 4 s after the first boot. A supervisor in the app now checks that the model runner answers every 10 s. After three misses in a row, it shows a restart icon on the matrix, has the host service restart the runner, and restores the prompt cache.

5. **It happened again, and this time the logs pointed at the laptop.** During the held-out eval on 2026-11-27, adb dropped after 67 commands. The board's journal shows the USB host suspending and resuming the link (`FUNCTIONFS_SUSPEND` / `RESUME`) from 04:01, which is what a Mac does as it idles toward sleep, and suspended USB kills adb and its port forwards. The board kept logging normally (cron at 04:17, fwupd at 04:24), and then the journal simply stops, with no error, until a new boot at 04:31. A clean stop like that looks like the power going away, not a crash. The first "freeze" matches the same pattern: normal logs, then nothing.

My best explanation is the laptop, not the board: the Mac sleeping suspends the USB link, and the board may lose power if it is powered from that port. That still needs confirming (where the board gets its power, and whether the Mac slept). The 20 rounds above were app and runner restarts, not power cycles, because I can't power-cycle the board remotely. A true cold-boot loop needs a passwordless `reboot` rule like the existing `poweroff` one. The eval and benchmark tools now keep the Mac awake while they run.
