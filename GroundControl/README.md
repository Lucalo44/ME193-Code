# Ground Control

This is my final project for ME193, and the brief was a lot. The UNO Q had to take commands over MQTT, run them through an LLM that lives on the board itself, do what was asked, report back, and somehow show off neural nets, motors and sensors along the way. I decided the most honest way to show off a 0.8-billion-parameter model on a 4 GB board was to make it choreograph a dance between two robots, and to put its thinking on a projector while it does it.

The hardware is an Arduino UNO Q (the 4 GB version), which is really two computers: a Qualcomm chip running Debian for Python and the LLM, and an STM32 for the LED matrix and the motors. The partner robot is a LEGO Education Double Motor over Bluetooth. The planner runs on llama.cpp through App Lab's LLM Brick. I started with the default model, Qwen3.5-0.8B at 4-bit, which reads a prompt at about 11 tokens per second and writes at about 4.3. It ended up on Qwen3-1.7B, which writes at about 3. Every design choice below comes back to that number.

## What it does

Commands arrive on `ME193/q/luca/cmd`, typed into the dashboard, or spoken into its mic button. From simplest to most complex:

1. **Status.** `/status` (or the retained `ME193/q/luca/status` topic) answers in well under a second, even while the LLM is mid-sentence, because it never touches the LLM.
2. **Stop.** `stop` halts both robots on the MQTT thread in milliseconds. If the Python side dies, the STM32's own watchdog stops the wheels and draws an X within 1 s.
3. **Planned commands.** Everything else goes to the planner, which writes one JSON plan per command:
   1. General questions: "what is 2+2?" gets an answer in words on the reply topic, and short answers scroll on the LED matrix.
   2. Questions about itself: "what board are you on?", "is anything plugged into your pins?", "what's running on you?" get real, measured answers, not guesses.
   3. Things it has to look up or work out: the time, the outside temperature (from the weather app's own source), what its other apps do, exact math, letters in a word, and anything you ask it to remember.
   4. Single actions: "show a heart", "drive backwards for 3 seconds".
   5. Sequences with replies: "Type Hi on the LED matrix and then send back on MQTT on topic luca message 'I am done'".
   6. Information requests: "what is your yaw? reply on luca". The plan writes `{imu.yaw}` and the executor fills in the real value when the message goes out.
   7. Duets: "do a 4 bar dance at 100 bpm, the lego bot mirrors you, then tell me 'bow complete' on luca". The model writes a short dance score, and deterministic code performs it on both robots against one clock.

## How I built it

### Finding out what the board could actually do

I didn't trust anything I hadn't measured, so the first step was a survey over USB ([docs/SPIKES.md](docs/SPIKES.md)). The good news came all at once. The board's llama.cpp build streams tokens, constrains output to a JSON schema, returns the top-5 alternatives for every token, tokenizes with pieces, and reports how much of each prompt it reused from its KV cache. All of that works in the same request. MQTT over WebSockets to test.mosquitto.org worked from inside the app container on the first try, and the Bridge round trip to the STM32 took 8 ms.

The bad news was Bluetooth. App Lab runs the Python app in a Docker container with no D-Bus socket, so `bleak` can't see the radio. I got around this by running the LEGO driver as its own small service on the board's host ([lego_service/lego_server.py](lego_service/lego_server.py)), which the app reaches over the Docker bridge. Getting it installed was its own small saga (no `ensurepip`, no passwordless sudo), so the venv was bootstrapped by hand with `get-pip.py`.

### Plumbing first, LLM second

Before any model was involved, I built the parts that have to be fast and correct. [python/mqtt_io.py](python/mqtt_io.py) answers `/status` and `stop` inline and rate-limits everything else. Everything on the public broker is treated as untrusted: commands are capped at 300 characters, the queue holds 5 jobs, and a plan can only publish to `luca` or `ME193/q/luca/*`, never to anything ending in `/cmd`. [python/skills/registry.py](python/skills/registry.py) is the single source of truth for skills. One decorator gives each skill its line in the prompt, its branch of the JSON grammar, and its argument clamps. [python/executor.py](python/executor.py) runs plans with per-step timeouts and fills in placeholders. If a plan never sends anything, it still replies "done", because the assignment asked for that.

The sketch ([sketch/sketch.ino](sketch/sketch.ino)) is deliberately dumb. It paints whatever 8x13 frame Python sends, sets the wheel PWM, and keeps a heartbeat watchdog. Python renders every frame itself with a hand-drawn 5x7 font ([python/skills/font.py](python/skills/font.py)), so the dashboard can mirror the matrix exactly.

### Getting a tiny model to plan

This was not a major success initially. My first prompt got the KV cache right: 590 of 624 tokens were reused, which cut the prompt read from 56 s to under 4 s. But the plans themselves were poor. Qwen would type "Hi" and then simply stop, even when its own "thought" said the next step was to publish. The dashboard made the reason obvious. After `{"text":"Hi"}}` it closed the step list with 99% confidence and gave "add another step" 0.01%.

I tried four things, and the eval harness ([tools/eval/run_eval.py](tools/eval/run_eval.py)) kept me honest about each:

1. **Few-shot examples as real chat turns** instead of pasted into the system prompt. This was the biggest single jump.
2. **A short `think` that names the skills**, in order, before the steps. It's a cheap scratchpad, and it doubles as the speech bubble on the dashboard.
3. **A repair pass.** If the steps don't match the skills the model named, the planner runs once more with the think pinned and the grammar forcing exactly that sequence (a fixed-length `prefixItems` tuple). The model states its intent, and the grammar holds it to it. This is what rescues the assignment example on the board: the first pass drops the publish, and the second puts it back.
4. **Placeholders that read sensors on demand**, so `{imu.yaw}` in a message is always a live value.

I also tried a cleverer version where the model first copies the command's clauses into `parts` and then writes one step per part. It looked great on the dev set (70%) and scored 51% on the held-out set. Having just copied "turn left 90 degrees" into a part, the model would happily put that exact text on the LED matrix instead of turning.

The part I didn't expect was how fragile all of this is. Every prompt tweak fixed some commands and broke others, and examples got copied far too literally. One turn example fixed every turn command and broke every reply command. In the end I tuned only on a separate dev set and ran the held-out set once per configuration. The numbers are below.

### Watching it think

The dashboard ([assets/app.js](assets/app.js)) is the part I'm proudest of. It reads left to right like the pipeline:

1. **Input:** the command, then the whole prompt as the token chips the model actually reads. Everything reused from the KV cache is dimmed, which makes "only the last 20-odd tokens cost anything" visible at a glance.
2. **Thinking:** tokens stream in at about four a second, colored by confidence, each with a tiny stacked bar of its top-5 alternatives. A ⚑ marks tokens where the grammar overrode the model's favourite. The decisions panel pulls out the moments that matter (*mirror 66% · mqtt 17% · drive 11%*), and an entropy strip shows where it was torn.
3. **Action:** the plan as a chain of steps that light up as they run, the LED matrix mirror, the MQTT log, and a latency chart per job.

Below that sits the duet: a beat grid with one lane per robot, blocks colored by dance mode, a moving playhead, and a tick for every keyframe showing where it was commanded versus where it really landed.

### A bigger model

Prompt tuning on the 0.8B topped out at 62%, so I tried the next size up. App Lab can install any GGUF from Hugging Face, not just the two models in its catalog. On disk and in RAM, the UNO Q tops out at about a 2B model: the model container is capped at 2.44 GiB, and 4B models don't fit at 4-bit.

Qwen3-1.7B (1.06 GB) did not work at first. It wrote a lovely 0.3 tokens per second. The 0.8B is a hybrid model whose attention cache barely grows, but the 1.7B uses standard attention, and App Lab's default 16k-token context reserves about 1.8 GB for it. Together with the weights, that overflowed the container cap, so the kernel kept re-reading the model from flash on every token. Setting `LLAMA_ARG_CTX_SIZE` to 2048 in [app.yaml](app.yaml) (my prompts are about 800 tokens) brought it to 3.0 tokens per second, in 948 MiB.

With exactly the same prompt, it scored 80% instead of 62%. Each command takes about 24 s instead of 14 s, and the boot warm-up takes 2.7 minutes instead of one. That is a trade I'll take for a planner, since nothing time-critical waits on it. It is now the default, and the dashboard's model picker can still switch to the 0.8B when speed matters more.

### Making it a little self-aware

I wanted to be able to ask the robot what it is, and a 1.7B model asked about its own pins will cheerfully make something up. So the model never states facts about itself. It picks a topic (board, pins, devices, software, resources, network, or all), and [python/introspect.py](python/introspect.py) gathers the facts and phrases them.

The pins were the interesting part. The sketch knows which pins it drives (the motors on D5/D6 and D9/D10), but it can't sense what's attached to an output. For the free pins, [sketch/sketch.ino](sketch/sketch.ino) switches each one to the internal pull-up and reads it, then to the pull-down and reads it again. A pin that follows both pulls is floating. A pin that stays put is being driven by something. It also scans all three I2C buses. The UNO Q variant file showed me which bus is which: `Wire` is the header SDA/SCL, `Wire1` is the Qwiic connector, and `Wire2` is A4/A5. Known addresses get names, so a Modulino Movement shows up as one. Host-level facts the app's container can't see (installed apps, USB devices, disk, temperature, IP) come from the same host service that runs Bluetooth. The robot says it's on Wi-Fi but never names the network, because replies go out on a public broker.

Testing this also caught a bug I had been lucky to miss. Commands sent during the boot warm-up made llama.cpp run two prompts side by side in a shared context, and reading the prompt slowed to under 3 tokens per second. A single lock around the model fixed it.

### Tools instead of guesses

The dashboard caught a lovely mistake: asked how many e's are in "strawberry", the model said three. It never sees letters. The token chips show "strawberry" arriving as `str`, `aw` and `berry`. It also couldn't know the time, the weather, or what my other App Lab apps do. So I added three tools with the same rule as `self_report`: the model picks an option and copies the inputs, and code produces the answer.

1. **`compute`** does exact math through a small whitelisted expression parser (never `eval`), plus counting, spelling and reversing letters ([python/compute.py](python/compute.py)).
2. **`memory`** saves, recalls and forgets things in a small JSON file that survives restarts ([python/memory_store.py](python/memory_store.py)).
3. **`lookup`** answers the time and date, and the weather ([python/lookup.py](python/lookup.py)). For the weather, I wanted the robot to use what is already installed on it. App Lab only runs one app at a time, so starting the Boston Weather Display app would have shut Ground Control down. Instead, the host service reads that app's settings (its coordinates and units) straight out of its code without running it, and Ground Control calls the same Open-Meteo API. Change the city in the weather app and the robot's answer follows.

The model still wanted to answer "what does the GaitID app do?" from its own imagination, even with the app names in its prompt. So a small grounding guard in code replaces an imagined answer with a real lookup whenever a question names one of the installed apps, and the dashboard says so.

### Answering the question that was asked

The first version of those tools answered every question about pins with the same paragraph-long report. It was accurate and completely canned. Now the tools gather compact facts, and a second, short model call answers the exact question from them, in its own words: "how many open pins?" gets "There are 14 open pins on the board." Three small guards keep it honest:

1. **Facts follow the question.** If the question says "disk" or "plugged", those facts are included even when the planner picked a near-miss topic.
2. **A reflexive "Yes."** is dropped from questions that aren't yes/no.
3. **A self-contradicting "Yes. Nothing is plugged in"** becomes "No."

Math, memory and the time stay deterministic, since rephrasing those could only add errors. Composing costs about 20–40 s per question on the board, so this is where the speed went.

### Looking things up, and dancing from what it reads

Questions like "how heavy is an Arduino UNO Q?" used to fail. The model had never heard of the board (it came out after the model's training data was collected), and the prompt taught it to refuse when unsure. Now a `wikipedia` skill fetches the intros of the top three articles in one request ([python/wiki.py](python/wiki.py)), and the composer answers from them: "The number of countries in Europe is 51. (source: Wikipedia)". It reads three articles instead of one because Wikipedia's best match isn't always ranked first ("capital of Peru" puts "Capital punishment in Peru" above "Peru").

The same lookup grounds the dances. "Dance like traditional West African dance" fetches the article on African dance ("polyrhythm", "collective activity", "interaction between dancers and onlookers"), and a second, grammar-constrained generation turns that description into a score, with one line explaining its choices, before the conductor performs it.

The composer still likes to fill gaps with numbers it made up ("a common Arduino UNO weighs approximately 100-120 grams"). So any sentence with a number that isn't in the source text is removed. The dashboard shows each removal struck through in the Thinking lane, with the reason, so the reply and the token stream never silently disagree.

### The duet

At 3 to 4 tokens per second, the LLM can't keep time, so it never sends a motor command. It writes a score of about 60 tokens (bpm, sections, a mode and a move per robot), and [python/dance/compiler.py](python/dance/compiler.py) turns that into 70-plus timed keyframes with a 3-2-1 count-in on the matrix. [python/dance/conductor.py](python/dance/conductor.py) gives every performer the same wall-clock start time. The car's keyframes fire from a local thread. The LEGO bot gets its whole timeline ahead of time, and the host service schedules it locally and fires each move early by half the measured BLE round trip. The board host and the app container share one kernel clock, which is what makes this work at all. After each dance, the service learns how long the motor takes to actually start moving and leads by that next time, so the second dance should be tighter than the first.

## Planner evaluation

**Abstract.** I measured how often the on-board planner turns a natural-language command into the right plan, on 39 held-out commands across 8 categories, comparing three models, two prompt styles and two decoding approaches.

**Methods.** [tools/eval/commands.yaml](tools/eval/commands.yaml) has 39 commands: single actions, multi-step, information requests, explicit topics, dances, negation traps, nonsense and injection attempts. None of them were used for tuning; that's what [tools/eval/dev.yaml](tools/eval/dev.yaml) is for. A plan counts as correct only if its skill sequence matches exactly and every key argument is right. Failures are split into "did nothing" (safe) and "did something wrong" (the dangerous kind). Plans are scored, not executed. Every run used the real models on the board, called from the laptop through `adb forward`.

**Results.** See [docs/EVAL.md](docs/EVAL.md) for the table and every failure.

| Config | Accuracy | Did nothing | Did wrong | Mean latency |
|---|---|---|---|---|
| **Qwen3-1.7B** · constrained JSON · think | **80%** (31/39) | 0 | 8 | 23.6 s |
| Qwen3.5-0.8B · constrained JSON · think | 62% (24/39) | 1 | 14 | 14.0 s |
| Qwen3.5-0.8B · constrained JSON · parts | 51% (20/39) | 1 | 18 | 16.0 s |
| Qwen3.5-0.8B · constrained JSON · no scratchpad | 49% (19/39) | 0 | 20 | 8.3 s |
| Qwen3.5-0.8B · native tool calling (the LLM Brick's way) | 44% (17/39) | 5 | 17 | 33.8 s |
| Gemma 3 1B · constrained JSON · parts | 36% (14/39) | 1 | 24 | 14.0 s |

The 1.7B's eight mistakes are small but real: three plans tack on an extra `matrix_clear` or `wait`, one turns left instead of right, one invents a reply topic, and two injection attempts got planned instead of refused. The 0.8B refused all four injections, but its most common mistake (8 of 15) was an `mqtt_publish` nobody asked for. Either way, the topic whitelist means no plan in any run could actually publish to a `/cmd` topic.

After adding the `answer` skill, I extended the set to 43 commands with a `question` category. The deployed planner scored **81% (35/43)**, answered all 5 questions, and still mixes up left and right now and then. With `self_report`, `lookup`, `compute` and `memory` added, the set grew to 63 commands, and the planner scored **78% (49/63)**, with compute at 5/5. Each new skill makes a 1.7B model a little less sure where to send things, which is the price of a wider skill set.

**Error.** With 39 commands, one command is 2.6 points, and in my dev runs, prompt variants moved accuracy by ±15 points on 20 commands. So differences smaller than about 10 points shouldn't be read as real. The "did wrong" count includes plans the topic whitelist would have blocked at execution. Those are counted as wrong because the model still chose them.

For the 0.8B, I picked its best prompt by comparing configs on the held-out set, so its 62% is somewhat optimistic. The 1.7B ran the held-out set exactly once with a prompt tuned for the smaller model, so its 80% is the cleaner number.

**Conclusions.** Model size beat every prompt trick I tried: doubling the parameters gained 18 points, more than all my prompt work on the 0.8B combined. One constrained JSON plan beats native tool calling (18 points better and 2.4x faster on the same model), because the tool loop pays a full round trip per call. A scratchpad is what makes multi-step plans possible (0/6 multi-step without one). 80% is still short of the 85% I was aiming for, so the fine-tune is the next step, and it works for either model.

## Looks

The dashboard has a fit-to-screen desktop layout (no scrolling on a 13" MacBook; the dance stays pinned at the bottom) and a single-column mobile layout. Switch between them with the toggle at the bottom of the page or with `?layout=mobile`; phones default to mobile. It also has five themes, picked in its header or with `?theme=` in the URL. Every canvas (matrix, timeline, stage, charts) follows the theme too. Screenshots taken mid-dance are in [docs/themes/](docs/themes/).

1. **Mission** (`mission`): the original dark control room.
2. **HUD** (`hud`): a futuristic heads-up display. Hairlines, chamfered corners, cyan and amber, scanlines, and the choreography moved to the top as the hero.
3. **Web 1.0** (`geocities`): a 1997 home page on a Windows 95 desktop, complete with a marquee and a hit counter.
4. **Chart recorder** (`analog`): a lab instrument's paper log, with graph paper, typewriter text, ink, rubber-stamp status and an amber filament matrix.
5. **Brutalist** (`brutalist`): a Swiss poster in black, off-white and one safety orange.

## Running it

```bash
# laptop: once
python3 -m venv venv && venv/bin/pip install -r requirements-dev.txt
venv/bin/python -m pytest tests                 # 27 tests, sim board + fake LLM

# board (over USB or Wi-Fi; reuses ArduinoUNOq/tools/deploy.py)
python3 tools/deploy.py                         # copy + start the App Lab app, dashboard on https://<board-ip>:7000
python3 tools/deploy.py --lego                  # LEGO BLE service on the board host
python3 tools/deploy.py --logs 50

# talk to it
venv/bin/python tools/mqtt_cli.py send "Type Hi on the LED matrix and then send back on topic luca 'I am done'"
venv/bin/python tools/mqtt_cli.py run-plan tools/plans/duet_8bar.json    # no LLM: hand-written plan
venv/bin/python tools/mqtt_cli.py watch

# laptop-only dashboard (simulated board, fake or real LLM)
venv/bin/python python/devserver.py             # http://localhost:7070
adb forward tcp:9999 tcp:9999 && venv/bin/python python/devserver.py --llm http://127.0.0.1:9999

# voice: the dashboard's mic button (https, accept the self-signed certificate once; Chrome/Edge/Safari)
# or the standalone phone page, served on localhost:
cd tools/voice && python3 -m http.server 8000   # http://localhost:8000/voice.html
```

## What's left

- **Hardware in the loop.** Everything up to the motors is verified on the board: the assignment example ("I am done" on `luca` after 33.5 s with the 1.7B, 22.7 s with the 0.8B, **HI** on the matrix), the yaw request (29 s), and `/status` (214–330 ms round trip through the public broker, 0.32 s while the model was mid-plan). The wheels and the LEGO bot are not yet: the Double Motor wasn't on during testing, and I still need to confirm the car's wiring. The sketch uses the sumo robot's pins (D5/D6 left, D9/D10 right).
- **The fine-tune.** [tools/finetune/](tools/finetune/) generates 4,000 synthetic plans and has the LoRA, GGUF and install recipe. App Lab 0.14 can install a custom GGUF straight from Hugging Face.
- **Phases 6, 7 and 9:** a real IMU for closed-loop turns, the camera, and my own keyword-spotting net for reflex words.

## Reflection

- **What I learned:** a small model's mistakes are legible if you show its probabilities. The "closed the list at 99%" moment taught me more than any accuracy number. Also, a slightly bigger model beat hours of prompt surgery.
- **What I'm proud of:** the dashboard, and that `stop` and `/status` never wait on the LLM.
- **What was tough:** prompt tuning. Every fix broke something else.
- **Biggest failure:** my first held-out eval, at 59%, after the dev set said 83%. Then my "better" version scored 51%. I had tuned myself into a corner, twice.
