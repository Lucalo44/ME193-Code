# The planner: what it can and can't do

Ground Control's brain is a 1.7-billion-parameter language model running on the UNO Q itself, with no cloud and no laptop in the loop. I was honestly surprised how far it got. It started as a model that would type "Hi" and then forget the rest of the command, and two days later it plans 98% of my held-out commands correctly. It's still a small model on a 4 GB board, though, so this page is as much about what it can't do as what it can, and about how much of the "intelligence" is actually ordinary code standing around it.

The model is Qwen3-1.7B with my own LoRA fine-tune (v2), quantized to 4-bit (Q4_0, about 1 GB). It runs in llama.cpp inside App Lab's model runner with a 2,048-token context. It reads a prompt at about 5 tokens per second and writes at about 2, so a typical command takes 21.5 s from send to plan, and the board needs about 3 minutes to warm up after every boot.

## What it can do

Every command becomes one JSON plan: a short `think` naming the skills, then up to 8 steps. From simplest to most complex:

1. **Answer without the model.** These never wait on the LLM, even while it's mid-sentence:
   1. `/status` answers in well under a second.
   2. "stop", "stop dancing", "please stop", "cancel" or "that's enough" halts everything in milliseconds.
   3. A new command cancels whatever is still running and starts right away.
   4. "do it again" replays the last plan.
   5. "loop the previous dance twice, faster, on the matrix" replays the last dance with those changes.
2. **Single actions.** "show a heart", "write HELLO", "drive backwards for 3 seconds", "turn left 90 degrees", "spin around three times". The LED matrix has 24 icons, and anything without an icon ("draw a giraffe") gets written as a word.
3. **Sequences and replies.** "Type Hi on the LED matrix and then send back on topic luca 'I am done'", up to 6 steps in a chain. Messages can carry live values: "what is your yaw? reply on luca" writes `{imu.yaw}`, and the executor fills in the real reading as the message goes out.
4. **Questions it answers with tools, not guesses.** The model picks a tool and copies the inputs, and code produces the facts:
   1. `compute` for exact math and letters ("how many e's in strawberry?").
   2. `lookup` for the time, date, weather and forecast, and what my other App Lab apps do.
   3. `memory` to save, recall and forget things ("remember my name is Luca").
   4. `self_report` for facts about the board itself: pins, I2C devices, RAM, disk, temperature.
   5. `wikipedia` for everything else ("how tall is the Empire State Building?"). It reads the top three article intros, and a second short model call answers the exact question from them.
5. **Dances.** The model writes a short score, and deterministic code performs it:
   1. Plain dances: "dance 4 bars at 100 bpm, the lego bot mirrors you". It picks the tempo, sections, a mode (unison, mirror, call and response, canon, solos) and a move for each robot.
   2. Two performers told what to do: "the robot spins while the lego bot claps".
   3. Named styles: "dance a waltz", "two breakdancers on the matrix at 120bpm". Twelve styles come from a built-in library with their own meter, tempo, rhythms and poses. Any other style is choreographed from its Wikipedia article.
   4. Where: on the robots, as stick figures on the LED matrix (one or two dancers), or both.
   5. Looping: "loop it", "3 times", "until I say stop".
   6. Reports afterwards: "then tell me how in sync you were on luca" fills in the measured sync error.
6. **Refusing.** Gibberish gets no steps. So do unsafe requests and prompt injections, and negations are respected ("show a heart but don't send anything").

## What it can't do

1. **It has no memory of the conversation.** Each command is planned on its own. "Do it again" and "the previous dance" only work because code remembers the last plan and the last dance and replays them, not because the model remembers anything.
2. **It can't keep time.** At 2 tokens per second, the model never sends a motor command. It writes a score of about 60 tokens, and the compiler and conductor turn that into about 110 timed keyframes for a 4-bar duet.
3. **It can't see letters or know facts after its training.** "strawberry" reaches it as `str`, `aw` and `berry`, so counting letters goes to `compute`. It had never heard of the UNO Q, so facts go to Wikipedia. When it answers from facts, any sentence with a number that isn't in the source gets removed.
4. **It only knows the skills it was trained on.** A new skill means new training data and a new fine-tune (26 minutes on the Tufts HPC cluster). That's why "the previous dance" is handled in code and not by the model.
5. **It's slow, and it's one model at a time.** About 21.5 s per command, with a p90 of 32 s, and a 3-minute warm-up after every boot. While it plans one command, the next one waits (or cancels it). The board froze once during a warm-up and needed a power cycle.
6. **It still makes mistakes.** On the held-out set, it turned "turn 180 degrees" (no direction) into -180, and heard "spin to times" (speech-to-text for "two times") as three. For styles outside the library, it guessed polka is in 3/4 (it's 2/4).
7. **The stick figures are tiny.** Each dancer is 5 LEDs wide and 8 tall, and the matrix fits at most two side by side.
8. **Not yet verified on the motors.** Everything up to the motors is tested on the board. The car's wiring is still unconfirmed, and the LEGO bot runs simulated until the Double Motor is connected. So the 16th-note rhythms drive only the matrix figures for now, and the robots dance on the beat.
9. **Limits on purpose.** Commands are capped at 300 characters, plans at 8 steps, the queue at 5 jobs, and dances at 60 s (5 minutes looped). A plan can only publish to `luca` or `ME193/q/luca/*`, never to a `/cmd` topic, because everything arrives on a public broker.

## How it works

A command goes through six stages. Only the third one is the model.

1. **Reflexes** ([python/mqtt_io.py](../python/mqtt_io.py), [python/core.py](../python/core.py)). Status, stop, "do it again" and "the previous dance" are caught here with plain pattern matching. A new command cancels the running one first.
2. **The prompt** ([python/planner/prompt.py](../python/planner/prompt.py)). The rules and the skill list come to 867 tokens, with no few-shot examples, since the fine-tune moved those into the weights. llama.cpp keeps that prefix in its KV cache, so after the warm-up only the 20-odd tokens of the command itself cost anything.
3. **Constrained generation** ([python/planner/planner.py](../python/planner/planner.py)). A JSON grammar built from the skill registry ([python/skills/registry.py](../python/skills/registry.py)) means the model can only write a valid plan: real skill names, arguments inside their ranges. It writes the `think` first, which works as a scratchpad. If the steps don't match the skills it named, a repair pass pins the think and forces exactly that sequence.
4. **Guards.** Small pieces of code catch the mistakes the model makes most:
   1. Questions about my installed apps get a real lookup instead of an imagined answer.
   2. Composed answers lose any sentence with an invented number.
   3. A think cut off at its length cap is matched as a prefix, not treated as a mismatch.
   4. A named style or "two dancers" in the user's own words wins over a generic dance plan. "Breakdancers" reaches the model as "break" + "d" + "ancers", and it once came back as a plain dance.
5. **Execution** ([python/executor.py](../python/executor.py)). Each step runs with its own timeout, placeholders like `{imu.yaw}` and `{dance.max_sync_error_ms}` are filled in with live values, and a plan that sends nothing still replies "done".
6. **Dancing** ([python/dance/](../python/dance/)). The compiler turns a score into keyframes per performer, with a count-in (4-3-2-1, or 3-2-1 for a waltz). The conductor gives every performer the same start time and measures how in sync they were. The matrix figures run on a 16th-note grid: a pose changes only on a rhythm hit, the figure flashes on the hit, and corner lights mark every beat.

The dashboard shows all of this live: the prompt as token chips (dimmed where the cache saved the work), every token with its top-5 alternatives, the decisions where the model picked a skill or mode, and the dance on a beat grid.

## How it developed

### Day one: a small model that forgot half the command

I started with App Lab's default model, Qwen3.5-0.8B. The first job was measuring what the board could do ([SPIKES.md](SPIKES.md)): llama.cpp on the board streams tokens, constrains output to a JSON schema and returns the top-5 alternatives for every token, all in one request. That made the dashboard possible before any planning worked.

The planning did not work at first. The model would type "Hi" and close the plan, with 99% confidence, even when its own think said the next step was to publish. Few-shot examples as chat turns, the think scratchpad and the repair pass got it to 59% on the held-out set. A cleverer version that copied the command's clauses first scored 70% on my dev set and 51% held out. I had tuned myself into a corner. The best prompt for the 0.8B reached 62%.

### A bigger model, and tools instead of guesses

Switching to Qwen3-1.7B with the same prompt jumped to 80%. It didn't run at first, though: it wrote 0.3 tokens per second, because App Lab's default 16k context reserved about 1.8 GB of KV cache and overflowed the runner's memory cap. Setting the context to 2,048 tokens brought it to 3 tokens per second.

Then I kept adding things a small model can't do on its own: an `answer` skill, `self_report` with real pin probing, `compute`, `lookup`, `memory` and `wikipedia`. Each one followed the same rule: the model picks an option, and code produces the facts. Accuracy hovered around 78–81% while the skill list grew, because every new skill makes a 1.7B model a little less sure where to send things. The LED matrix became a third dancer here too, as a stick figure.

### Day two: fine-tuning

Prompt tuning had stopped paying off, so I fine-tuned. [tools/finetune/gen_data.py](../tools/finetune/gen_data.py) generates synthetic commands from per-skill templates, so every training plan is right by construction, and anything close to an eval command is held out. v1 trained on 6,555 examples on a free Colab T4 in about 2½ hours, after fighting a torchao incompatibility and a T4 that claims to support bf16. It scored 92% against the base model's 75% on 72 commands, and it was 24% faster, because dropping the few-shot examples halved the prompt.

v1's misses showed me what the data lacked: negations (a dead branch in my generator), gibberish, two performers, reports after a dance. The second data set filled every gap I could list: 8,930 examples, including voice transcripts, numbers as words, typos and longer chains. Colab was going to run out of session time, so I moved training to the Tufts HPC cluster, where an A100 trained v2 in 26 minutes. v2 scored 98% on 84 commands. Its first eval said 90%, which turned out to be my bug: the planner gave it the old few-shot prompt it was never trained on.

### Making the dances worth watching

With planning solved, the weak spot moved to the performance. Testing showed v2 planned two-dancer and tempo commands correctly, but the matrix could only draw one figure, every style looked like "mirror, step, wave", and some moves flickered four times a beat. So I added:

1. Two dancers side by side, facing each other in mirror sections.
2. A 16th-note rhythm grid with a brightness flash on every hit and beat lights in the corners.
3. A library of twelve styles, including a waltz in real 3/4.
4. A 3×2 head on each figure. With a 1-pixel head, raising the arms made the figure look headless.

Alongside that came looping, a stop that understands "stop dancing", new commands that replace old ones, and "the previous dance".

## Planner evaluation

**Abstract.** I measured how often the planner turns a natural-language command into the right plan, on 84 held-out commands, comparing the fine-tuned v2 against v1 and the earlier base models. Full tables and every failure are in [EVAL.md](EVAL.md).

**Methods.** [tools/eval/commands.yaml](../tools/eval/commands.yaml) covers single actions, multi-step plans, information requests, explicit topics, dances, questions, self-reports, tools, Wikipedia, styles, the matrix, negation traps, nonsense and injection attempts. A plan counts as correct only if its skill sequence matches exactly and every key argument is right. Plans are scored, not executed. Every run used the real model on the board, called from the laptop over `adb forward`.

**Results.**

| Model | Held-out accuracy | Mean time per command |
|---|---|---|
| Qwen3.5-0.8B, best prompt | 62% (24/39) | 14.0 s |
| Qwen3-1.7B, same prompt | 80% (31/39) | 23.6 s |
| Qwen3-1.7B, base, after the tools were added | 75% (54/72) | 27.9 s |
| Fine-tune v1 | 92% (66/72) | 21.3 s |
| **Fine-tune v2 (deployed)** | **98% (82/84)** | **21.5 s** |

**Error.** With 84 commands, one command is 1.2 points, so v2's lead over v1 (87% on the same 84) is 9 commands, not noise. The 12 newest commands are my own wording, written to differ from the training templates, but I wrote both, so the held-out set is probably still closer to the training data than a stranger's commands would be. The set also grew between rows, so only rows on the same set compare directly.

**Conclusions.** Model size beat every prompt trick, and fine-tuning beat model size. The remaining misses are a missing direction and a speech-to-text homophone, not misunderstood commands. The biggest wins came from code around the model: grammar-constrained output, tools that produce facts, and guards that catch the mistakes the model makes most.

## Reflection

- **What I learned:** a small model's mistakes are easy to read once its probabilities are on screen. Also, the training data decides everything: v1 couldn't do negations because one branch of my generator never ran.
- **What I'm proud of:** 98% from a model that fits in 1 GB on the board, and the fact that stop, status and replays never wait on it.
- **What was tough:** prompt tuning. Every fix broke something else, until fine-tuning made most of it unnecessary.
- **Biggest failure:** scoring v2 with the wrong prompt and nearly reporting 90%. The model was fine; my code picked the prompt by exact name.
