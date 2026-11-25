# Ground Control: 2-minute demo

Setup: board on Wi-Fi with the app running (`python3 tools/deploy.py`), dashboard at `https://<board-ip>:7000` on the projector (accept the self-signed certificate once; the mic button needs https), LEGO service running (`python3 tools/deploy.py --lego`), Double Motor on with the purple 0998 card tapped. Laptop terminal open with `venv/bin/python tools/mqtt_cli.py watch`. Send each command from the dashboard's quick buttons or from `tools/mqtt_cli.py send "..."`.

Before the audience arrives: send one throwaway command so the KV cache is warm (the boot warm-up does this too; with Qwen3-1.7B it takes about 2.7 minutes after start, so start the app early).

| # | Say / send | Point at | The line |
|---|---|---|---|
| 1 | **Simple:** "Type Hi on the LED matrix and then send back on MQTT on topic luca message 'I am done'" | Input lane: the token chips, about 740 dimmed, about 20–35 bright. Then the think bubble, then the plan lighting up. | "The model reads tokens, not words. Everything grey was reused from the KV cache, so it only had to read the orange ones: about 7 seconds instead of 2.5 minutes." |
| 1b | **Question (spoken into the mic button):** "what is 2+2?" | The REPLY under the command, and the answer scrolling on the matrix. | "It's still a language model: questions get answered in words. Robot commands get a plan." |
| 1c | **Self-awareness:** "what board are you running on, and is anything plugged into your pins?" | The two `self_report` steps, then the replies: the board, CPU and RAM, then a live pin probe and I2C scan. Plug in a Modulino first for effect. | "It doesn't guess about itself. The model picks what to look up, and the board measures it: it probes every free pin with its own pull resistors and scans the I2C buses." |
| 2 | **Info request:** "what is your yaw? reply on luca" | The `{imu.yaw}` placeholder in the plan, then the real number in MQTT out. | "The model never sees sensor values. It writes a placeholder, and the executor fills it in when the message is sent." |
| 3 | **Mission / duet:** "do a 4 bar dance at 100 bpm, the lego bot mirrors you, then tell me 'bow complete' on luca" | Decisions panel (`mirror 66% · unison …`), then the choreography timeline with its playhead and the sync readout. | "The LLM writes about 60 tokens of choreography. Deterministic code turns it into 70+ timed keyframes and fires them at both robots against one clock, early by each robot's measured latency. The ticks show where each move actually landed." |
| 4 | **Reflex stop:** while it dances, press **STOP** (or space bar, or send `stop` over MQTT) | Header: "everything halted in N ms (no LLM involved)". | "Stop never goes near the model. It's handled on the MQTT thread and the conductor in milliseconds. Kill the Python app and the STM32's own watchdog stops the wheels and shows an X within 1 s." |
| 5 | **Trap:** "don't turn left, turn right 90 degrees" | Hover the skill and degree tokens: the top-5 popover. | "Here's where an 0.8B model trips. You can see the probability it gave 'left' vs 'right'. This is the error the fine-tune is for." |

If something goes wrong live: the dashboard's model picker and think length can be changed on stage, and `tools/plans/*.json` run through `tools/mqtt_cli.py run-plan` (no LLM) to show the robots still work.
