# Stop-motion brief: Hand Safe

You are making a short stop-motion video, about 2–3 minutes long, about a student project called **Hand Safe**. This file has everything you need. The facts section is verified against the real project, so use those numbers exactly and don't invent others. The scene plan is a suggestion; you can change the staging, but every required topic must appear.

## Required topics (the assignment)

The video **must** clearly answer all four of these. Each is mapped to a scene below.

1. **Lock policy:** how does the lock make decisions? (Scenes 4–6)
2. **Limitations:** at least four potential limitations of the lock. (Scene 8; show all six if time allows)
3. **The neural net:** how does it work? (Scenes 2–3)
4. **UNO Q vs. computer:** how did the UNO Q setup compare with the laptop? (Scene 7)

## The project in one paragraph

Hand Safe is a combination safe for a mechanical engineering class (ME193), built on an **Arduino UNO Q** board. You push a **button**, the board's **LED matrix** lights up, and you enter a **three-step code**. Each step is a number from 1 to 5, either **held up with your fingers** in front of a USB webcam or **spoken out loud** into the webcam's mic. Which steps are spoken is part of the secret. If the whole sequence is right, a **servo motor** swings the bolt open. Two small neural networks, one for hands and one for voice, do the recognition. Both were built and trained from scratch in PyTorch.

## Verified facts (use these numbers)

**Hardware**
- Arduino UNO Q. It has two chips: a Linux computer (4× Cortex-A53 cores, 3.6 GB RAM) runs the Python and the neural nets, and a small microcontroller (STM32) runs the button, servo and LEDs. They talk over a link called the Bridge.
- One USB webcam supplies both the camera and the microphone.
- The button has 2 wires: one to pin D2, one to ground.
- The servo has 3 wires: ground, 5 V, and signal to pin D9.
- The built-in 8×13 LED matrix shows the safe's status.

**Hand network (GestureNet)**
- Input: a 96×96-pixel color crop of a fixed square box in the camera image.
- Output: 6 classes, "none" (no hand) or 1, 2, 3, 4, 5 fingers.
- Structure: 4 blocks. Each block applies two 3×3 convolution filters (each followed by batch norm and ReLU), then a 2×2 max pool that halves the image size. The channel count doubles each block: 16 → 32 → 64 → 128. Then global average pooling (128 numbers), then one linear layer to 6 scores, then softmax to turn the scores into probabilities.
- Size: 294,774 learned weights.
- Training data: about 900 photos per class, about 5,400 in total, recorded in 4 sessions.
- Training: 30 epochs, AdamW optimizer, with random rotation, lighting and color changes applied to every photo every epoch.
- Accuracy: **91.9%** on photos it never trained on. The most common mistake is reading a **3 as a 2** (29 of 180).
- An early failure worth showing: the first dataset recorded each number in a different room. "2" was the only number shot in front of a beige wall, so the network learned "beige wall = 2" and called everything 2. Its accuracy was **0.9%**. Re-recording every number in every location fixed it.

**Voice network (VoiceNet)**
- Same network shape, but the input is a **spectrogram**: a picture of sound with time on one axis and pitch on the other. Each one covers 1 second of audio as 40 pitch bands × 98 time steps.
- A loudness trigger cuts the 1 s clip, starting 0.2 s before you speak.
- Output: "none" or the words one through five.
- About 50 recordings per word. Accuracy: **94.8%** on 58 held-out clips.

**Lock policy (how it decides)**
1. Idle until the button is pressed. Then the LEDs light up and you have 30 seconds.
2. Hand steps: every camera frame "votes" for whatever the network thinks it sees, if it's at least 50% sure. Each vote adds time to that number's credit, and drains the other numbers' credit at half speed. The first number to reach **3 seconds of credit** is accepted. A steady hand takes exactly 3 s; a flickery reading just takes a bit longer. The accepted number flashes big on the LED matrix. Then you must pull your hand away before the next step.
3. Voice steps: a spoken number counts if the voice network is at least 70% sure. A spoken step flashes the number with "sound bars" beside it.
4. After 3 steps, the safe checks the whole sequence, **including which steps were spoken**, against a scrambled (hashed) copy of the code. It never stores the code itself. Every step accepts either hands or voice, so someone watching can't learn which steps are supposed to be spoken.
5. Correct: the servo opens the bolt for 6 seconds, then relocks. Wrong: a red X. Three wrong tries: locked out for 30 seconds.
6. Safety backstop: if the bolt has been open for 15 seconds, the microcontroller relocks it on its own, even if the main program has crashed.

**UNO Q vs. laptop (same code, same weights)**
- Hand network, one frame: **about 2 ms** on the MacBook vs. **95 ms** on the UNO Q, roughly 50× slower.
- Voice network, one clip: **about 2 ms** vs. **161 ms**.
- Consequence: on the board, the camera, the voice network and the live video stream had to run as separate parallel tasks (threads). Before that, the video stuttered and hand holds kept getting interrupted. After: the board processes about **15 frames per second**.
- The webcam's microphone was set to maximum gain (+24 dB) on the board's Linux, so ordinary room noise kept triggering the voice recognizer. Fix: turn the gain down to 50% and make the trigger follow the room's background noise level.
- Development happens on the laptop: record data, train, then copy the code and models to the board with one script.
- A laptop companion program shows the board's camera view live, at about 15 fps.

**Limitations (show at least four)**
1. The hand photos were taken at a desk, not from the webcam's final position inside the safe, so accuracy in the real safe may be lower.
2. 3 and 2 get confused the most.
3. The voice network has only about 50 recordings per word, all from one sitting.
4. Anyone watching you enter the code can copy it. Owner checks (palm creases for hands, speaker ID for voice) are built but not trained yet.
5. No liveness check, so a photo of a hand or a recording of a voice could pass.
6. Only 1,000 possible codes (125 number sequences × 8 hand/voice patterns), protected mainly by the 30-second lockout.

## Scene plan (about 2.5 minutes at 12 fps)

| # | Scene | What happens on screen | Caption / voice-over idea | Approx. length |
|---|---|---|---|---|
| 1 | Cold open | A cardboard safe with a big red button. A hand presses it; the LED matrix (paper squares or beads) lights amber. The hand shows 4 fingers, then a speech bubble says "two", then the hand shows 1 finger. The door swings open. | "A safe that opens for a hand-and-voice code." | 15 s |
| 2 | How it sees a hand | A camera frame with a square box. Cut the box out; it shrinks to a 96×96 tile. The tile passes through 4 "blocks" (stacked paper layers that get smaller and thicker) and becomes 6 bars: none, 1–5. The "3" bar grows tallest. | "Four filter blocks shrink the picture and look for finger shapes. Six scores come out. The biggest wins." | 20 s |
| 3 | Training + the beige-wall bug | Photo cards pile up (~900 per number). Comic beat: every card with a beige wall is stamped "2"; the network wears a "2" hat for everything. Then the cards get reshuffled across all rooms and the hat comes off. Show a scoreboard flipping 0.9% → 91.9%. | "First try: it learned the wall, not the fingers." | 20 s |
| 4 | How it hears | A sound wave turns into a striped spectrogram card (time × pitch). Same 4-block machine, output "one…five". | "Voice gets turned into a picture of sound, then the same kind of network reads it." | 15 s |
| 5 | Lock policy: the 3-second hold | A sand timer or a filling bar fills while the hand holds "4". Misread flickers ("2!") briefly drain a second, smaller bar but barely slow the main one. The bar fills, the LED matrix flashes a big 4, and the hand pulls away. | "Every frame votes. Three seconds of votes and the number locks in." | 20 s |
| 6 | Lock policy: the check | Three tokens, ✋4, 🗣2, ✋1, slide into a slot labeled with a padlock. A "hash" grinder turns them into scrambled confetti that matches a stored confetti pile → servo turns, bolt slides. Alternate take: wrong tokens → red X, then three X's → a 30 s timer. | "It checks the numbers and which ones were spoken. Three wrong tries: 30-second lockout." | 20 s |
| 7 | Laptop vs. UNO Q | Race: a laptop character finishes a frame in a blink (2 ms); the tiny UNO Q board walks it over (95 ms). Then the UNO Q splits into helpers (camera, voice, stream) working side by side, and the counter reads 15 fps. A small mic knob turns from max down to half. | "Same brain, 50× slower body. Fix: split the work into parallel jobs." | 20 s |
| 8 | Limitations | A checklist on a clipboard, one item ticked per beat: desk vs. safe camera angle; 3 vs. 2 confusion; few voice samples; shoulder-surfing; a printed photo of a hand fooling it; only 1,000 codes. | "What could still go wrong." | 20 s |
| 9 | Close | Door closes, matrix dims to one dot, the button waits. | "Hand Safe: ME193." | 5 s |

## Visual references

Real figures from the project are in `media/` next to this file. Use them as references for props, or show them briefly on screen:
- `fig-pipeline.png`: the full system diagram (scenes 1, 6)
- `fig-cnn.png`: the network's block shapes (scene 2)
- `fig-data.png`: real training photos (scene 3)
- `fig-confusion.png`: what the network gets right and wrong (scenes 3, 8)
- `fig-voice.png`: real spectrograms of "one" to "five" (scene 4)
- `fig-hold.png`: the credit bar filling to 3 s (scene 5)
- `fig-board-vs-laptop.png`: the speed comparison (scene 7)

LED matrix screens, for building props (8 rows × 13 columns):
- **Idle:** one dim dot at the bottom center.
- **Armed:** three small boxes across the top (one per step, filled in as steps are entered), plus a bar along the bottom that fills during a hand hold.
- **Step accepted:** that number drawn large in the center for about 0.6 s. Spoken steps add short vertical "sound bars" on both sides.
- **Open:** a check mark. **Wrong:** a blinking X. **Lockout:** a dimly glowing X.

## Rules

- **Never show the owner's real code.** The example code in this brief (hand 4, say "two", hand 1) is made up; use it or invent another.
- Keep the numbers in this file exact; don't round 91.9% to "over 90%" or 50× to "100×".
- Describe hardware mechanically: the servo turns, the board runs code. Don't make the board "think" or "learn" on its own; the training happens on the laptop.
- The tone is a student engineer showing their build: clear, a little dry humor, no hype.
