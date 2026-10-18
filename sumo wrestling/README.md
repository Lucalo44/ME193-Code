# Sumo Wrestling

LEGO robot that flips opponents out of a blue-tape ring. Limit switches on the
paddle, read by the UnoQ, report contact over MQTT; the laptop runs the policy
and drives the LEGO double motor. If pushed out of the ring, the robot stops
and plays a sad song.

Constraints: <= 1 kg, fits in a 30 cm cube, decision policy (behavior tree /
decision tree / KNN / K-means), >= 2 sensors (>= 1 autonomous, >= 1 teleop).

## How the pieces talk

```
        LEGO Controller ──┐  Bluetooth                    WiFi
      LEGO Color Sensor ──┼──────────────►  LAPTOP  ◄──────────►  MQTT broker
LEGO Double Motor (flip)◄─┘                 sumo.py                    ▲
                                       (decision tree)                 │ WiFi
                                                                       ▼
                                 limit switches ──► UnoQ ──► motor driver ──► wheel motors
                                                ArduinoUNOq/sumo-unoq
```

| Part | Role | Connection |
|---|---|---|
| LEGO Controller | Teleop: levers = tank drive, button = start / manual flip | Bluetooth to laptop |
| LEGO Color Sensor | Autonomous: sees the blue tape → escape / "we're out" | Bluetooth to laptop |
| LEGO Double Motor | Swings the flipper paddle; its speaker plays the sad song | Bluetooth to laptop |
| UnoQ | Reads the limit switches, drives the wheel motors | WiFi → MQTT |
| Laptop | Runs `sumo.py` (the decision tree) | Bluetooth + WiFi |

MQTT topics (broker `test.mosquitto.org`, base `ME193/Luca/sumo`):

| Topic | From → To | Payload |
|---|---|---|
| `…/switch` | UnoQ → laptop | `{"pressed": true, "left": true, "right": false}` on change + every 0.2 s |
| `…/drive` | laptop → UnoQ | `{"left": 200, "right": -150}` (-255..255); UnoQ stops the wheels after 0.75 s of silence |
| `…/state` | laptop → anyone | `{"state": "TELEOP", ...}` for debugging |

## Wiring (UnoQ side)

The LEGO parts have their own batteries and need no wires. Only the UnoQ is wired.

**Motor driver**: the same two-channel driver and pins as `mqtt-minifig-drive`:

| Driver | UnoQ pin | Goes to |
|---|---|---|
| M1A / M1B | D5 / D6 | left wheel motor (M1 screw terminal) |
| M2A / M2B | D9 / D10 | right wheel motor (M2 screw terminal) |
| VB+ / VB- | — | motor battery + / − |
| GND | GND | shared ground with the UnoQ (automatic if it's a shield) |

**Limit switches**: two microswitches on the front of the paddle:

| Switch terminal | UnoQ pin |
|---|---|
| left switch COM | GND |
| left switch NO (normally open) | D2 |
| right switch COM | GND |
| right switch NO | D4 |

Each microswitch has three pins: **COM** (common), **NO** (normally open)
and **NC** (normally closed), usually printed on the switch's side. Use COM
and NO and **leave NC unconnected**. If the pins aren't labelled, use a
multimeter's continuity mode: the pin that beeps with COM only while you
press the lever is NO.

The pins use the UnoQ's internal pull-up resistors, so no resistors are
needed: released = HIGH, pressed = LOW. **Never connect a switch to 5V**; the
UnoQ's microcontroller pins are 3.3 V. If you only have one switch, wire it to
D2 and leave D4 empty (it reads as released).

**Power**: the UnoQ rides on the robot and needs its own supply, either a
small USB-C power bank or the motor battery into VIN. Check the board's VIN
rating before connecting a battery there. Weigh the batteries; they count
toward the 1 kg limit.

## Mounting

- **Color sensor**: at the **rear-center** of the chassis, pointing straight
  down, close to the floor. The robot gets pushed backwards out of the ring, so
  the rear reaches the tape first. `sumo.py` drives forward (ESCAPE) when it
  sees blue.
- **Limit switches**: on the paddle's front face with the levers sticking out
  past the edge, so touching an opponent presses at least one of them.
- **Double Motor**: drives the paddle hinge. If the two outputs drive it from
  opposite sides (mirror image), keep `FLIP_MIRRORED = True` in `sumo.py`;
  otherwise set it to `False`.

## Setup

```bash
cd "sumo wrestling"
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

1. Deploy the UnoQ app (board on WiFi, IP current in `ArduinoUNOq/tools/board_secrets.py`).
   `sumo-unoq/` in this folder is a copy of `ArduinoUNOq/sumo-unoq/`, the copy
   that actually gets deployed. Copy it into your own `ArduinoUNOq/` folder
   first, and keep the two in sync when you change either one:
   ```bash
   cd ../ArduinoUNOq && python3 tools/deploy.py sumo-unoq
   ```
   The log prints `[switch] left=ON ...` when you press a switch.
2. Turn on the three LEGO devices, then on the laptop:
   ```bash
   python sumo.py --motor-card <serial> --sensor-card <serial> --controller-card <serial>
   ```
3. Press the controller button to start the match. Ctrl+C stops everything.

## First-test checklist

- [ ] Wheels lifted off the table: push both levers forward, both wheels spin
      forward. If not, flip `INVERT_LEFT/RIGHT` in `sumo.py` or swap that motor's wires.
- [ ] Press each limit switch by hand: the paddle flips, `sumo.py` shows `AUTO_FLIP`.
- [ ] Short-press the controller button during a match: `MANUAL_FLIP`.
- [ ] Hold the robot's rear over the tape: `ESCAPE`, then `LOST` and the sad song after 0.6 s.
- [ ] Ctrl+C on the laptop: the wheels stop within 0.75 s.
