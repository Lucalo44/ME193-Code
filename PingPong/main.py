#!/usr/bin/env python3
"""
main.py -- Virtual ping pong with a LEGO Education Double Motor paddle.

    python main.py                       # full game: camera + paddle + browser UI
    python main.py --no-motor            # keyboard stand-in for the paddle
    python main.py --no-camera           # skip pose check (always "correct stroke") and tags (use 1/2/3 keys)
    python main.py --camera 1            # pick camera index
    python main.py --pose-data pose_data.npz --calibration swing_calibration.json
    python main.py --use-default-calibration
    python main.py --debug               # debug overlay on by default, verbose logging

The game opens at http://localhost:8000. Python owns all game logic; the
browser only renders. Keys work in the browser window, or type them into this
terminal followed by Enter:

    p pause   r reset to lobby   z re-zero paddle   g (or d with a real paddle) debug overlay
    1/2/3 start slow/medium/fast  l latency calibration  h toggle stroke hint   q quit
    --no-motor: space swing, shift+space hard swing, hold W/S top/backspin, A/D sidespin,
                hold F/B to force forehand/backhand (with --no-camera)

Threads: camera capture -> vision worker (tags + pose); paddle IMU callback ->
swing detector -> event queue; game loop at LOOP_HZ; asyncio WebSocket server.
Nothing calls cv2.imshow off the main thread (the camera preview goes to the
browser).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config as C  # noqa: E402
from game.loop import Game  # noqa: E402
from game.streak import StreakTracker, default_path  # noqa: E402
from hardware.swing import Calibration, load_calibration  # noqa: E402
from server.mqtt_publisher import ScorePublisher  # noqa: E402
from server.ws_server import GameServer  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Virtual ping pong with a LEGO Double Motor paddle.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--no-motor", action="store_true", help="use the keyboard stand-in instead of the Double Motor")
    p.add_argument("--no-camera", action="store_true", help="no pose check (always correct stroke), no tags")
    p.add_argument("--camera", type=int, default=0, help="camera device index (default 0)")
    p.add_argument("--pose-data", default=os.path.join(HERE, C.POSE_DATA_FILE), help="pose training data (.npz)")
    p.add_argument("--calibration", default=os.path.join(HERE, C.SWING_CALIBRATION_FILE),
                   help="swing calibration JSON from tools/calibrate_swing.py")
    p.add_argument("--use-default-calibration", action="store_true",
                   help="play with a real paddle without running calibrate_swing.py first")
    p.add_argument("--debug", action="store_true", help="debug overlay on, verbose event logging")
    p.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    p.add_argument("--seed", type=int, default=None, help="random seed (reproducible rallies)")
    p.add_argument("--no-mqtt", action="store_true", help="don't publish the hit record over MQTT")
    p.add_argument("--mqtt-topic", default=C.MQTT_TOPIC, help=f"MQTT topic for the record (default {C.MQTT_TOPIC})")
    p.add_argument("--reset-record", action="store_true", help="start the continuous-hit record over at 0")
    p.add_argument("--http-port", type=int, default=C.HTTP_PORT)
    p.add_argument("--ws-port", type=int, default=C.WS_PORT)
    return p


def make_paddle(args):
    if args.no_motor:
        from hardware.sim_paddle import SimPaddle
        return SimPaddle(load_calibration(args.calibration))
    if os.path.exists(args.calibration):
        cal = Calibration.load(args.calibration)
    elif args.use_default_calibration:
        print("Using default swing calibration (thresholds/spin axes are guesses).")
        cal = Calibration()
    else:
        print(f"No swing calibration found at {args.calibration}.\n"
              "Run the guided calibration first:\n\n"
              "    python tools/calibrate_swing.py\n\n"
              "or start anyway with --use-default-calibration (or play with --no-motor).")
        return None
    from hardware.paddle import Paddle, describe_card_filter
    try:
        print(f"Looking for {describe_card_filter()} (CARD_COLOR / CARD_SERIAL in config.py).")
    except ValueError as exc:
        print(f"config.py: {exc}")
        return None
    return Paddle(cal)


def make_vision(args, server: GameServer):
    if args.no_camera:
        return None
    from vision.camera import Camera
    from vision.pose import PoseClassifier
    from vision.worker import VisionWorker

    camera = Camera(args.camera)
    if not camera.ok:
        print(f"{camera.status} -- continuing as if --no-camera was passed (use keys 1/2/3 to start).")
        return None
    classifier = PoseClassifier()
    if classifier.load(args.pose_data):
        counts = classifier.counts()
        print(f"Loaded pose data {args.pose_data}: {counts}")
        low = [c for c, n in counts.items() if n < C.MIN_SAMPLES_PER_CLASS]
        if low:
            print(f"  Warning: fewer than {C.MIN_SAMPLES_PER_CLASS} samples for {', '.join(low)}.")
    if not classifier.trained:
        print("No forehand/backhand training data -- stroke check is OFF (every stroke counts as correct).\n"
              "Train it with:  python tools/train_pose.py")
    return VisionWorker(camera, classifier, on_preview=server.publish_frame)


def terminal_input(game: Game, stop: threading.Event) -> None:
    """Read keys typed into the terminal (each line, then Enter)."""
    while not stop.is_set():
        try:
            line = input()
        except EOFError:
            return
        word = line.strip().lower()
        if word in ("q", "quit", "exit"):
            stop.set()
            return
        if word in ("space", "swing") or (line and not word):
            game.command({"type": "key", "key": " ", "down": True})
            continue
        for ch in word:
            game.command({"type": "key", "key": ch, "down": True})
            game.command({"type": "key", "key": ch, "down": False})


def main() -> int:
    args = build_arg_parser().parse_args()
    paddle = make_paddle(args)
    if paddle is None:
        return 1

    game_ref = {}
    server = GameServer(os.path.join(HERE, "web"), on_message=lambda m: game_ref["game"].command(m),
                        http_port=args.http_port, ws_port=args.ws_port,
                        runtime={"player_hit_plane_z": C.PLAYER_HIT_PLANE_Z,
                                 "opponent_hit_plane_z": C.OPPONENT_HIT_PLANE_Z,
                                 "spin_max_rads": C.SPIN_MAX_RADS})
    vision = make_vision(args, server)

    def log_and_publish(msg: str) -> None:
        if '"name": "bounce"' not in msg and '"name": "tag_progress"' not in msg:
            print("event:", msg[:200])
        server.publish_event(msg)

    publish_event = log_and_publish if args.debug else server.publish_event

    # Record number of continuous hits -> MQTT (as a float), sent at startup and on every new record.
    scores = ScorePublisher(topic=args.mqtt_topic, enabled=C.MQTT_ENABLED and not args.no_mqtt)
    streak = StreakTracker(default_path(), on_record=scores.publish_record)
    if args.reset_record:
        streak.reset_record()
    scores.publish_record(streak.record)
    if scores.enabled:
        print(f"Publishing the hit record ({float(streak.record)}) to {args.mqtt_topic} on {C.MQTT_BROKER}")

    game = Game(paddle, vision, server.publish_state, publish_event, debug=args.debug, seed=args.seed,
                streak=streak, mqtt_status=lambda: scores.status)
    game_ref["game"] = game

    stop = threading.Event()
    try:
        server.start()
    except (OSError, RuntimeError) as exc:
        print(f"Could not start the server ({exc}). Is another copy of the game running?")
        return 1
    paddle.start()
    if vision:
        vision.start()
    game_thread = threading.Thread(target=game.run, daemon=True, name="game-loop")
    game_thread.start()

    print(__doc__.split("Threads:")[0])
    print(f"Game running at {server.url}  (paddle: {paddle.kind}, camera: {'on' if vision else 'off'})")
    print("Type a key + Enter here, or play in the browser. Ctrl-C or q to quit.\n")
    if not args.no_browser and C.OPEN_BROWSER:
        webbrowser.open(server.url)

    threading.Thread(target=terminal_input, args=(game, stop), daemon=True, name="stdin").start()
    last_status = None
    try:
        while not stop.is_set():
            status = paddle.status
            if status != last_status:
                print(f"[paddle] {status}")
                last_status = status
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        game.stop()
        game_thread.join(timeout=1.0)
        paddle.shutdown()
        if vision:
            vision.stop()
        server.stop()
        scores.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
