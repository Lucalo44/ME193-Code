"""Campus Navigator -- UNO Q Linux side.

Every ~0.1 s: drain the IMU queue from the sketch and feed it through the navigator
(heading filter -> StepNet steps -> stairs/activity -> odometry -> behavior-cloned policy).
A web server on port 8080 serves:
    http://<board>:8080/       the phone page (navigate / record a demo / DAgger)
    http://<board>:8080/dash   the laptop dashboard (raw IMU, movement, live decision view)
Walks are saved under data/walks/ here; Navigation/tools/pull_walks.py brings them to the laptop.

Edit this in Navigation/app/python/; Navigation/tools/sync_app.py assembles the deployable app
in ArduinoUNOq/navigation/ (nav/, gaitid/, web/, config.py and models/ go next to this file).
"""
import time

from arduino.app_utils import App, Bridge

from nav.display import frame
from nav.imu import parse_drain
from nav.server import NavServer
from nav.session import Session

PORT = 8080
POLL_S = 0.1

session = Session()
server = NavServer(session, PORT)
last_icon, last_icon_t, errors = None, 0.0, 0


def show(icon):
    try:
        Bridge.call("draw", bytes(frame(icon)))    # the sketch takes bytes, not a list
    except Exception as e:
        print("draw failed:", e)


def loop():
    global last_icon, last_icon_t, errors
    t0 = time.monotonic()
    try:
        rows = parse_drain(Bridge.call("get_imu"))
        errors = 0
    except Exception as e:
        errors += 1
        if errors in (1, 20):
            print("get_imu failed:", e)
        time.sleep(0.5)
        return
    for rec in session.feed(rows):
        if rec["i"] % 50 == 0:
            st = rec["state"]
            print(f"[nav] step {rec['i']}  heading {st['heading']:.0f}  turns L{st['n_left']} R{st['n_right']}  "
                  f"{rec['movement']}  {session.instruction['primary']}")
    icon = session.matrix_icon()
    if icon != last_icon or time.monotonic() - last_icon_t > 10:     # resend now and then
        show(icon)
        last_icon, last_icon_t = icon, time.monotonic()
    time.sleep(max(0.0, POLL_S - (time.monotonic() - t0)))


App.run(user_loop=loop)
