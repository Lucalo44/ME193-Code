"""Car motion + IMU skills. Every motor skill clamps its args (registry), checks
the cancel flag while it runs, has a hard timeout (executor), and always stops
the wheels in a finally."""
import time

from skills.context import SkillError
from skills.registry import Int, Num, skill

TURN_SPEED = 170              # wheel PWM for open-loop turns
OPEN_LOOP_DEG_PER_S = 180.0   # at TURN_SPEED; calibrate on the floor (docs/CALIBRATION)
YAW_TOL = 3.0
CLOSED_LOOP_SOURCES = {"sim", "imu", "modulino"}   # real sensors (and the simulator's truth)


def _unwrap(prev, new):
    d = (new - prev + 180) % 360 - 180
    return prev + d


@skill("read_imu", "sets {imu.yaw} {imu.pitch} {imu.roll}", ns="imu",
       returns=["yaw", "pitch", "roll"])
def read_imu(ctx):
    r = ctx.imu()
    if not r or r.get("yaw") is None:
        raise SkillError("no IMU connected")
    return {k: r.get(k) for k in ("yaw", "pitch", "roll", "source")}


# {imu.*} placeholders also read the IMU on demand when a message is sent
# (executor.py), so in "parts" plan style the planner never sees read_imu.


@skill("drive", "move forward (speed > 0) or backward (speed < 0)", args={"speed": Int(-100, 100), "seconds": Num(0, 5)},
       motor=True, timeout_s=8)
def drive(ctx, speed, seconds):
    pwm = int(speed * 2.55)
    try:
        ctx.status.update(car={"connected": ctx.hal.connected, "left": pwm, "right": pwm})
        ctx.hal.drive(pwm, pwm)
        ctx.sleep(seconds)
    finally:
        ctx.hal.stop()
        ctx.status.update(car={"connected": ctx.hal.connected, "left": 0, "right": 0})
    return {"speed": speed, "seconds": seconds}


def _turn(ctx, degrees, timeout=8.0):
    imu = ctx.imu()
    try:
        if imu and imu.get("yaw") is not None and imu.get("source") in CLOSED_LOOP_SOURCES:
            start = yaw = float(imu["yaw"])
            target = start + degrees
            t0 = time.monotonic()
            while True:
                err = target - yaw
                if abs(err) <= YAW_TOL:
                    break
                if time.monotonic() - t0 > timeout:
                    raise SkillError(f"turn timed out {err:.0f} deg short")
                pwm = max(90, min(200, abs(err) * 3.0))
                sgn = 1 if err > 0 else -1
                ctx.hal.drive(-sgn * pwm, sgn * pwm)      # +yaw = counter-clockwise
                ctx.sleep(0.02)
                yaw = _unwrap(yaw, float(ctx.imu()["yaw"]))
            ctx.hal.stop()
            ctx.sleep(0.1)
            final = float(ctx.imu()["yaw"])
            return {"yaw": round(final % 360, 1), "turned": round(_unwrap(start, final) - start, 1),
                    "mode": "closed-loop"}
        sgn = 1 if degrees > 0 else -1
        ctx.hal.drive(-sgn * TURN_SPEED, sgn * TURN_SPEED)
        ctx.sleep(abs(degrees) / OPEN_LOOP_DEG_PER_S)
        return {"yaw": None, "turned": degrees, "mode": "open-loop"}
    finally:
        ctx.hal.stop()


@skill("turn_by", "turn or rotate in place; left > 0, right < 0", args={"degrees": Int(-360, 360)}, ns="imu",
       motor=True, timeout_s=10)
def turn_by(ctx, degrees):
    return _turn(ctx, degrees)


@skill("spin", "spin around in place", args={"times": Int(1, 3)}, motor=True, timeout_s=20)
def spin(ctx, times):
    for _ in range(times):
        _turn(ctx, 360, timeout=6)
    return {"spins": times}


@skill("stop", "stop moving")
def stop(ctx):
    ctx.hal.stop()
    if ctx.conductor:
        ctx.conductor.stop_all()
