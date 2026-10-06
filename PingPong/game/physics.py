"""
game/physics.py -- ball flight, bounces, net, and the shot-aiming solver.

World frame (see config.py): meters, origin at table center, y up, player at
negative z. Spin is an angular-velocity vector in rad/s. Topspin for a ball
moving in direction v is spin along (y_hat x v_hat); positive y spin curves a
ball moving +z toward +x.

Everything here is plain-Python floats rather than numpy: these are 3-vectors
stepped a few hundred times per prediction, and per-call numpy overhead would
dominate.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import config as C

Vec = Tuple[float, float, float]

HALF_L = C.TABLE_LENGTH / 2.0
HALF_W = C.TABLE_WIDTH / 2.0
NET_HALF_SPAN = HALF_W + C.NET_OVERHANG
DT = 1.0 / C.PHYSICS_HZ


# --------------------------------------------------------------------------
# Tiny vector helpers
# --------------------------------------------------------------------------

def v_add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_scale(a: Vec, s: float) -> Vec:
    return (a[0] * s, a[1] * s, a[2] * s)


def v_cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def v_norm(a: Vec) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def v_unit(a: Vec) -> Vec:
    n = v_norm(a)
    return (0.0, 0.0, 0.0) if n < 1e-9 else (a[0] / n, a[1] / n, a[2] / n)


def side_of(z: float) -> str:
    """Which half of the table a z coordinate is on."""
    return "player" if z < 0 else "cpu"


def on_table(x: float, z: float) -> bool:
    return abs(x) <= HALF_W and abs(z) <= HALF_L


def spin_vector(vel: Vec, topspin_rads: float, sidespin_rads: float) -> Vec:
    """Build a spin vector from signed topspin (+) / backspin (-) and sidespin
    components, relative to the ball's direction of travel."""
    horiz = v_unit((vel[0], 0.0, vel[2]))
    top_axis = v_cross((0.0, 1.0, 0.0), horiz)
    return v_add(v_scale(top_axis, topspin_rads), (0.0, sidespin_rads, 0.0))


def topspin_component(vel: Vec, spin: Vec) -> float:
    """Inverse of spin_vector's topspin part (positive = topspin)."""
    top_axis = v_cross((0.0, 1.0, 0.0), v_unit((vel[0], 0.0, vel[2])))
    return spin[0] * top_axis[0] + spin[1] * top_axis[1] + spin[2] * top_axis[2]


# --------------------------------------------------------------------------
# Ball state and integration
# --------------------------------------------------------------------------

@dataclass
class Ball:
    pos: Vec = (0.0, 0.3, 0.0)
    vel: Vec = (0.0, 0.0, 0.0)
    spin: Vec = (0.0, 0.0, 0.0)
    dead: bool = False

    def copy(self) -> "Ball":
        return Ball(self.pos, self.vel, self.spin, self.dead)


@dataclass
class PhysicsEvent:
    kind: str                     # "bounce" | "net" | "floor"
    pos: Vec
    side: str = ""                # for bounces: "player" or "cpu"
    extra: dict = field(default_factory=dict)


def acceleration(vel: Vec, spin: Vec) -> Vec:
    speed = v_norm(vel)
    drag = v_scale(vel, -C.DRAG_COEFF * speed)
    magnus = v_scale(v_cross(spin, vel), C.MAGNUS_COEFF)
    return (drag[0] + magnus[0], drag[1] + magnus[1] - C.GRAVITY, drag[2] + magnus[2])


def step(ball: Ball, dt: float = DT, rng: Optional[random.Random] = None,
         collide: bool = True) -> List[PhysicsEvent]:
    """Advance the ball one step in place. Returns any collision events.
    With collide=False the table and net are ignored (used by the aim solver)."""
    if ball.dead:
        return []
    rng = rng or random
    events: List[PhysicsEvent] = []
    a = acceleration(ball.vel, ball.spin)
    vx, vy, vz = ball.vel[0] + a[0] * dt, ball.vel[1] + a[1] * dt, ball.vel[2] + a[2] * dt
    x0, y0, z0 = ball.pos
    x, y, z = x0 + vx * dt, y0 + vy * dt, z0 + vz * dt
    decay = math.exp(-C.SPIN_DECAY_PER_S * dt)
    spin = v_scale(ball.spin, decay)

    if collide:
        # --- net: the plane z = 0 between the posts ---
        if (z0 < 0) != (z < 0) and z0 != 0:
            f = z0 / (z0 - z)
            xc, yc = x0 + (x - x0) * f, y0 + (y - y0) * f
            if abs(xc) <= NET_HALF_SPAN and 0 <= yc < C.NET_HEIGHT + C.BALL_RADIUS:
                if yc < C.NET_HEIGHT - C.BALL_RADIUS * 0.5:
                    # Into the net face: bounce back weakly and drop.
                    vz = -vz * C.NET_FACE_RESTITUTION
                    vx *= 0.4
                    vy = min(vy, 0.0) * 0.5
                    z = math.copysign(C.BALL_RADIUS, z0)
                    events.append(PhysicsEvent("net", (xc, yc, 0.0), extra={"cord": False}))
                else:
                    # Clipped the top cord: dribbles over at low speed.
                    vz *= rng.uniform(0.25, 0.5)
                    vy = abs(vy) * 0.3 + 0.4
                    vx *= 0.6
                    events.append(PhysicsEvent("net", (xc, yc, 0.0), extra={"cord": True}))

        # --- table surface ---
        if y <= C.BALL_RADIUS and vy < 0 and y0 >= C.BALL_RADIUS - 0.03 and on_table(x, z):
            y = C.BALL_RADIUS
            vy = -vy * C.RESTITUTION
            # Spin kick: topspin drives the ball forward, backspin checks it.
            kick = v_scale(v_cross(spin, (0.0, 1.0, 0.0)), C.SPIN_BOUNCE_COEFF * C.BALL_RADIUS)
            vx = vx * C.TABLE_FRICTION_KEEP + kick[0]
            vz = vz * C.TABLE_FRICTION_KEEP + kick[2]
            spin = v_scale(spin, C.SPIN_BOUNCE_KEEP)
            if abs(vy) < 0.25:
                vy = 0.0  # rolling
            events.append(PhysicsEvent("bounce", (x, y, z), side=side_of(z)))

        # --- floor: ball is dead ---
        if y <= C.FLOOR_Y + C.BALL_RADIUS:
            y = C.FLOOR_Y + C.BALL_RADIUS
            ball.dead = True
            vx = vy = vz = 0.0
            events.append(PhysicsEvent("floor", (x, y, z)))

    ball.pos = (x, y, z)
    ball.vel = (vx, vy, vz)
    ball.spin = spin
    return events


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------

@dataclass
class Landing:
    pos: Vec                      # where the ball comes down to table height
    t: float                      # seconds from launch
    net_clearance: float          # ball bottom minus net top when crossing z=0 (inf if it never crosses)


def plane_landing(ball: Ball, max_t: float = 3.0) -> Landing:
    """Fly the ball with no table/net collisions until it descends to table
    height. Monotone in launch angle, which is what the aim solver needs."""
    b = ball.copy()
    t = 0.0
    clearance = math.inf
    while t < max_t:
        z0 = b.pos[2]
        step(b, DT, collide=False)
        t += DT
        z1 = b.pos[2]
        if (z0 < 0) != (z1 < 0):
            clearance = b.pos[1] - C.BALL_RADIUS - C.NET_HEIGHT
        if b.pos[1] <= C.BALL_RADIUS and b.vel[1] < 0:
            return Landing(b.pos, t, clearance)
    return Landing(b.pos, t, clearance)


@dataclass
class Arrival:
    t: float                      # seconds from now
    pos: Vec
    vel: Vec


def predict_receive(ball: Ball, receiver: str, plane_z: float, max_t: float = 3.0,
                    rng: Optional[random.Random] = None, bounced: bool = False,
                    serve: bool = False) -> Optional[Arrival]:
    """Forward-simulate a ball heading toward `receiver` and return where/when
    it can be struck: the first time after it bounces on the receiver's half
    that it reaches the receiver's hit plane, or (for short balls) the moment
    just before it would bounce a second time. None if the shot is not good
    (net, out, bounces on the hitter's side). Pass bounced=True if the ball
    has already bounced on the receiver's side, serve=True for a serve that
    still has to bounce once on the server's side first."""
    b = ball.copy()
    # A fixed seed keeps predictions reproducible through net-cord randomness.
    rng = rng or random.Random(0)
    t = 0.0
    toward = -1.0 if receiver == "player" else 1.0
    while t < max_t and not b.dead:
        events = step(b, DT, rng)
        t += DT
        for ev in events:
            if ev.kind == "bounce":
                if ev.side != receiver and serve and not bounced:
                    serve = False          # the serve's bounce on the server's side
                    continue
                if ev.side != receiver or bounced or serve:
                    return None
                bounced = True
            elif ev.kind == "floor":
                return None
        if bounced:
            past_plane = (b.pos[2] - plane_z) * toward >= 0
            about_to_rebounce = b.vel[1] < 0 and b.pos[1] < 0.07
            if past_plane or about_to_rebounce:
                return Arrival(t, b.pos, b.vel)
    return None


# --------------------------------------------------------------------------
# Shot solver
# --------------------------------------------------------------------------

def launch_velocity(speed: float, theta: float, heading: Tuple[float, float]) -> Vec:
    c = math.cos(theta)
    return (speed * c * heading[0], speed * math.sin(theta), speed * c * heading[1])


def _bisect(fn: Callable[[float], float], lo: float, hi: float, iters: int = 18) -> float:
    """Root of an increasing function on [lo, hi]; clamps to the ends."""
    flo, fhi = fn(lo), fn(hi)
    if flo >= 0:
        return lo
    if fhi <= 0:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


THETA_MIN = math.radians(-20)
THETA_MAX = math.radians(50)


@dataclass
class Shot:
    vel: Vec
    spin: Vec
    theta: float
    target: Tuple[float, float]


def aim(start: Vec, target_xz: Tuple[float, float], speed: float,
        topspin_rads: float = 0.0, sidespin_rads: float = 0.0,
        spin_aware: bool = True, net_margin: float = C.NET_MARGIN) -> Shot:
    """Pick a launch angle so a ball hit from `start` at `speed` lands at
    `target_xz`, never lower than needed to clear the net.

    spin_aware=False solves as if the ball had no spin and then applies the
    spin anyway -- topspin then makes the ball dip short, backspin floats it
    long. That's how player returns are made, so spin matters."""
    dx, dz = target_xz[0] - start[0], target_xz[1] - start[2]
    dist = math.hypot(dx, dz)
    heading = (dx / dist, dz / dist) if dist > 1e-6 else (0.0, 1.0)

    def make(theta: float, with_spin: bool) -> Ball:
        vel = launch_velocity(speed, theta, heading)
        spin = spin_vector(vel, topspin_rads, sidespin_rads) if with_spin else (0.0, 0.0, 0.0)
        return Ball(start, vel, spin)

    def range_err(theta: float) -> float:
        land = plane_landing(make(theta, spin_aware))
        along = (land.pos[0] - start[0]) * heading[0] + (land.pos[2] - start[2]) * heading[1]
        return along - dist

    def net_err(theta: float) -> float:
        return plane_landing(make(theta, spin_aware)).net_clearance - net_margin

    theta = _bisect(range_err, THETA_MIN, THETA_MAX)
    crosses_net = (start[2] < 0) != (target_xz[1] < 0)
    if crosses_net and net_err(theta) < 0:
        theta = _bisect(net_err, theta, THETA_MAX)
    vel = launch_velocity(speed, theta, heading)
    return Shot(vel, spin_vector(vel, topspin_rads, sidespin_rads), theta, target_xz)


def player_return(start: Vec, speed: float, topspin: float, sidespin: float,
                  rng: random.Random, assist: float = C.ASSIST_LEVEL) -> Shot:
    """The player's return: aimed spin-blind into the opponent's half, then
    nudged back toward the table by `assist` (0 = none, 1 = lands whenever
    physically possible -- a max-speed full-backspin ball floats long regardless)."""
    top_rads = topspin * C.SPIN_MAX_RADS
    side_rads = sidespin * C.SIDESPIN_MAX_RADS
    tx = max(-0.55, min(0.55, rng.uniform(-0.45, 0.45) + 0.25 * sidespin))
    tz = C.PLAYER_TARGET_Z + rng.uniform(-0.2, 0.15)
    shot = aim(start, (tx, tz), speed, top_rads, side_rads, spin_aware=False)
    if assist <= 0:
        return shot

    land = plane_landing(Ball(start, shot.vel, shot.spin))
    good = on_table(land.pos[0], land.pos[2]) and land.pos[2] > 0 and land.net_clearance > 0
    if good:
        return shot
    # Pull the aim toward a spin-aware solution, and shift the lateral aim to
    # cancel sidespin curl, by `assist`.
    corrected_x = tx
    if abs(land.pos[0]) > HALF_W - 0.05:
        corrected_x = tx + assist * (math.copysign(HALF_W - 0.15, land.pos[0]) - land.pos[0])
    exact = aim(start, (corrected_x, tz), speed, top_rads, side_rads, spin_aware=True)
    theta = shot.theta + assist * (exact.theta - shot.theta)
    hx0, hz0 = v_unit((shot.vel[0], 0.0, shot.vel[2]))[0::2]
    hx1, hz1 = v_unit((exact.vel[0], 0.0, exact.vel[2]))[0::2]
    heading = v_unit((hx0 + assist * (hx1 - hx0), 0.0, hz0 + assist * (hz1 - hz0)))
    vel = launch_velocity(speed, theta, (heading[0], heading[2]))
    return Shot(vel, spin_vector(vel, top_rads, side_rads), theta, (corrected_x, tz))


# --------------------------------------------------------------------------
# Serves
# --------------------------------------------------------------------------

def serve_result(ball: Ball, server: str, max_t: float = 3.0) -> str:
    """Fly a serve and classify it: "good" (bounces once on the server's side,
    then on the receiver's), "let" (good, but touched the net), or "fault"."""
    b = ball.copy()
    rng = random.Random(0)
    own = False
    net = False
    t = 0.0
    while t < max_t and not b.dead:
        for ev in step(b, DT, rng):
            if ev.kind == "net":
                net = True
            elif ev.kind == "bounce":
                if ev.side == server and not own:
                    own = True
                elif ev.side != server and own:
                    return "let" if net else "good"
                else:
                    return "fault"
            elif ev.kind == "floor":
                return "fault"
        t += DT
    return "fault"


def serve_shot(start: Vec, speed: float, topspin: float, sidespin: float, server: str,
               rng: random.Random, assist: float = C.ASSIST_LEVEL) -> Shot:
    """A serve: aimed to bounce first on the server's own half (spin-blind, like
    player returns), then nudged toward a legal serve by `assist`."""
    top_rads = topspin * C.SPIN_MAX_RADS * C.SERVE_SPIN_SCALE
    side_rads = sidespin * C.SIDESPIN_MAX_RADS * C.SERVE_SPIN_SCALE
    sgn = -1.0 if server == "player" else 1.0           # the server's half has this sign of z
    tx = max(-0.5, min(0.5, start[0] * 0.5 + rng.uniform(-0.2, 0.2)))
    nominal = aim(start, (tx, sgn * C.SERVE_OWN_BOUNCE_Z), speed, top_rads, side_rads, spin_aware=False)
    if assist <= 0 or serve_result(Ball(start, nominal.vel, nominal.spin), server) == "good":
        return nominal
    # Search own-side bounce depths (nearest the nominal first), then slightly
    # different speeds, for a legal serve.
    depths = sorted([0.25 + 0.05 * i for i in range(20)], key=lambda z: abs(z - C.SERVE_OWN_BOUNCE_Z))
    for factor in (1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6):
        for z in depths:
            fix = aim(start, (tx, sgn * z), speed * factor, top_rads, side_rads, spin_aware=True)
            if serve_result(Ball(start, fix.vel, fix.spin), server) == "good":
                vel = tuple(n + assist * (f - n) for n, f in zip(nominal.vel, fix.vel))
                return Shot(vel, spin_vector(vel, top_rads, side_rads), fix.theta, (tx, sgn * z))
    return nominal


def toss_contact(start: Vec, height: float, contact_y: float) -> Tuple[Vec, float]:
    """Vertical serve toss from `start` rising `height` m. Returns the launch
    velocity and the time it falls back to `contact_y` (when it should be hit)."""
    vy = math.sqrt(2 * C.GRAVITY * height)
    b = Ball(start, (0.0, vy, 0.0))
    t = 0.0
    while t < 3.0:
        step(b, DT, collide=False)
        t += DT
        if b.vel[1] < 0 and b.pos[1] <= contact_y:
            break
    return (0.0, vy, 0.0), t
