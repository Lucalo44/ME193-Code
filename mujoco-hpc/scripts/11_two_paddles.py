"""Two paddles rally the ball back and forth. Every hit should send it over to the other paddle
with its apex at 1.2 m.

    python 11_two_paddles.py              # tune the controller, then render the best run
    python 11_two_paddles.py --no-video   # tune only (handy for quick tests)

The paddles rest at 0.5 m, 0.8 m apart. The ball is dropped from 1.0 m onto the left one. Each paddle
can tilt (hinge) and strike along its own normal (slide). A flat strike only sends the ball straight
up, so the paddle has to tilt to send it across, and strike to put back the energy the lossy contact
eats.

Each run is scored with the same rewards as 10_paddle.py, plus one for aim:
  1. energy leaving the paddle   (+)  ball energy right after a hit, as a fraction of the energy
                                      needed to cross with a 1.2 m apex (capped at 1)
  2. distance from bounce height (-)  |apex - 1.2 m| for every hit
  3. paddle movement             (-)  slide travel plus tilt travel (measured at the paddle edge), m
  4. landing distance            (-)  how far from the receiving paddle's center the ball lands, m
Hitting the same paddle twice in a row, or the floor, fails the run.
"""
import math
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering; must be set before importing mujoco

import mujoco
import numpy as np

TARGET = 1.2          # m, apex height every hit should reach (ball center)
BALL_START = 1.0      # m, ball is dropped from here (ball center), above the left paddle
PADDLE_START = 0.5    # m, paddle rest height (paddle center)
D = 0.4               # m, paddle centers sit at x = -D and x = +D
EPISODE = 8.0         # s of simulation per scored run
OUT = "two_paddles.mp4"

# Reward weights.
W_ENERGY = 1.0        # per hit
W_HEIGHT = 5.0        # per m of apex error, per hit
W_LAND = 5.0          # per m of landing error, per hit
W_MOVE = 0.5          # per m of paddle travel
FAIL_PENALTY = 10.0   # floor, wrong paddle, or ball lost

BALL_R = 0.03
PADDLE_HALF = 0.15    # half width of the square paddle
PADDLE_HALF_T = 0.01  # half thickness
TILT_MAX = 0.6        # rad

PADDLE_XML = """
    <!-- {s} paddle: outer body tilts about y, inner body strikes along the paddle normal.
         The slide range keeps the paddle at least 0.15 m off the floor; gravcomp cancels its weight. -->
    <body name="tilt{s}" pos="{x} 0 {z}" gravcomp="1">
      <inertial pos="0 0 0" mass=".05" diaginertia="1e-4 1e-4 1e-4"/>
      <joint name="tilt{s}" type="hinge" axis="0 1 0" range="-{tmax} {tmax}"/>
      <body name="paddle{s}" gravcomp="1">
        <joint name="lift{s}" type="slide" axis="0 0 1" range="-0.25 0.3"/>
        <geom name="paddle{s}" type="box" size="{h} {h} {t}" mass=".5" rgba=".75 .75 .8 1"/>
      </body>
    </body>"""

ACTUATOR_XML = """
    <position name="tilt{s}" joint="tilt{s}" kp="40" kv="0.6" ctrlrange="-{tmax} {tmax}"/>
    <velocity name="lift{s}" joint="lift{s}" kv="200" ctrlrange="-3 3" forcerange="-100 100"/>"""

SIDES = ("L", "R")
_kw = dict(z=PADDLE_START, h=PADDLE_HALF, t=PADDLE_HALF_T, tmax=TILT_MAX)
XML = f"""
<mujoco>
  <compiler angle="radian"/>   <!-- MuJoCo reads joint ranges in degrees unless told otherwise -->
  <option timestep="0.002"/>
  <default>
    <!-- solref = "-stiffness -damping". This damping loses ~25% of the speed per bounce. -->
    <geom solref="-20000 -25"/>
  </default>
  <worldbody>
    <light pos="0 -1 4"/>
    <light pos="0 -3 1" dir="0 1 0" diffuse=".5 .5 .5"/>
    <camera name="side" pos="0 -2.6 0.85" xyaxes="1 0 0 0 0 1"/>
    <geom name="floor" type="plane" size="5 5 .1" rgba=".3 .4 .5 1"/>

    <!-- Target height marker (visual only, no collisions). -->
    <geom name="target" type="box" pos="0 .3 {TARGET}" size=".7 .002 .002"
          rgba=".2 .9 .3 1" contype="0" conaffinity="0"/>
{PADDLE_XML.format(s="L", x=-D, **_kw)}
{PADDLE_XML.format(s="R", x=D, **_kw)}

    <body name="ball" pos="{-D} 0 {BALL_START}">
      <freejoint name="ball"/>
      <geom name="ball" type="sphere" size="{BALL_R}" mass=".05" rgba="1 .3 .2 1"/>
    </body>
  </worldbody>

  <actuator>{ACTUATOR_XML.format(s="L", **_kw)}{ACTUATOR_XML.format(s="R", **_kw)}
  </actuator>
</mujoco>
"""

# Controller parameters:
#   trigger      m    start the strike when the ball is this close to the paddle face
#   e_hat        -    the controller's guess of the contact restitution
#   speed_bias   m/s  added to the computed strike speed (covers servo lag and the soft contact)
#   tilt_scale   -    multiplies the computed tilt angle
NAMES = ("trigger", "e_hat", "speed_bias", "tilt_scale")
LOW = np.array([0.00, 0.4, -0.5, 0.5])
HIGH = np.array([0.10, 1.0, 0.5, 1.5])
HOME_GAIN = 8.0       # 1/s, how hard a paddle is pulled back to its rest height between hits
ARM = PADDLE_HALF     # m, tilt travel is counted as rad * ARM (distance the paddle edge moves)


class Sim:
    def __init__(self):
        self.model = m = mujoco.MjModel.from_xml_string(XML)
        self.g = -m.opt.gravity[2]
        self.ball_mass = m.body("ball").mass[0]
        self.ball_geom = m.geom("ball").id
        self.floor_geom = m.geom("floor").id
        self.paddle_geom = [m.geom(f"paddle{s}").id for s in SIDES]
        self.tilt_q = [m.jnt_qposadr[m.joint(f"tilt{s}").id] for s in SIDES]
        self.lift_q = [m.jnt_qposadr[m.joint(f"lift{s}").id] for s in SIDES]
        self.tilt_u = [m.actuator(f"tilt{s}").id for s in SIDES]
        self.lift_u = [m.actuator(f"lift{s}").id for s in SIDES]
        self.ball_q = m.jnt_qposadr[m.joint("ball").id]     # x y z qw qx qy qz
        self.ball_v = m.jnt_dofadr[m.joint("ball").id]      # vx vy vz wx wy wz
        self.home_x = (-D, D)

        # Nominal rally: ball center meets the paddle at z_c and must rise to TARGET.
        self.z_c = PADDLE_START + PADDLE_HALF_T + BALL_R
        self.vz_out = math.sqrt(2 * self.g * (TARGET - self.z_c))
        self.flight = 2 * self.vz_out / self.g
        vx_nom = 2 * D / self.flight
        self.e_target = self.ball_mass * (self.g * TARGET + 0.5 * vx_nom ** 2)

    def touching(self, d):
        """(set of paddle indices the ball touches, ball on floor)."""
        paddles, floor = set(), False
        for c in d.contact[:d.ncon]:
            pair = (c.geom1, c.geom2)
            if self.ball_geom in pair:
                floor |= self.floor_geom in pair
                for k, gid in enumerate(self.paddle_geom):
                    if gid in pair:
                        paddles.add(k)
        return paddles, floor

    def energy_term(self, z, v):
        """Ball energy just after a hit over the energy a perfect cross needs, capped at 1.

        Past 1 the extra energy only overshoots the apex, which the height reward already punishes.
        """
        e_out = self.ball_mass * (self.g * z + 0.5 * float(v @ v))
        return min(e_out / self.e_target, 1.0)

    def plan(self, k, p, v, e_hat, speed_bias, tilt_scale):
        """Tilt and strike speed for paddle k to send the incoming ball to the other paddle.

        Predicts where and how fast the ball arrives at contact height, picks the outgoing velocity
        that lands on the other paddle with a TARGET apex, tilts the paddle normal along the needed
        velocity change, and solves the restitution law for the strike speed along that normal.
        """
        x, z, vx, vz = p[0], p[2], v[0], v[2]
        t = (vz + math.sqrt(max(vz * vz + 2 * self.g * (z - self.z_c), 0.0))) / self.g
        x_c = x + vx * t
        v_in = np.array([vx, vz - self.g * t])
        v_out = np.array([(self.home_x[1 - k] - x_c) / self.flight, self.vz_out])
        dv = v_out - v_in
        tilt = float(np.clip(tilt_scale * math.atan2(dv[0], dv[1]), -TILT_MAX, TILT_MAX))
        n = np.array([math.sin(tilt), math.cos(tilt)])
        speed = (v_out @ n + e_hat * (v_in @ n)) / (1 + e_hat) + speed_bias
        return tilt, float(np.clip(speed, 0.0, 3.0))

    def run(self, params, renderer=None, fps=30):
        """Simulate one episode with the given controller parameters. Returns (score, info)."""
        trigger, e_hat, speed_bias, tilt_scale = params
        m = self.model
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)

        k = 0                          # the paddle the ball is heading to
        phase = "track"                # track (tilt to aim) -> strike -> track the other paddle
        strike_speed = 0.0
        was_on = set()
        prev_vz = 0.0
        prev_joints = np.array([d.qpos[i] for i in self.lift_q + self.tilt_q])
        travel = 0.0
        hits = []                      # dicts: paddle, land, energy, apex
        land_err = 0.0
        failed = None
        frames = []

        while d.time < EPISODE:
            p = d.qpos[self.ball_q:self.ball_q + 3].copy()
            v = d.qvel[self.ball_v:self.ball_v + 3].copy()
            on, on_floor = self.touching(d)

            c = d.geom_xpos[self.paddle_geom[k]]
            n = d.geom_xmat[self.paddle_geom[k]].reshape(3, 3)[:, 2]
            gap = float((p - c) @ n) - BALL_R - PADDLE_HALF_T

            # Events: ball lands on a paddle, ball leaves a paddle, ball reaches its apex.
            for j in on - was_on:
                if j != k:
                    failed = "hit the wrong paddle"
                land_err = abs(p[0] - self.home_x[j])
            for j in was_on - on:
                hits.append({"paddle": SIDES[j], "land": land_err,
                             "energy": self.energy_term(p[2], v), "apex": None})
            if prev_vz > 0 >= v[2] and not on and hits and hits[-1]["apex"] is None:
                hits[-1]["apex"] = p[2]

            if on_floor:
                failed = "hit the floor"
            elif abs(p[1]) > 0.2 or abs(p[0]) > D + 0.4 or p[2] < 0.3:
                failed = "lost the ball"
            if failed:
                break

            # Receiving paddle: aim while the ball flies in, strike when it gets close, then hand
            # over to the other paddle once the ball is clear.
            if phase == "track":
                tilt, strike_speed = self.plan(k, p, v, e_hat, speed_bias, tilt_scale)
                d.ctrl[self.tilt_u[k]] = tilt
                if gap < trigger and float(v @ n) < 0:
                    phase = "strike"
            elif phase == "strike" and k not in on and float(v @ n) > 0 and gap > 0.01:
                phase, k = "track", 1 - k

            for j in (0, 1):
                if j == k and phase == "strike":
                    d.ctrl[self.lift_u[j]] = strike_speed
                else:   # ease back to rest height, keep the current tilt
                    d.ctrl[self.lift_u[j]] = np.clip(-HOME_GAIN * d.qpos[self.lift_q[j]], -1.0, 1.0)

            was_on, prev_vz = on, v[2]
            mujoco.mj_step(m, d)
            joints = np.array([d.qpos[i] for i in self.lift_q + self.tilt_q])
            dj = np.abs(joints - prev_joints)
            travel += dj[:2].sum() + ARM * dj[2:].sum()
            prev_joints = joints

            if renderer is not None and d.time >= len(frames) / fps:
                renderer.update_scene(d, camera="side")
                frames.append(renderer.render())

        done = [h for h in hits if h["apex"] is not None]
        for h in done:
            h["err"] = abs(h["apex"] - TARGET)
        r_energy = W_ENERGY * sum(h["energy"] for h in done)
        r_height = -W_HEIGHT * sum(h["err"] for h in done)
        r_land = -W_LAND * sum(h["land"] for h in done)
        r_move = -W_MOVE * travel
        score = r_energy + r_height + r_land + r_move - (FAIL_PENALTY if failed else 0.0)
        info = dict(hits=done, travel=travel, failed=failed, frames=frames, r_energy=r_energy,
                    r_height=r_height, r_land=r_land, r_move=r_move)
        return score, info


def fmt(params):
    return ", ".join(f"{n} {x:.3f}" for n, x in zip(NAMES, params))


def tune(sim, iters=25, pop=40, n_elite=8, seed=0):
    """Cross-entropy search: sample params, keep the best few, refit, repeat."""
    rng = np.random.default_rng(seed)
    mean = (LOW + HIGH) / 2
    std = (HIGH - LOW) / 3
    best, best_score = mean, -np.inf
    for it in range(iters):
        samples = np.clip(rng.normal(mean, std, size=(pop, len(LOW))), LOW, HIGH)
        scores = np.array([sim.run(s)[0] for s in samples])
        elite = samples[np.argsort(scores)[-n_elite:]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-3
        if scores.max() > best_score:
            best_score, best = scores.max(), samples[scores.argmax()]
        print(f"iter {it + 1:2d}: best {scores.max():7.2f}  mean {scores.mean():7.2f}  [{fmt(mean)}]")
    return best, best_score


def report(score, info):
    print(f"\nscore {score:.2f} = energy {info['r_energy']:.2f} + height {info['r_height']:.2f} "
          f"+ landing {info['r_land']:.2f} + movement {info['r_move']:.2f}"
          + (f" - fail {FAIL_PENALTY:g} ({info['failed']})" if info["failed"] else ""))
    print(f"paddle travel {info['travel']:.2f} m, {len(info['hits'])} hits")
    for i, h in enumerate(info["hits"], 1):
        print(f"  hit {i:2d} ({h['paddle']}): apex {h['apex']:.3f} m (err {h['err'] * 100:5.1f} cm)  "
              f"landed {h['land'] * 100:4.1f} cm off center  energy {h['energy']:.2f}")


def main():
    sim = Sim()
    best, _ = tune(sim)
    print(f"\nbest params: {fmt(best)}")

    if "--no-video" in sys.argv:
        report(*sim.run(best))
        return

    import imageio
    renderer = mujoco.Renderer(sim.model, 480, 640)
    score, info = sim.run(best, renderer=renderer)
    renderer.close()
    report(score, info)
    imageio.mimwrite(OUT, info["frames"], fps=30)
    print(f"wrote {OUT}: {len(info['frames'])} frames")


if __name__ == "__main__":
    main()
