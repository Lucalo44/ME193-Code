"""A paddle juggles the ball. Every bounce should send the ball to 1.2 m.

    python 10_paddle.py              # tune the paddle controller, then render the best run
    python 10_paddle.py --no-video   # tune only (handy for quick tests)

The ball is dropped from 1.0 m onto a paddle resting at 0.5 m. The contact is lossy, so a paddle
that just sits there lets the ball die out; it has to strike upward to put energy back in.

Each run is scored with three rewards:
  1. energy leaving the paddle   (+)  ball energy right after it leaves the paddle, as a fraction of
                                      the energy needed to reach 1.2 m (capped at 1, see energy_term)
  2. distance from bounce height (-)  |apex - 1.2 m| for every bounce
  3. paddle movement             (-)  total distance the paddle travels, in m
A cross-entropy search tunes the controller's three numbers to maximize the score.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering; must be set before importing mujoco

import mujoco
import numpy as np

TARGET = 1.2          # m, apex height every bounce should reach (ball center)
BALL_START = 1.0      # m, ball is dropped from here (ball center)
PADDLE_START = 0.5    # m, paddle rest height (paddle center)
EPISODE = 8.0         # s of simulation per scored run
OUT = "paddle.mp4"

# Reward weights. Raise W_MOVE for a lazier paddle, W_HEIGHT for tighter apexes.
W_ENERGY = 1.0        # per bounce
W_HEIGHT = 5.0        # per m of apex error, per bounce
W_MOVE = 0.5          # per m of paddle travel
FAIL_PENALTY = 10.0   # ball hits the floor or slides off the paddle

BALL_R = 0.03
PADDLE_HALF_T = 0.01  # half thickness of the paddle

XML = f"""
<mujoco>
  <option timestep="0.002"/>
  <default>
    <!-- solref = "-stiffness -damping". This damping loses ~25% of the speed per bounce. -->
    <geom solref="-20000 -25"/>
  </default>
  <worldbody>
    <light pos="0 -1 4"/>
    <light pos="0 -3 1" dir="0 1 0" diffuse=".5 .5 .5"/>
    <camera name="side" pos="0 -2.2 0.85" xyaxes="1 0 0 0 0 1"/>
    <geom name="floor" type="plane" size="5 5 .1" rgba=".3 .4 .5 1"/>

    <!-- Target height marker (visual only, no collisions). -->
    <geom name="target" type="box" pos="0 .3 {TARGET}" size=".25 .002 .002"
          rgba=".2 .9 .3 1" contype="0" conaffinity="0"/>

    <!-- Paddle: slides up and down only. The joint range keeps it 5 cm to 0.9 m off the floor.
         gravcomp cancels its weight so the servo only has to accelerate it. -->
    <body name="paddle" pos="0 0 {PADDLE_START}" gravcomp="1">
      <joint name="lift" type="slide" axis="0 0 1" range="-0.45 0.4"/>
      <geom name="paddle" type="box" size=".15 .15 {PADDLE_HALF_T}" mass=".5" rgba=".75 .75 .8 1"/>
    </body>

    <body name="ball" pos="0 0 {BALL_START}">
      <freejoint name="ball"/>
      <geom name="ball" type="sphere" size="{BALL_R}" mass=".05" rgba="1 .3 .2 1"/>
    </body>
  </worldbody>

  <actuator>
    <!-- Velocity servo on the paddle (ctrl = commanded up/down speed, m/s). kv = 200 N s/m on a
         0.5 kg paddle reaches the commanded speed in ~10 ms. -->
    <velocity name="lift" joint="lift" kv="200" ctrlrange="-3 3" forcerange="-100 100"/>
  </actuator>
</mujoco>
"""

# Controller parameters: [trigger gap (m), strike speed (m/s), apex-error gain (m/s per m)]
LOW = np.array([0.00, 0.0, 0.0])
HIGH = np.array([0.20, 2.0, 4.0])
HOME_GAIN = 8.0       # 1/s, how hard the paddle is pulled back to its rest height between hits


class Sim:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        m = self.model
        self.g = -m.opt.gravity[2]
        self.ball_mass = m.body("ball").mass[0]
        self.ball_geom = m.geom("ball").id
        self.paddle_geom = m.geom("paddle").id
        self.floor_geom = m.geom("floor").id
        self.lift_q = m.jnt_qposadr[m.joint("lift").id]
        self.ball_q = m.jnt_qposadr[m.joint("ball").id]     # x y z qw qx qy qz
        self.ball_v = m.jnt_dofadr[m.joint("ball").id]      # vx vy vz wx wy wz

    def touching(self, data):
        """(ball on paddle, ball on floor) for the current contacts."""
        paddle = floor = False
        for c in data.contact[:data.ncon]:
            pair = {c.geom1, c.geom2}
            if self.ball_geom in pair:
                paddle |= self.paddle_geom in pair
                floor |= self.floor_geom in pair
        return paddle, floor

    def energy_term(self, z, vz):
        """Ball energy as it leaves the paddle, over the energy needed to reach TARGET.

        Capped at 1: past that point extra energy only overshoots, and the height reward already
        handles that. Without the cap the two rewards fight and the search learns to overshoot.
        """
        e_out = self.ball_mass * (self.g * z + 0.5 * vz * vz)
        e_target = self.ball_mass * self.g * TARGET
        return min(e_out / e_target, 1.0)

    def run(self, params, renderer=None, fps=30, verbose=False):
        """Simulate one episode with the given controller parameters. Returns (score, info)."""
        trigger, speed, gain = params
        m = self.model
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)

        phase = "wait"                 # wait -> strike -> wait ...
        last_apex = BALL_START
        was_touching = False
        prev_vz = 0.0
        prev_paddle = d.qpos[self.lift_q]
        travel = 0.0
        bounces = []                   # dicts: energy, apex, err
        failed = False
        frames = []

        while d.time < EPISODE:
            z = d.qpos[self.ball_q + 2]
            x, y = d.qpos[self.ball_q], d.qpos[self.ball_q + 1]
            vz = d.qvel[self.ball_v + 2]
            paddle_top = PADDLE_START + d.qpos[self.lift_q] + PADDLE_HALF_T
            gap = z - BALL_R - paddle_top
            on_paddle, on_floor = self.touching(d)

            # Events: ball leaves the paddle, ball reaches its apex.
            if was_touching and not on_paddle:
                bounces.append({"energy": self.energy_term(z, vz), "apex": None})
            if prev_vz > 0 >= vz and not on_paddle and bounces and bounces[-1]["apex"] is None:
                bounces[-1]["apex"] = z
                last_apex = z

            if on_floor or abs(x) > 0.15 or abs(y) > 0.15 or gap < -0.05:
                failed = True
                break

            # Strike controller: hold at rest, swing up when the falling ball gets close (faster if
            # the last apex was low), then ease back to rest once the ball is on its way up.
            if phase == "wait" and vz < 0 and gap < trigger:
                phase = "strike"
                strike_speed = np.clip(speed + gain * (TARGET - last_apex), 0.0, 3.0)
            elif phase == "strike" and not on_paddle and vz > 0 and gap > 0.01:
                phase = "wait"
            if phase == "strike":
                d.ctrl[0] = strike_speed
            else:
                d.ctrl[0] = np.clip(-HOME_GAIN * d.qpos[self.lift_q], -1.0, 1.0)

            was_touching, prev_vz = on_paddle, vz
            mujoco.mj_step(m, d)
            travel += abs(d.qpos[self.lift_q] - prev_paddle)
            prev_paddle = d.qpos[self.lift_q]

            if renderer is not None and d.time >= len(frames) / fps:
                renderer.update_scene(d, camera="side")
                frames.append(renderer.render())

        done = [b for b in bounces if b["apex"] is not None]
        for b in done:
            b["err"] = abs(b["apex"] - TARGET)
        r_energy = W_ENERGY * sum(b["energy"] for b in done)
        r_height = -W_HEIGHT * sum(b["err"] for b in done)
        r_move = -W_MOVE * travel
        score = r_energy + r_height + r_move - (FAIL_PENALTY if failed else 0.0)
        info = dict(bounces=done, travel=travel, failed=failed, frames=frames,
                    r_energy=r_energy, r_height=r_height, r_move=r_move)
        return score, info


def tune(sim, iters=25, pop=40, n_elite=8, seed=0):
    """Cross-entropy search: sample params, keep the best few, refit, repeat."""
    rng = np.random.default_rng(seed)
    mean = (LOW + HIGH) / 2
    std = (HIGH - LOW) / 3
    best, best_score = mean, -np.inf
    for it in range(iters):
        samples = np.clip(rng.normal(mean, std, size=(pop, 3)), LOW, HIGH)
        scores = np.array([sim.run(p)[0] for p in samples])
        elite = samples[np.argsort(scores)[-n_elite:]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-3
        if scores.max() > best_score:
            best_score, best = scores.max(), samples[scores.argmax()]
        print(f"iter {it + 1:2d}: best {scores.max():7.2f}  mean {scores.mean():7.2f}  "
              f"params [trigger {mean[0]:.3f} m, speed {mean[1]:.2f} m/s, gain {mean[2]:.2f}]")
    return best, best_score


def report(score, info):
    print(f"\nscore {score:.2f} = energy {info['r_energy']:.2f} + height {info['r_height']:.2f} "
          f"+ movement {info['r_move']:.2f}" + (f" - fail {FAIL_PENALTY:g}" if info["failed"] else ""))
    print(f"paddle travel {info['travel']:.2f} m, {len(info['bounces'])} bounces")
    for i, b in enumerate(info["bounces"], 1):
        print(f"  bounce {i:2d}: apex {b['apex']:.3f} m  (err {b['err'] * 100:5.1f} cm)  "
              f"energy {b['energy']:.2f}")


def main():
    sim = Sim()

    _, idle = sim.run([0.0, 0.0, 0.0])
    print(f"paddle held still: apexes {[round(float(b['apex']), 2) for b in idle['bounces'][:5]]} m ...\n")

    best, _ = tune(sim)
    print(f"\nbest params: trigger {best[0]:.3f} m, speed {best[1]:.2f} m/s, gain {best[2]:.2f}")

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
