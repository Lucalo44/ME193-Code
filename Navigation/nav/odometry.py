"""Step-based dead reckoning: turns step events + heading into the walk state the policy sees.

Position is steps x heading (x, y in steps, for the map). Turns are detected from heading
history: a change of more than TURN_DEG over TURN_WINDOW steps, outside a short refractory
period, is one turn. The running count of left and right turns is the policy's best clue for
"where along the route am I", since step counts vary between walkers.
"""
import math

import config as C


class Odometry:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.x = self.y = 0.0
        self.headings = []             # heading (deg) at every step
        self.turns = []                # (step, +1 left / -1 right)
        self.last_turn_step = -10**6
        self.stairs_steps = 0
        self.stairs_segments = 0
        self._on_stairs = False
        self.h0 = None
        self.step_times = []

    def on_step(self, t, heading_deg, p_stairs=0.0):
        """Advances one step; returns the state dict for this step."""
        if self.h0 is None:
            self.h0 = heading_deg
        h = heading_deg - self.h0
        self.steps += 1
        self.headings.append(h)
        self.step_times.append(t)
        rad = math.radians(h)
        self.x += math.cos(rad)
        self.y += math.sin(rad)

        turned = 0
        if len(self.headings) > C.TURN_WINDOW and self.steps - self.last_turn_step > C.TURN_REFRACTORY:
            dh = h - self.headings[-1 - C.TURN_WINDOW]
            if abs(dh) >= C.TURN_DEG:
                turned = 1 if dh > 0 else -1
                self.turns.append((self.steps, turned))
                self.last_turn_step = self.steps

        on = p_stairs > 0.5
        if on:
            self.stairs_steps += 1
        if on and not self._on_stairs:
            self.stairs_segments += 1
        self._on_stairs = on
        return self.state(t, p_stairs, turned)

    def dh(self, n):
        """Heading change over the last n steps (deg)."""
        if len(self.headings) <= n:
            return self.headings[-1] - self.headings[0] if self.headings else 0.0
        return self.headings[-1] - self.headings[-1 - n]

    def cadence(self, n=8):
        if len(self.step_times) <= n:
            return 0.0
        dt = self.step_times[-1] - self.step_times[-1 - n]
        return n / dt if dt > 0 else 0.0

    def state(self, t, p_stairs=0.0, turned=0):
        return {
            "t": t, "steps": self.steps, "heading": self.headings[-1] if self.headings else 0.0,
            "x": self.x, "y": self.y, "dh3": self.dh(3), "dh8": self.dh(8),
            "turned": turned,
            "n_left": sum(1 for _, s in self.turns if s > 0),
            "n_right": sum(1 for _, s in self.turns if s < 0),
            "since_turn": self.steps - self.last_turn_step if self.turns else self.steps,
            "last_turn": self.turns[-1][1] if self.turns else 0,
            "p_stairs": p_stairs, "stairs_steps": self.stairs_steps,
            "stairs_segments": self.stairs_segments, "cadence": self.cadence(),
        }
