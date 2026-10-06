"""
game/state_machine.py -- the game's phases and which transitions are legal.

    LOBBY --(tag confirmed)--> SERVE --> RALLY --(miss/out/net)--> POINT_OVER --(1.5 s)--> SERVE
                                                           \\--(match ends)--> GAME_OVER --(tag)--> SERVE

LATENCY_CAL is a side trip from LOBBY / GAME_OVER for the latency
calibration screen. Pausing is a flag on top of any phase, not a phase.
"""

from __future__ import annotations

LOBBY = "LOBBY"
SERVE = "SERVE"
RALLY = "RALLY"
POINT_OVER = "POINT_OVER"
GAME_OVER = "GAME_OVER"
LATENCY_CAL = "LATENCY_CAL"

TRANSITIONS = {
    LOBBY: {SERVE, LATENCY_CAL},
    SERVE: {RALLY, LOBBY},
    RALLY: {POINT_OVER, LOBBY},
    POINT_OVER: {SERVE, GAME_OVER, LOBBY},
    GAME_OVER: {SERVE, LOBBY, LATENCY_CAL},
    LATENCY_CAL: {LOBBY},
}

TAG_PHASES = {LOBBY, GAME_OVER}


class StateMachine:
    def __init__(self, now: float = 0.0):
        self.phase = LOBBY
        self.entered_at = now
        self.paused = False

    def to(self, phase: str, now: float) -> None:
        if phase == self.phase:
            return
        if phase not in TRANSITIONS[self.phase]:
            raise ValueError(f"illegal transition {self.phase} -> {phase}")
        self.phase = phase
        self.entered_at = now

    def reset(self, now: float) -> None:
        """'r' key: back to the lobby from anywhere."""
        self.phase = LOBBY
        self.entered_at = now
        self.paused = False

    def elapsed(self, now: float) -> float:
        return now - self.entered_at

    @property
    def tags_active(self) -> bool:
        """AprilTags are only read in the lobby and after a match, so a tag
        accidentally in frame mid-rally does nothing."""
        return self.phase in TAG_PHASES
