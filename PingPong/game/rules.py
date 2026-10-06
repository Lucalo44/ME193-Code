"""
game/rules.py -- scoring, serve rotation, game and match end.

Games go to POINTS_TO_WIN (11), win by 2. The serve switches every
SERVES_PER_TURN points (5); with DEUCE_SERVE_EVERY_POINT it alternates every
point once both players reach 10. The first server alternates between games.
A match is best of GAMES_PER_MATCH.

`server` decides who actually serves (see PLAYER_SERVE in config.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import config as C

PLAYERS = ("player", "cpu")


def other(who: str) -> str:
    return "cpu" if who == "player" else "player"


@dataclass
class PointResult:
    winner: str
    game_over: bool = False
    game_winner: Optional[str] = None
    match_over: bool = False
    match_winner: Optional[str] = None


class Match:
    def __init__(self, games_per_match: int = C.GAMES_PER_MATCH,
                 points_to_win: int = C.POINTS_TO_WIN, first_server: str = "cpu",
                 serves_per_turn: int = C.SERVES_PER_TURN,
                 deuce_every_point: bool = C.DEUCE_SERVE_EVERY_POINT):
        self.games_per_match = games_per_match
        self.serves_per_turn = serves_per_turn
        self.deuce_every_point = deuce_every_point
        self.points_to_win = points_to_win
        self.match_first_server = first_server
        self.reset()

    def reset(self) -> None:
        self.points = {"player": 0, "cpu": 0}
        self.games = {"player": 0, "cpu": 0}
        self.game_first_server = self.match_first_server
        self.match_winner: Optional[str] = None

    @property
    def games_needed(self) -> int:
        return self.games_per_match // 2 + 1

    @property
    def server(self) -> str:
        p, c = self.points["player"], self.points["cpu"]
        deuce_at = self.points_to_win - 1
        total = p + c
        per = self.serves_per_turn
        if self.deuce_every_point and p >= deuce_at and c >= deuce_at:
            flips = (2 * deuce_at) // per + (total - 2 * deuce_at)
        else:
            flips = total // per
        return self.game_first_server if flips % 2 == 0 else other(self.game_first_server)

    def game_winner(self) -> Optional[str]:
        p, c = self.points["player"], self.points["cpu"]
        if max(p, c) >= self.points_to_win and abs(p - c) >= 2:
            return "player" if p > c else "cpu"
        return None

    def award_point(self, winner: str) -> PointResult:
        if self.match_winner:
            return PointResult(winner, match_over=True, match_winner=self.match_winner)
        self.points[winner] += 1
        gw = self.game_winner()
        if gw is None:
            return PointResult(winner)
        return self._finish_game(gw, winner)

    def award_game(self, winner: str) -> PointResult:
        """End the current game immediately (WRONG_STROKE_PENALTY = "game")."""
        if self.match_winner:
            return PointResult(winner, match_over=True, match_winner=self.match_winner)
        return self._finish_game(winner, winner)

    def _finish_game(self, game_winner: str, point_winner: str) -> PointResult:
        self.games[game_winner] += 1
        result = PointResult(point_winner, game_over=True, game_winner=game_winner)
        if self.games[game_winner] >= self.games_needed:
            self.match_winner = game_winner
            result.match_over = True
            result.match_winner = game_winner
        return result

    def start_next_game(self) -> None:
        self.points = {"player": 0, "cpu": 0}
        self.game_first_server = other(self.game_first_server)

    def snapshot(self) -> dict:
        return {
            "player": self.points["player"],
            "cpu": self.points["cpu"],
            "games_player": self.games["player"],
            "games_cpu": self.games["cpu"],
            "server": self.server,
            "games_needed": self.games_needed,
        }


def required_stroke(x_arrival: float, handedness: str = C.HANDEDNESS,
                    deadband: float = C.CENTER_DEADBAND) -> str:
    """The centerline rule. Right-handed: ball arriving right of the screen
    centerline (x > 0) needs a forehand, left needs a backhand. Inside the
    deadband either stroke is fine. Uses ball position only -- never pose
    landmark names, which are mirrored."""
    if abs(x_arrival) <= deadband:
        return "either"
    right_side = x_arrival > 0
    if handedness == "left":
        right_side = not right_side
    return "forehand" if right_side else "backhand"


def stroke_ok(required: str, judged: Optional[str]) -> bool:
    if judged not in ("forehand", "backhand"):
        return False
    return required == "either" or required == judged
