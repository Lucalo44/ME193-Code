"""Continuous-hit record: counting, persistence, MQTT payload, and the game loop hook."""

import json

import config as C
from game import state_machine as SM
from game.streak import StreakTracker
from server.mqtt_publisher import ScorePublisher
from tests.test_game_loop import advance, make_game, swing_at


def test_streak_counts_and_resets_and_records():
    published = []
    s = StreakTracker(on_record=published.append)
    for _ in range(3):
        s.hit()
    s.reset_streak()
    s.hit()
    assert s.snapshot() == {"current": 1, "record": 3}
    assert published == [1, 2, 3]          # only new records are published


def test_record_persists(tmp_path):
    path = str(tmp_path / "rec.json")
    s = StreakTracker(path)
    for _ in range(4):
        s.hit()
    assert json.load(open(path)) == {"record": 4}
    assert StreakTracker(path).record == 4


def test_payload_is_a_float():
    assert ScorePublisher.payload(7) == "7.0"
    assert float(ScorePublisher.payload(0)) == 0.0


def test_disabled_publisher_never_touches_network():
    p = ScorePublisher(enabled=False)
    p.publish_record(5)
    assert p.status == "disabled"


def test_game_counts_returns_that_land():
    published = []
    game, clock, events = make_game(seed=7)
    game.streak = StreakTracker(on_record=published.append)
    game.start_game("medium")
    for _ in range(400):
        if game.sm.phase == SM.RALLY and game.incoming and not game.incoming.resolved and game.pending is None:
            inc = game.incoming
            advance(game, clock, max(0.0, inc.t - game.sim_t - 0.03))
            swing_at(game, inc.t)
        advance(game, clock, 0.05)
    assert game.streak.record >= 1
    assert published and published[-1] == game.streak.record
    assert any(e["name"] == "record" for e in events)


def test_streak_survives_points_won_and_breaks_on_points_lost():
    game, clock, events = make_game()
    game.streak = StreakTracker()
    game.start_game("medium")
    for _ in range(3):
        game.streak.hit()
    game.sm.phase = SM.RALLY
    game._end_point("player", "CPU MISSED")      # player wins the point
    assert game.streak.current == 3
    game.sm.phase = SM.RALLY
    game._end_point("cpu", "NO SWING")           # player loses the point
    assert game.streak.snapshot() == {"current": 0, "record": 3}


def test_record_starts_over_with_each_new_game():
    published = []
    game, clock, events = make_game(seed=7)
    game.streak = StreakTracker(on_record=published.append)
    game.streak.record = game.streak.current = 5            # left over from the last game
    game.start_game("medium")
    assert game.streak.snapshot() == {"current": 0, "record": 0}
    assert published == [0]                                 # the reset is published too


def test_score_is_published_at_game_start_and_after_every_point():
    scores = []
    game, clock, events = make_game()
    game._on_score = scores.append
    game.start_game("medium")
    assert scores[0]["event"] == "game_start" and (scores[0]["player"], scores[0]["cpu"]) == (0, 0)
    advance(game, clock, 30.0)                               # no swings: the CPU wins points
    points = [s for s in scores if s["event"] == "point"]
    assert len(points) == len([e for e in events if e["name"] == "point"]) >= 2
    assert points[-1]["cpu"] == game.match.points["cpu"] or points[-1]["game_over"]
    assert all(s["winner"] == "cpu" and s["reason"] for s in points)


def test_score_payload_is_compact_json():
    msg = ScorePublisher.score_payload({"event": "point", "player": 3, "cpu": 2})
    assert json.loads(msg) == {"event": "point", "player": 3, "cpu": 2} and " " not in msg


def test_disabled_publisher_ignores_scores():
    p = ScorePublisher(enabled=False)
    p.publish_score({"player": 1, "cpu": 0})
    assert p.score_topic == C.MQTT_TOPIC + "/score" and p.status == "disabled"
