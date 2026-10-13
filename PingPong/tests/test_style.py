"""Style opponents: a profile built from recorded games, and an opponent that
plays from it (game/style.py)."""

import json
import os
import random

import config as C
from game import physics as P
from game.recorder import GameRecorder
from game.style import StyleOpponent, build_profile


def write_session(root, returns=(), misses=(), serves=(), hand="left", lobby=None, player="p1"):
    """A recorded session with the given returns [(stroke, in_speed, speed, topspin, result, land_xz)].
    hand: the player's own hand; lobby: the hand picked in the game (default: the same)."""
    lobby = lobby or hand
    rec = GameRecorder(player, root=str(root), meta={"hand": hand})
    rec.open()
    rec.event("game_start", 0.0, 0.0, {"speed": "slow", "handedness": lobby})
    sid = 0
    for stroke, in_speed, speed, top, result, land in returns:
        sid += 1
        rec.swing({"id": sid, "t": float(sid), "sim_t": float(sid), "outcome": "hit", "topspin": top, "sidespin": 0.0,
                   "handedness": lobby, "ball": {"required": stroke, "vel": [0.0, 0.0, -in_speed]}})
        rec.event("player_shot", 0.0, float(sid), {"swing_id": sid, "serve": False, "stroke": stroke, "speed": speed,
                                                  "ball_vel": [0.0, 0.0, -in_speed]})
        rec.event("shot_result", 0.0, float(sid), {"swing_id": sid, "result": result,
                                                  "pos": [land[0], 0.02, land[1]] if land else [0, 0, 0]})
    for speed, top in serves:
        sid += 1
        rec.swing({"id": sid, "t": float(sid), "sim_t": float(sid), "outcome": "serve_hit", "topspin": top, "sidespin": 0.0,
                   "handedness": lobby})
        rec.event("player_shot", 0.0, float(sid), {"swing_id": sid, "serve": True, "speed": speed})
        rec.event("shot_result", 0.0, float(sid), {"swing_id": sid, "result": "in", "pos": [0, 0.02, 0.5]})
    for stroke, in_speed, disputed in misses:
        sid += 1
        rec.event("missed_ball", 0.0, float(sid), {"required": stroke, "serve": False, "ball_vel": [0, 0, -in_speed]})
        if disputed:
            rec.event("dispute", 0.0, float(sid), {"reason": "NO SWING"})
    rec.close()


def test_profile_counts_returns_serves_and_undisputed_misses(tmp_path):
    write_session(tmp_path,
                  returns=[("forehand", 3.0, 8.0, 0.8, "in", (0.2, 0.6)), ("backhand", 4.5, 6.5, -0.5, "OUT", None)],
                  misses=[("forehand", 3.0, False), ("forehand", 3.0, True)],
                  serves=[(5.0, 0.3)])
    prof = build_profile("p1", log_root=str(tmp_path))
    assert prof["hand"] == "left" and len(prof["returns"]) == 2 and len(prof["serves"]) == 1
    fh = next(r for r in prof["returns"] if r["stroke"] == "forehand")
    assert fh["land"] == [0.2, 0.6] and fh["speed"] == 8.0 and fh["topspin"] == 0.8
    assert prof["faced"]["forehand/0"] == [2, 1]           # the disputed NO SWING doesn't count
    assert build_profile("nobody", log_root=str(tmp_path)) is None


def profile(returns, hand="left", faced=None):
    return {"player": "p1", "hand": hand, "returns": returns, "serves": [], "faced": faced or {}}


def ret(stroke, land, result="in", speed=6.5, top=0.5):
    return {"stroke": stroke, "in_speed": 4.0, "speed": speed, "topspin": top, "sidespin": 0.0,
            "result": result, "land": land}


def strike_from(bot, x):
    ball = P.Ball((x, 0.15, C.OPPONENT_HIT_PLANE_Z), (0.0, -0.5, 4.0), (0.0, 0.0, 0.0))
    bot.x = x
    shot = bot.strike(ball, 0.0)
    return shot, P.plane_landing(P.Ball(ball.pos, shot.vel, shot.spin))


def test_left_hander_has_its_forehand_on_the_other_side():
    lefty = StyleOpponent(random.Random(0), profile([], hand="left"))
    righty = StyleOpponent(random.Random(0), profile([], hand="right"))
    lefty.x = righty.x = 0.0
    assert lefty._stroke_for(0.3) == "forehand" and righty._stroke_for(0.3) == "backhand"


def test_returns_copy_the_players_shot_turned_around():
    # Every recorded forehand landed deep on the player's right (+x from their end).
    bot = StyleOpponent(random.Random(1), profile([ret("forehand", [0.4, 1.0])]))
    for _ in range(10):
        shot, land = strike_from(bot, 0.3)                    # lefty: +x is its forehand
        assert shot is not None
        assert P.on_table(land.pos[0], land.pos[2]) and land.pos[2] < 0
        assert abs(land.pos[0] - (-0.4)) < 0.15 and abs(land.pos[2] - (-1.0)) < 0.15


def test_failed_returns_fail_again():
    bot = StyleOpponent(random.Random(2), profile([ret("forehand", None, result="OUT")]))
    shot, land = strike_from(bot, 0.3)
    assert not (P.on_table(land.pos[0], land.pos[2]) and land.net_clearance > 0)


def test_whiff_rate_follows_the_player():
    always = {f"forehand/{b}": [100, 100] for b in range(3)}
    never = {f"forehand/{b}": [400, 0] for b in range(3)}
    for faced, expect_whiffs in ((always, True), (never, False)):
        bot = StyleOpponent(random.Random(3), profile([ret("forehand", [0.3, 0.8])], faced=faced))
        whiffs = 0
        for _ in range(40):
            ball = P.Ball((0.3, 0.2, 0.0), (0.0, 0.5, 4.0), (0.0, 0.0, 0.0))
            bot.on_incoming(ball, 0.0)
            whiffs += bot._whiff
        assert (whiffs > 30) if expect_whiffs else (whiffs < 5)


def test_key_4_plays_the_style_opponent_or_falls_back(tmp_path, monkeypatch):
    from game.style import StyleOpponent as SO
    from tests.test_game_loop import make_game
    monkeypatch.setattr(C, "GAME_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(C, "STYLE_DIR", str(tmp_path / "styles"))
    game, clock, events = make_game()
    game.start_game("luca")                                       # nothing recorded yet
    assert not isinstance(game.opponent, SO)
    assert any("No recorded games" in e.get("text", "") for e in events)
    write_session(tmp_path / "logs", returns=[("forehand", 3.0, 7.0, 0.5, "in", (0.1, 0.7))], player="luca")
    game.start_game("luca")
    assert isinstance(game.opponent, SO) and game.opponent.hand == "left"
    assert os.path.exists(tmp_path / "styles" / "luca.json")     # saved for sharing
    assert json.load(open(tmp_path / "styles" / "luca.json"))["player"] == "luca"
    game.start_game("medium")
    assert not isinstance(game.opponent, SO)
    assert C.TAG_IDS[3] == "luca" and C.HIT_REACH_EARLY_M["luca"] == C.HIT_REACH_EARLY_M["medium"]


def test_only_the_players_own_hand_counts(tmp_path, monkeypatch):
    import time as _time
    write_session(tmp_path, returns=[("forehand", 3.0, 8.0, 0.8, "in", (0.2, 0.6))], misses=[("forehand", 3.0, False)])
    _time.sleep(1.1)                                       # sessions are named by the second
    write_session(tmp_path, returns=[("backhand", 3.0, 5.0, 0.0, "in", (0.0, 0.6))] * 3,
                  misses=[("backhand", 3.0, False)], lobby="right")   # lobby left on the wrong hand
    prof = build_profile("p1", log_root=str(tmp_path))
    assert [r["stroke"] for r in prof["returns"]] == ["forehand"]
    assert "backhand/0" not in prof["faced"] and prof["skipped_other_hand"] == 3


def test_recent_sessions_count_a_little_more(tmp_path, monkeypatch):
    import time as _time
    from game.recorder import session_weights
    monkeypatch.setattr(C, "GAME_DATA_RECENCY", 0.9)
    for speed in (6.0, 7.0, 8.0):
        write_session(tmp_path, returns=[("forehand", 3.0, speed, 0.5, "in", (0.2, 0.6))])
        _time.sleep(1.1)
    prof = build_profile("p1", log_root=str(tmp_path))
    w = {r["speed"]: r["w"] for r in prof["returns"]}
    assert w == {6.0: 0.81, 7.0: 0.9, 8.0: 1.0}
    assert session_weights(3) == [0.9 ** 2, 0.9, 1.0]
    # The opponent copies newer shots more often.
    bot = StyleOpponent(random.Random(5), prof)
    picks = [bot._sample(prof["returns"], 3.0)["speed"] for _ in range(3000)]
    assert picks.count(8.0) > picks.count(7.0) > picks.count(6.0)
