"""Input normalizer (python/planner/normalize.py): STT fixes must never touch normal speech."""
import pytest

from planner.normalize import normalize, turn_default_left

FIXES = {
    "spin to times": "spin two times",
    "spin too times": "spin two times",
    "dance for bars at 100 bpm": "dance four bars at 100 bpm",
    "turn to degrees": "turn two degrees",
    "wait for seconds": "wait four seconds",
    "loop it won time": "loop it one time",
    "right hi on the matrix": "write hi on the matrix",
    "please right hello on the led matrix": "please write hello on the led matrix",
    "dance ate bars": "dance eight bars",
}
UNTOUCHED = [
    "go to the left", "drive to the wall", "dance for 4 bars at 100 bpm", "turn right 90 degrees",
    "turn right on the matrix", "right 90 degrees", "send done to luca", "spin around for a while",
    "what time is it", "show a heart for me", "it's time to dance", "I want to dance",
    "turn right now", "write hi on the matrix", "spin 2 times", "dance for two bars",
    "turn left then right", "look to the right", "copyright on the matrix",
]


@pytest.mark.parametrize("raw,want", FIXES.items())
def test_stt_fixes(raw, want):
    out, changes = normalize(raw)
    assert out == want and changes


@pytest.mark.parametrize("raw", UNTOUCHED)
def test_normal_speech_is_untouched(raw):
    assert normalize(raw) == (raw, [])


def test_eval_commands_change_only_where_expected():
    import yaml
    from pathlib import Path
    cases = yaml.safe_load(open(Path(__file__).parent.parent / "tools" / "eval" / "commands.yaml"))
    changed = {c["id"]: normalize(c["text"])[0] for c in cases if normalize(c["text"])[1]}
    assert changed == {"vc2": "spin two times"} or set(changed) <= {"vc1", "vc2", "vc3"}, changed


def test_turn_without_direction_defaults_to_left():
    plan = {"steps": [{"skill": "turn_by", "args": {"degrees": -180}}]}
    assert turn_default_left(plan, "turn 180 degrees")["steps"][0]["args"]["degrees"] == 180
    plan = {"steps": [{"skill": "turn_by", "args": {"degrees": -90}}]}
    assert turn_default_left(plan, "turn right 90 degrees")["steps"][0]["args"]["degrees"] == -90
