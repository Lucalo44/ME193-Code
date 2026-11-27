"""Fast path (python/fastpath): what it takes, what it must leave to the LLM, and the exact plans."""
import pytest

from fastpath.parse import build, words_to_digits
from fastpath.router import Router, core_text

R = Router()

TAKES = {
    "show a heart": [{"skill": "matrix_icon", "args": {"name": "heart"}}],
    "hey robot, what time is it, please?": [{"skill": "lookup", "args": {"what": "time", "about": ""}}],
    "turn right 90 degrees": [{"skill": "turn_by", "args": {"degrees": -90}}],
    "turn 180 degrees": [{"skill": "turn_by", "args": {"degrees": 180}}],
    "spin twice": [{"skill": "spin", "args": {"times": 2}}],
    "clear the display": [{"skill": "matrix_clear", "args": {}}],
    "how hot are you?": [{"skill": "self_report", "args": {"topic": "health"}}],
    "dance like traditional West African dance": [{"skill": "dance_style", "args": {
        "style": "west african", "bars": 4, "bpm": 0, "where": "robots"}}],
}
LEAVES = [
    "show a heart and send done to luca",      # two actions
    "don't show a heart",                      # negation
    "show a heart then turn left",             # sequence
    "draw a giraffe on the matrix",            # no icon: the LLM writes the word
    "how tall is the Eiffel Tower?",           # Wikipedia
    "asdf qwerty",                             # gibberish
    "what is 2+2?",                            # compute/answer
]


@pytest.mark.parametrize("text,steps", TAKES.items())
def test_takes_common_one_step_commands(text, steps):
    d = R.route(text)
    assert d["fast"], d["reason"]
    assert d["plan"]["steps"] == steps
    assert d["ms"] < 50


@pytest.mark.parametrize("text", LEAVES)
def test_leaves_everything_else_to_the_llm(text):
    assert not R.route(text)["fast"]


def test_slot_parsers_refuse_what_they_cant_read():
    assert build("turn", "turn to the righht 45 degrees")[0]["args"]["degrees"] == -45   # typo still a direction
    assert build("spin", "can you spin thre times?") is None                          # misspelled count
    assert build("drive", "roll forwards quicly for 1 seconds") is None                # unknown speed word
    assert build("write_text", "put yourself on the matrix") is None                    # not text
    assert build("memory_recall", "what is my team reaal quick?") is None               # leftover words
    assert build("memory_recall", "what's my lab partner again pls?")[0]["args"]["key"] == "lab partner"


def test_politeness_is_not_a_clause():
    assert core_text("hey robot, what time is it, please?") == "what time is it"
    assert core_text("turn to the right now") == "turn to the right"
    assert words_to_digits("forty five degrees") == "45 degrees"
