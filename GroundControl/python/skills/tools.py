"""Tool skills: the model picks the option and copies the inputs; code produces the
answer, so facts and arithmetic are never hallucinated.

    lookup(what, about)       time / date / weather / forecast / apps / app_info
    compute(op, text, of)     math / count / length / spell / reverse
    memory(op, key, value)    save / recall / forget
"""
import config
from compute import OPS_LIST as COMPUTE_OPS
from compute import ComputeError, compute as do_compute
from lookup import WHAT as LOOKUP_WHAT
from memory_store import OPS_LIST as MEMORY_OPS
from skills.context import SkillError
from skills.registry import Enum, Str, skill


def _reply(ctx, text):
    topic = ctx.reply_to or config.TOPIC_REPLY
    ctx.publish(topic, text)
    ctx.published.append(topic)
    ctx.answers.append(text)
    if len(text) <= 24:
        ctx.matrix.text(text)


@skill("lookup", "real facts: time, date, the weather outside now, forecast (later or tomorrow), your apps or what one of them does",
       args={"what": Enum(LOOKUP_WHAT), "about": Str(40)}, timeout_s=20)
def lookup(ctx, what, about):
    if ctx.lookup is None:
        raise SkillError("lookup isn't available")
    facts = ctx.lookup.answer(what, about)
    text = None
    if what not in ("time", "date") and ctx.compose is not None and ctx.question:
        text = ctx.compose(ctx.question, facts)         # direct answer to the exact question
    text = text or facts
    _reply(ctx, text)
    return {"what": what, "text": text}


@skill("compute", "exact math; count = how many times a letter is in a word; length = how many letters",
       args={"op": Enum(COMPUTE_OPS), "text": Str(80), "of": Str(20)})
def compute(ctx, op, text, of):
    try:
        out = do_compute(op, text, of)
    except ComputeError as e:
        out = f"Sorry, {e}."
    _reply(ctx, out)
    return {"op": op, "text": out}


@skill("memory", "remember, recall or forget what the user tells you about themselves (my name, my favorite ...)",
       args={"op": Enum(MEMORY_OPS), "key": Str(30), "value": Str(120)})
def memory(ctx, op, key, value):
    if ctx.memory is None:
        raise SkillError("memory isn't available")
    out = ctx.memory.do(op, key, value)
    _reply(ctx, out)
    return {"op": op, "text": out}


@skill("wikipedia", "look up facts on Wikipedia: people, places, history, science, products, anything you're unsure of",
       args={"query": Str(60)}, timeout_s=60)
def wikipedia(ctx, query):
    if ctx.wiki is None:
        raise SkillError("Wikipedia isn't available")
    try:
        res = ctx.wiki.lookup(query)
    except Exception as e:  # noqa: BLE001  no network shouldn't kill the job
        res, err = None, e
    else:
        err = None
    if res is None:
        text = (f"I couldn't reach Wikipedia ({err})." if err
                else f"I couldn't find a Wikipedia article about {query}.")
        _reply(ctx, text)
        return {"query": query, "text": text}
    # The top few articles, not just one: the best match isn't always ranked first
    # ("capital of Peru" ranks "Capital punishment in Peru" above "Peru").
    results = res.get("results") or [res]
    facts = " | ".join(f"Wikipedia '{r['title']}': {r['extract'][:280]}" for r in results)
    text = ctx.compose(ctx.question, facts) if ctx.compose is not None and ctx.question else None
    text = text or res["extract"].split(". ")[0].rstrip(".") + "."
    if "wikipedia" not in text.lower():
        text += " (source: Wikipedia)"
    _reply(ctx, text)
    return {"query": query, "articles": [r["title"] for r in results], "urls": [r["url"] for r in results],
            "text": text}
