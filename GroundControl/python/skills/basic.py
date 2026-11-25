"""Matrix, MQTT and timing skills (no motors)."""
import config
from mqtt_io import topic_allowed
from skills.context import SkillError
from skills.font import ICON_NAMES
from skills.registry import Enum, Num, Str, skill


@skill("matrix_text", "write, type or display text on the LED matrix", args={"text": Str(40)})
def matrix_text(ctx, text):
    one_pass = ctx.matrix.text(text)
    if one_pass:                      # long text: let it scroll past once before the next step
        ctx.sleep(min(one_pass, 6.0))
    return {"shown": text}


@skill("matrix_icon", "show a picture on the LED matrix",
       args={"name": Enum([n for n in ICON_NAMES if n != "think"])})
def matrix_icon(ctx, name):
    ctx.matrix.icon(name)
    return {"shown": name}


@skill("matrix_clear", "blank the LED matrix")
def matrix_clear(ctx):
    ctx.matrix.clear()


@skill("mqtt_publish", "tell, say, send, reply or report a message on a topic",
       args={"topic": Str(60), "message": Str(120)})
def mqtt_publish(ctx, topic, message):
    topic = topic.strip()
    if topic.lower() in ("reply", "me", "back", ""):
        topic = ctx.reply_to or config.TOPIC_REPLY
    if not topic_allowed(topic):
        raise SkillError(f"topic '{topic}' is not allowed")
    ctx.publish(topic, message)
    ctx.published.append(topic)
    return {"topic": topic, "message": message}


@skill("wait", "pause", args={"seconds": Num(0, 10)})
def wait(ctx, seconds):
    ctx.sleep(seconds)
