"""Skill registry: ONE source of truth for
  (a) the compact skill catalog text in the prompt,
  (b) the JSON schema the decoder is constrained by,
  (c) argument validation and clamping at execution time.

    @skill("matrix_text", "Show text on the LED matrix.", args={"text": Str(40)})
    def matrix_text(ctx, text): ...
"""
from dataclasses import dataclass, field


class ArgError(ValueError):
    pass


# ---------------------------------------------------------------- arg types
class Str:
    def __init__(self, max_len=40):
        self.max_len = max_len

    def schema(self):
        return {"type": "string", "maxLength": self.max_len}

    def coerce(self, v):
        if v is None:
            raise ArgError("missing text")
        return str(v)[: self.max_len]

    def sig(self, name):
        return name


class Int:
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def schema(self):
        return {"type": "integer", "minimum": self.lo, "maximum": self.hi}

    def coerce(self, v):
        try:
            v = int(round(float(v)))
        except (TypeError, ValueError):
            raise ArgError(f"not a number: {v!r}") from None
        return max(self.lo, min(self.hi, v))

    def sig(self, name):
        return f"{name} {self.lo}..{self.hi}"


class Num(Int):
    def schema(self):
        return {"type": "number"}          # grammar can't bound floats; clamp after

    def coerce(self, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ArgError(f"not a number: {v!r}") from None
        return max(self.lo, min(self.hi, v))


class Enum:
    def __init__(self, values):
        self.values = list(values)

    def schema(self):
        return {"enum": self.values}

    def coerce(self, v):
        if v not in self.values:
            raise ArgError(f"{v!r} not one of {self.values}")
        return v

    def sig(self, name):
        return f"{name}: {'|'.join(self.values)}"


class Obj:
    def __init__(self, fields: dict):
        self.fields = fields

    def schema(self):
        return {"type": "object",
                "properties": {k: t.schema() for k, t in self.fields.items()},
                "required": list(self.fields),
                "additionalProperties": False}

    def coerce(self, v):
        if not isinstance(v, dict):
            raise ArgError("expected an object")
        return {k: t.coerce(v.get(k)) for k, t in self.fields.items()}

    def sig(self, name):
        return "{" + ", ".join(t.sig(k) if not isinstance(t, Enum) else k
                               for k, t in self.fields.items()) + "}"


class Arr:
    def __init__(self, item, min_items=1, max_items=4):
        self.item, self.min_items, self.max_items = item, min_items, max_items

    def schema(self):
        return {"type": "array", "items": self.item.schema(),
                "minItems": self.min_items, "maxItems": self.max_items}

    def coerce(self, v):
        if not isinstance(v, list) or len(v) < self.min_items:
            raise ArgError(f"expected a list of {self.min_items}..{self.max_items}")
        return [self.item.coerce(x) for x in v[: self.max_items]]

    def sig(self, name):
        return f"{name}: [{self.item.sig('')}]"


# ---------------------------------------------------------------- registry
@dataclass
class Skill:
    name: str
    desc: str
    args: dict
    fn: callable
    ns: str | None = None            # result namespace for {ns.key} placeholders
    motor: bool = False
    timeout_s: float = 15.0
    sig: str | None = None           # optional hand-written catalog signature
    enabled: bool = True
    returns: list = field(default_factory=list)

    def catalog_line(self):
        sig = self.sig or ", ".join(t.sig(k) for k, t in self.args.items())
        return f"{self.name}({sig})" + (f" {self.desc}" if self.desc else "")

    def step_schema(self):
        return {"type": "object",
                "properties": {
                    "skill": {"const": self.name},
                    "args": {"type": "object",
                             "properties": {k: t.schema() for k, t in self.args.items()},
                             "required": list(self.args),
                             "additionalProperties": False}},
                "required": ["skill", "args"],
                "additionalProperties": False}

    def coerce_args(self, args):
        args = args or {}
        if not isinstance(args, dict):
            raise ArgError("args must be an object")
        return {k: t.coerce(args.get(k)) for k, t in self.args.items()}


REGISTRY: dict[str, Skill] = {}


def skill(name, desc, args=None, ns=None, motor=False, timeout_s=15.0, sig=None, returns=None):
    def deco(fn):
        REGISTRY[name] = Skill(name, desc, args or {}, fn, ns, motor, timeout_s, sig,
                               returns=returns or [])
        return fn
    return deco


# Skills hidden from the planner per plan style. In "parts" style, read_imu is
# hidden ({imu.*} placeholders read on demand) so one clause = one step.
HIDDEN = {"parts": {"read_imu"}, "think": set()}


def enabled(style="parts"):
    """Skills the planner sees (catalog + grammar). All skills stay callable
    from hand-written plans."""
    return [s for s in REGISTRY.values() if s.enabled and s.name not in HIDDEN.get(style, set())]


def catalog_text(style="parts"):
    return "\n".join(s.catalog_line() for s in enabled(style))


def plan_schema(think_chars=60, max_steps=8, fixed=None, style="parts"):
    """Schema for the plan.

    style "parts": {parts: [the command's clauses, user's words], steps: [one per part]}
        fixed=parts -> repair pass: parts pinned, exactly len(parts) step slots.
    style "think": {think: short text naming the skills in order, steps: [...]}
        fixed=(think, [skill names]) -> repair pass: think pinned, steps forced
        to exactly that skill sequence.
    think_chars caps each part / the think; 0 drops it (ablation)."""
    props = {}
    steps_any = {"anyOf": [s.step_schema() for s in enabled(style)]}
    key = "parts" if style == "parts" else "think"
    if fixed is not None and style == "parts":
        props["parts"] = {"const": fixed}
        props["steps"] = {"type": "array", "prefixItems": [steps_any for _ in fixed],
                          "minItems": len(fixed), "maxItems": len(fixed)}
    elif fixed is not None:
        think, seq = fixed
        props["think"] = {"const": think}
        props["steps"] = {"type": "array", "prefixItems": [REGISTRY[n].step_schema() for n in seq],
                          "minItems": len(seq), "maxItems": len(seq)}
    else:
        if think_chars > 0:
            props[key] = ({"type": "array", "items": {"type": "string", "maxLength": think_chars},
                           "maxItems": max_steps} if style == "parts"
                          else {"type": "string", "maxLength": think_chars})
        props["steps"] = {"type": "array", "maxItems": max_steps, "items": steps_any}
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def tools_spec(style="parts"):
    """OpenAI-style tool list (Baseline B: native tool calling, like the LLM Brick)."""
    out = []
    for s in enabled(style):
        out.append({"type": "function", "function": {
            "name": s.name, "description": s.desc,
            "parameters": {"type": "object",
                           "properties": {k: t.schema() for k, t in s.args.items()},
                           "required": list(s.args)}}})
    return out


def validate_plan(plan, max_steps=8):
    """-> list of (Skill, coerced_args | ArgError). Raises ArgError if the plan is junk."""
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        raise ArgError("plan has no steps list")
    out = []
    for st in plan["steps"][:max_steps]:
        if not isinstance(st, dict) or st.get("skill") not in REGISTRY:
            out.append((None, ArgError(f"unknown skill {st.get('skill') if isinstance(st, dict) else st!r}")))
            continue
        sk = REGISTRY[st["skill"]]
        try:
            out.append((sk, sk.coerce_args(st.get("args"))))
        except ArgError as e:
            out.append((sk, e))
    return out
