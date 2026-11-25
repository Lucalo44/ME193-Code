"""Exact answers for things a language model is bad at: arithmetic and anything
inside a word. The model sees "strawberry" as the tokens str|aw|berry, so it
can't count letters; code can.

    compute("math", "37 times 89")          -> "37 × 89 = 3293."
    compute("count", "strawberry", "e")     -> "There is 1 'e' in 'strawberry'."
    compute("length", "dance")              -> "'dance' has 5 letters."
    compute("spell", "robot")               -> "robot: R-O-B-O-T."
    compute("reverse", "hello")             -> "'hello' reversed is 'olleh'."

Math goes through a small ast whitelist, never eval().
"""
import ast
import math
import operator
import re

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "sin": math.sin, "cos": math.cos, "tan": math.tan,
         "log": math.log, "log10": math.log10, "exp": math.exp, "floor": math.floor, "ceil": math.ceil}
NAMES = {"pi": math.pi, "e": math.e}
WORDS = [(r"\bsquare root of\b", "sqrt "), (r"\bsquared\b", "**2"), (r"\bcubed\b", "**3"),
         (r"\bto the power of\b", "**"), (r"\bmultiplied by\b", "*"), (r"\bdivided by\b", "/"),
         (r"\btimes\b", "*"), (r"\bplus\b", "+"), (r"\bminus\b", "-"), (r"\bover\b", "/"),
         (r"\bmod(ulo)?\b", "%"), (r"\bpercent of\b", "/100*")]


class ComputeError(ValueError):
    pass


def _normalize(expr):
    s = expr.lower().strip().rstrip("?.!")
    s = re.sub(r"^(what is|what's|calculate|compute|how much is)\s+", "", s)
    for pat, rep in WORDS:
        s = re.sub(pat, rep, s)
    s = s.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "")
    s = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", s)            # 3x4, 3 x 4 (not the x in "exp")
    s = re.sub(r"sqrt\s+(\d+(\.\d+)?)", r"sqrt(\1)", s)
    return s


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in NAMES:
        return NAMES[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 100 or abs(a) > 1e6):
            raise ComputeError("that power is too big")
        return OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        return UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS and len(node.args) == 1:
        return FUNCS[node.func.id](_eval(node.args[0]))
    raise ComputeError("I can only do arithmetic")


def _fmt(v):
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.6g}"
    return str(v)


def math_expr(expr):
    s = _normalize(expr)
    if not s or len(s) > 120:
        raise ComputeError("I need a math expression")
    try:
        v = _eval(ast.parse(s, mode="eval"))
    except ZeroDivisionError:
        raise ComputeError("that divides by zero") from None
    except (SyntaxError, TypeError, OverflowError, ValueError) as e:
        if isinstance(e, ComputeError):
            raise
        raise ComputeError(f"I couldn't work out '{expr}'") from None
    shown = re.sub(r"\s+", " ", s.replace("**", "^").replace("*", " × ").replace("/", " ÷ ")).strip()
    return f"{shown} = {_fmt(v)}."


def _word(text):
    """The word the user means: strip quotes and lead-ins like 'the word'."""
    t = text.strip().strip("\"'“”‘’").strip()
    t = re.sub(r"^(the )?(word|name|text) ", "", t, flags=re.I)
    return t.strip("\"'“”‘’ ")


def compute(op, text, of=""):
    text = (text or "").strip()
    of = (of or "").strip().strip("\"'“”‘’").lower()
    if op == "math":
        return math_expr(text)
    w = _word(text)
    if not w:
        raise ComputeError("I need some text to work on")
    if op == "count":
        if not of:
            n = len(w.split())
            return f"'{w}' has {n} word{'s' if n != 1 else ''}."
        m = re.fullmatch(r"(?:the )?(?:letter )?([a-z0-9])(?:'?s)?", of)   # e, e's, es, the letter e
        if m:
            ch = m.group(1)
            n = w.lower().count(ch)
            return f"There {'is' if n == 1 else 'are'} {n} '{ch}' in '{w}'."
        if " " not in w:                                  # "berry" in "strawberry"
            n = w.lower().count(of)
        else:
            n = len(re.findall(rf"\b{re.escape(of)}\b", w.lower()))
        return f"'{of}' appears {n} time{'s' if n != 1 else ''} in '{w}'."
    if op == "length":
        n = sum(c.isalpha() for c in w)
        return f"'{w}' has {n} letter{'s' if n != 1 else ''}."
    if op == "spell":
        return f"{w}: {'-'.join(c.upper() for c in w if not c.isspace())}."
    if op == "reverse":
        return f"'{w}' reversed is '{w[::-1]}'."
    raise ComputeError(f"unknown operation {op}")


OPS_LIST = ["math", "count", "length", "spell", "reverse"]
