"""Wikipedia lookups: real facts for questions the model can't (or shouldn't)
answer from memory, and grounding for style dances ("dance like West African
dance" reads what Wikipedia says about it).

    Wiki().lookup("blue whale") -> {"title", "extract", "url"} | None

Search (MediaWiki API) finds the best article, then the REST summary endpoint
gives its lead paragraph. Requests identify as a generic class-project client
(Wikimedia asks for a descriptive User-Agent); no personal details are sent.
Results are cached for an hour.
"""
import json
import re
import time
import urllib.parse
import urllib.request

UA = "GroundControl-UNOQ/1.0 (ME193 robotics class project; https://www.mediawiki.org/wiki/API:Etiquette)"
MAX_CHARS = 600
TTL_S = 3600


def _trim(text, n=MAX_CHARS):
    """Cut at a sentence boundary near n characters (every char is prefill time)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= n:
        return text
    cut = text[:n]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > n // 3 else cut.rstrip() + "…"


class Wiki:
    def __init__(self, fetch=None):
        self.fetch = fetch or self._get
        self._cache = {}

    @staticmethod
    def _get(url, tries=3):
        """GET JSON. On 429/503 (Wikimedia rate limiting), back off (Retry-After,
        capped at 3 s) and retry, up to `tries` attempts."""
        import urllib.error
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(req, timeout=8) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code not in (429, 503) or attempt == tries - 1:
                    raise
                try:
                    wait = float(e.headers.get("Retry-After", "")) if e.headers else 1.5 * (attempt + 1)
                except ValueError:
                    wait = 1.5 * (attempt + 1)
                time.sleep(min(3.0, max(0.5, wait)))

    def lookup(self, query, k=3):
        """-> {"title", "extract", "url", "results": [{title, extract, url} x up to k]} | None.
        One request: search plus each result's plain-text intro (MediaWiki
        generator=search + prop=extracts), in Wikipedia's own ranking order."""
        q = (query or "").strip()
        if not q:
            return None
        key = q.lower()
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < TTL_S:
            return hit[1]
        d = self.fetch("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query", "generator": "search", "gsrsearch": q, "gsrlimit": 5,
            "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exchars": MAX_CHARS,
            "inprop": "url", "format": "json", "formatversion": 2}))
        pages = sorted((d.get("query") or {}).get("pages") or [], key=lambda p: p.get("index", 99))
        results = [{"title": p["title"], "extract": _trim(p.get("extract", "")), "url": p.get("fullurl", "")}
                   for p in pages if p.get("extract")][:k]
        out = None
        if results:
            out = {**results[0], "results": results}
        self._cache[key] = (time.time(), out)
        return out


# ----------------------------------------------------------------------------- picking facts
# The composer prompt isn't cached, so every character of facts costs prefill time
# (about 5 tok/s on the board). Instead of each article's first N characters, keep
# the sentences that match the question: "how tall is X" needs the sentence with the
# height, which is often the last one in the intro.
_STOP = set("a an the is are was were be of in on at to for and or what whats who whom how many much does do did "
            "tell me about please can you your my it its this that which when where why there their with from by "
            "as into than then also has have had i draw show matrix led leds".split())
_EXPAND = {
    "tall": "height tall feet ft metres meters m) story stories floors high",
    "high": "height tall feet ft metres meters m) elevation high",
    "height": "height tall feet ft metres meters m) story stories",
    "old": "built founded established constructed born opened year years age",
    "age": "born founded built years age",
    "when": "year built founded established born opened began invented",
    "big": "area size km square largest miles",
    "large": "area size km square largest miles",
    "long": "length long km miles metres",
    "far": "distance km miles away",
    "deep": "depth deep metres feet",
    "heavy": "weight weighs kg tonnes tons pounds lb",
    "weigh": "weight weighs kg tonnes tons pounds lb",
    "fast": "speed km/h mph fastest",
    "population": "population inhabitants people residents",
    "people": "population inhabitants people residents",
    "who": "invented founded designed created built wrote painted discovered by",
    "where": "located in city country region",
    "invented": "invented invention developed created patented",
}


def _words(text):
    return re.findall(r"[a-z0-9]+(?:[/.)][a-z]*)?", (text or "").lower())


def relevance(question, sentence):
    """How well a fact sentence matches the question (0 = unrelated)."""
    q = [w for w in _words(question) if w not in _STOP]
    want = set(q)
    for w in q:
        want |= set(_EXPAND.get(w, "").split())
    s = set(_words(sentence))
    score = len(want & s)
    numeric = re.search(r"\b(how (tall|high|old|big|long|far|deep|heavy|fast|many|much)|when|what year|population)\b",
                        question or "", re.I)
    if numeric and re.search(r"\d", sentence):
        score += 1.5
    return score


def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", (text or "").strip()) if s]


def select_facts(question, results, budget=720):
    """Facts for the composer: each article's first sentence (what it is about) plus the
    sentences that best match the question, within `budget` characters, in article order."""
    picked = []                                              # (article index, sentence index, sentence)
    cands = []
    for ai, r in enumerate(results):
        sents = split_sentences(r.get("extract", ""))
        for si, s in enumerate(sents):
            if si == 0:
                picked.append((ai, si, s[:220]))
            else:
                cands.append((relevance(question, s) - 0.3 * ai, ai, si, s))
    used = sum(len(s) for *_, s in picked)
    for score, ai, si, s in sorted(cands, key=lambda c: -c[0]):
        if score <= 0 or used + len(s) > budget:
            continue
        picked.append((ai, si, s))
        used += len(s)
    out = []
    for ai, r in enumerate(results):
        sents = [s for a, si, s in sorted(picked) if a == ai]
        if sents:
            out.append(f"Wikipedia '{r['title']}': " + " ".join(sents))
    return " | ".join(out)


def best_fact_sentence(question, facts, need_number=True):
    """The fact sentence most relevant to the question, or None."""
    best, best_score = None, 0
    for chunk in (facts or "").split(" | "):
        body = re.sub(r"^Wikipedia '[^']*':\s*", "", chunk)
        for s in split_sentences(body):
            if need_number and not re.search(r"\d", s):
                continue
            sc = relevance(question, s)
            if sc > best_score:
                best, best_score = s, sc
    return best if best_score >= 2 else None
