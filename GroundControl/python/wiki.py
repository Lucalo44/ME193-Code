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
