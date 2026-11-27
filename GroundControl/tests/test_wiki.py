import time

import config
from wiki import UA, Wiki, _trim


def wait_for(cond, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


EXTRACT = ("African dance refers to the various dance styles of sub-Saharan Africa. These dances are closely "
           "connected with the traditional rhythms and music traditions of the region. African dance uses the "
           "concepts of polyrhythm and total body articulation. African dances are often a collective activity "
           "performed in large groups, with significant interaction between dancers and onlookers.")


def fake_wiki():
    urls = []

    def fetch(url):
        urls.append(url)
        return {"query": {"pages": [
            {"index": 2, "title": "Dance in Ghana", "extract": "Ghanaian dances are performed at festivals.", "fullurl": "u2"},
            {"index": 1, "title": "African dance", "extract": EXTRACT,
             "fullurl": "https://en.wikipedia.org/wiki/African_dance"}]}}
    return Wiki(fetch=fetch), urls


def test_lookup_one_request_ranked_and_cached():
    w, urls = fake_wiki()
    r = w.lookup("West African dance")
    assert r["title"] == "African dance" and "polyrhythm" in r["extract"]
    assert [x["title"] for x in r["results"]] == ["African dance", "Dance in Ghana"]   # Wikipedia's order
    assert len(urls) == 1 and "generator=search" in urls[0] and "gsrsearch=West+African+dance" in urls[0]
    w.lookup("west african dance")
    assert len(urls) == 1                                  # cached


def test_trim_cuts_at_a_sentence():
    t = _trim("A" * 300 + ". " + "B" * 400 + ". " + "C" * 100, 600)
    assert t.endswith(".") and len(t) <= 600


def test_user_agent_is_generic():
    assert "GroundControl" in UA and "@" not in UA         # no personal details


def test_wikipedia_skill_end_to_end(gc):
    gc.wiki, _ = fake_wiki()
    gc.submit("how tall is Mount Everest?")
    assert wait_for(lambda: any("African dance" in p and "Dance in Ghana" in p for t, p in gc.sent
                                if t == config.TOPIC_REPLY))          # composed from the top results


def test_dance_style_outside_the_library_is_grounded_and_performed(gc):
    gc.wiki, urls = fake_wiki()
    gc.submit("dance an adowa")
    assert wait_for(lambda: any(p.startswith("Adowa: 4 bars at 126 bpm") for _, p in gc.sent), timeout=30)
    reply = next(p for _, p in gc.sent if p.startswith("Adowa"))
    assert "call response" in reply and "from Wikipedia: African dance" in reply
    assert "gsrsearch=adowa+dance" in urls[0]
    assert gc.hal.left == 0 and gc.hal.right == 0           # finished and stopped


def test_dance_style_in_the_library_needs_no_lookup(gc):
    gc.wiki, urls = fake_wiki()
    gc.submit("dance like traditional west african dance")
    assert wait_for(lambda: any(p.startswith("West African: 4 bars at 120 bpm") for _, p in gc.sent), timeout=30)
    reply = next(p for _, p in gc.sent if p.startswith("West African"))
    assert "polyrhythm" in reply and "from the style library" in reply
    assert urls == []                                       # no Wikipedia, no choreographer call


def test_get_retries_once_on_429(monkeypatch):
    import io
    import urllib.error
    import urllib.request
    import wiki
    calls = []

    def fake_urlopen(req, timeout=0):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "0"}, io.BytesIO())

        class R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return R(b'{"ok": 1}')
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(wiki.time, "sleep", lambda s: None)
    assert wiki.Wiki._get("https://en.wikipedia.org/x") == {"ok": 1} and len(calls) == 2



def test_stick_figure_ballet_at_user_tempo_on_matrix(gc):
    gc.wiki, _ = fake_wiki()
    gc.submit("make a stick figure dance at 120bpm on the led matrix like a ballet dancer")
    assert wait_for(lambda: any(p.startswith("Ballet on the LED matrix: 2 bars at 120 bpm") for _, p in gc.sent),
                    timeout=30)
    assert gc.hal.drive_log == []                                        # matrix only: no motors
    reply = next(p for _, p in gc.sent if p.startswith("Ballet on the LED matrix"))
    assert "arms and legs together" in reply and "from the style library" in reply and "car" not in reply.lower()
