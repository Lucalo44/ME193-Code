"""Build web/map.js, the campus map the phone page draws, from OpenStreetMap.

    python3 tools/make_map.py                 # download from the Overpass API
    python3 tools/make_map.py --json osm.json # or use a saved Overpass reply

Everything is projected to metres east/north of ORIGIN (the middle of the Tsungming Tu Complex)
and rounded to 0.1 m. Map data (c) OpenStreetMap contributors, ODbL.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ORIGIN = (42.40585, -71.11665)
BBOX = "42.4030,-71.1205,42.4090,-71.1125"
# the three joined buildings that make up the complex (OSM way ids)
TTC = {543805351: "Tsungming Tu Complex", 29537109: "Anderson", 745180417: "Robinson"}
QUERY = f"""[out:json][timeout:50];(
way["building"]({BBOX});way["highway"]({BBOX});way["railway"~"rail|light_rail|platform"]({BBOX});
way["leisure"~"pitch|garden|park"]({BBOX});way["landuse"~"grass|recreation_ground"]({BBOX});
way["natural"~"wood|water|scrub"]({BBOX});way["amenity"="parking"]({BBOX});
);out body geom;"""
ROADS = {"secondary": "major", "tertiary": "major", "residential": "street", "unclassified": "street",
         "service": "service", "pedestrian": "walk", "footway": "walk", "path": "walk", "steps": "steps"}


def project(geom):
    k = math.cos(math.radians(ORIGIN[0])) * 111320.0
    out = []
    for p in geom:
        out += [round((p["lon"] - ORIGIN[1]) * k, 1), round((p["lat"] - ORIGIN[0]) * 110540.0, 1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.json:
        data = json.loads(Path(a.json).read_text())
    else:
        # curl, not urllib: the macOS system Python's old TLS library can't reach the Overpass server
        raw = subprocess.run(["curl", "-sS", "-m", "90", "-A", "campus-map-script/1.0", "--data-urlencode",
                              "data=" + QUERY, "https://overpass-api.de/api/interpreter"],
                             check=True, capture_output=True).stdout
        data = json.loads(raw)
    m = {"origin": ORIGIN, "buildings": [], "roads": [], "rail": [], "green": [], "water": [], "parking": []}
    for e in data["elements"]:
        t, g = e.get("tags", {}), e.get("geometry")
        if not g:
            continue
        p = project(g)
        if "building" in t:
            b = {"p": p}
            if e["id"] in TTC:
                b["ttc"], b["n"] = 1, TTC[e["id"]]
            elif t.get("name"):
                b["n"] = t["name"]
            m["buildings"].append(b)
        elif t.get("highway") in ROADS:
            if t.get("area") == "yes" or t.get("tunnel") or t.get("indoor") == "yes":
                continue
            r = {"c": ROADS[t["highway"]], "p": p}
            if t.get("name") and r["c"] in ("major", "street"):
                r["n"] = t["name"]
            m["roads"].append(r)
        elif t.get("railway") in ("rail", "light_rail"):
            m["rail"].append(p)
        elif t.get("natural") == "water":
            m["water"].append(p)
        elif "leisure" in t or "landuse" in t or "natural" in t:
            m["green"].append(p)
        elif t.get("amenity") == "parking":
            m["parking"].append(p)
    out = HERE / "web" / "map.js"
    out.write_text("// Made by tools/make_map.py. Map data (c) OpenStreetMap contributors, ODbL.\n"
                   "window.CAMPUS = " + json.dumps(m, separators=(",", ":")) + ";\n")
    print(f"wrote {out} ({out.stat().st_size // 1024} kB): " + ", ".join(f"{len(v)} {k}" for k, v in m.items() if k != "origin"))


if __name__ == "__main__":
    main()
