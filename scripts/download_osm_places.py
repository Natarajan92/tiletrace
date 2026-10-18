# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31",
# ]
# ///
"""Download named settlements (city/town/village/hamlet/suburb/neighbourhood) around Puri from OpenStreetMap.

Source: OpenStreetMap via the Overpass API, ODbL 1.0 (c) OpenStreetMap contributors.
Output in data/raw/osm/:
  puri_places.geojson   one Point per settlement (ways/relations reduced to their centre)

Usage:  uv run scripts/download_osm_places.py
"""

import json
from pathlib import Path

import requests

# Tried in order; public Overpass instances go down independently.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Same extent as the Sentinel-2 download (scripts/download_sentinel2.py).
BBOX = (85.65, 19.72, 86.00, 19.95)  # west, south, east, north

PLACE_TYPES = "city|town|village|hamlet|suburb|neighbourhood|locality"
KEEP_TAGS = ["name", "name:en", "name:or", "place", "population", "wikidata", "is_in"]

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "osm"


def build_query() -> str:
    w, s, e, n = BBOX
    return f"""
[out:json][timeout:120];
nwr["place"~"^({PLACE_TYPES})$"]["name"]({s},{w},{n},{e});
out center tags;
"""


def fetch(query: str) -> dict:
    headers = {"User-Agent": "tiletrace/0.1 (geospatial RAG research project)"}
    last_err = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": query}, headers=headers, timeout=180)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as err:
            print(f"{url} failed: {err}")
            last_err = err
    raise RuntimeError("all Overpass endpoints failed") from last_err


def to_feature(el: dict) -> dict | None:
    if el["type"] == "node":
        lon, lat = el["lon"], el["lat"]
    elif "center" in el:
        lon, lat = el["center"]["lon"], el["center"]["lat"]
    else:
        return None
    tags = el.get("tags", {})
    props = {"osm_id": f"{el['type']}/{el['id']}"}
    props.update({k.replace(":", "_"): tags[k] for k in KEEP_TAGS if k in tags})
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": props,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = fetch(build_query())
    features = [f for f in map(to_feature, data["elements"]) if f]

    out = OUT_DIR / "puri_places.geojson"
    out.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )

    counts: dict[str, int] = {}
    for f in features:
        counts[f["properties"]["place"]] = counts.get(f["properties"]["place"], 0) + 1
    print(f"{len(features)} named places -> {out.name}")
    print("by type:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    print("sample:", [f["properties"]["name"] for f in features[:15]])


if __name__ == "__main__":
    main()
