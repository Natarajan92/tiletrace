# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31",
# ]
# ///
"""Harvest Copernicus EMS Rapid Mapping products for every activation in a set of countries (default: South Asia).

Two download routes, because EMS has two generations of infrastructure:
  * new dashboard API (roughly 2021+): rich JSON per activation (AOIs, products, impact stats,
    satellite image dates) plus one zip per product;
  * legacy activation pages (older activations): the dashboard API needs auth for these, but the public
    page lists every *_vector.zip on the cems-mapping-website S3 bucket.

Output in data/raw/ems/:
  catalog.json                    one entry per activation: list metadata, route used, products downloaded
  <EMSR code>/activation.json     dashboard-API detail (new route only) - impact stats, image dates, extents
  <EMSR code>/<product zip stem>/ extracted product (shapefiles + GeoJSON, summary xlsx, rasters, PDF maps)

Re-running skips products that were already extracted.

Usage:  uv run scripts/harvest_ems.py                      # South Asia
        uv run scripts/harvest_ems.py --countries India Nepal
        uv run scripts/harvest_ems.py --codes EMSR357 EMSR838
"""

import argparse
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

LIST_URL = "https://mapping.emergency.copernicus.eu/activations/api/activations/?limit=2000"
DETAIL_URL = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/?code={code}"
LEGACY_PAGE_URL = "https://mapping.emergency.copernicus.eu/activations/{code}/"
LEGACY_ZIP_RE = re.compile(r"https://cems-mapping-website[^\"' ]+?_vector\.zip")

SOUTH_ASIA = ["India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal", "Bhutan", "Afghanistan", "Maldives"]

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "ems"

session = requests.Session()
session.headers["User-Agent"] = "tiletrace/0.1 (geospatial RAG research project)"


def list_activations(countries: list[str], codes: list[str] | None) -> list[dict]:
    results = session.get(LIST_URL, timeout=120).json()["results"]
    rapid = [a for a in results if a["code"].startswith("EMSR")]  # skip EMSN risk & recovery studies
    if codes:
        return [a for a in rapid if a["code"] in codes]
    wanted = set(countries)
    return [a for a in rapid if any(c["short_name"] in wanted for c in a["countries"])]


def product_urls(code: str, act_dir: Path) -> tuple[str, list[str]]:
    """Return (route, zip urls). Saves dashboard detail JSON when available."""
    r = session.get(DETAIL_URL.format(code=code), timeout=120)
    if r.ok and r.json().get("results"):
        detail = r.json()["results"][0]
        (act_dir / "activation.json").write_text(json.dumps(detail, indent=1), encoding="utf-8")
        urls = [
            p["downloadPath"]
            for aoi in detail["aois"]
            for p in aoi["products"]
            if p.get("downloadPath")
        ]
        return "dashboard", urls

    page = session.get(LEGACY_PAGE_URL.format(code=code), timeout=120)
    page.raise_for_status()
    return "legacy", sorted(set(LEGACY_ZIP_RE.findall(page.text)))


def download_and_extract(url: str, act_dir: Path) -> tuple[str, str]:
    stem = url.rsplit("/", 1)[-1].removesuffix(".zip")
    dest = act_dir / stem
    if (dest / ".done").exists():
        return stem, "skipped"
    tmp = act_dir / f"{stem}.zip.part"
    with session.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(dest)
    tmp.unlink()
    (dest / ".done").touch()
    return stem, "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", default=SOUTH_ASIA)
    parser.add_argument("--codes", nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog_path = OUT_DIR / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}

    activations = list_activations(args.countries, args.codes)
    print(f"{len(activations)} rapid-mapping activations selected")

    for act in sorted(activations, key=lambda a: a["activationTime"]):
        code = act["code"]
        act_dir = OUT_DIR / code
        act_dir.mkdir(exist_ok=True)
        try:
            route, urls = product_urls(code, act_dir)
        except requests.RequestException as err:
            print(f"{code}: could not list products ({err})")
            catalog[code] = {**act, "route": None, "products": [], "error": str(err)}
            continue

        done, failed = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download_and_extract, u, act_dir): u for u in urls}
            for fut in as_completed(futures):
                try:
                    done.append(fut.result()[0])
                except (requests.RequestException, zipfile.BadZipFile) as err:
                    failed.append({"url": futures[fut], "error": str(err)})

        catalog[code] = {**act, "route": route, "products": sorted(done), "failed": failed}
        catalog_path.write_text(json.dumps(catalog, indent=1), encoding="utf-8")
        print(
            f"{code} {act['activationTime'][:10]} {act['category']['name']:<20} {route:<9} "
            f"{len(done)}/{len(urls)} products  | {act['name']}"
        )


if __name__ == "__main__":
    main()
