# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "geopandas>=1.0",
#   "pyogrio>=0.9",
#   "requests>=2.31",
# ]
# ///
"""Download India sub-district (ADM3 / tehsil) boundaries from geoBoundaries and clip to coastal Odisha.

Source: geoBoundaries gbOpen IND ADM3 (lgdirectory.gov.in via Pathways Data), ODbL 1.0.
Output in data/raw/geoboundaries/:
  geoBoundaries-IND-ADM3.geojson   full India file, as downloaded
  odisha_coast_adm3.geojson        sub-districts intersecting the EMS AOI01 (Odisha coast) extent

Usage:  uv run scripts/download_boundaries.py
"""

from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import box

API_URL = "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM3/"

# EMS EMSR357 AOI01 (Odisha coast delineation) extent; covers AOI05 Gopalpur and AOI08 Puri.
CLIP_BBOX = (84.5691, 19.0790, 87.1061, 20.8727)

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "geoboundaries"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"exists, skipping download: {dest.name}")
        return
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    print(f"downloaded {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = requests.get(API_URL, timeout=30).json()
    full = OUT_DIR / Path(meta["gjDownloadURL"]).name
    download(meta["gjDownloadURL"], full)

    gdf = gpd.read_file(full)
    coast = gdf[gdf.intersects(box(*CLIP_BBOX))].copy()
    out = OUT_DIR / "odisha_coast_adm3.geojson"
    coast.to_file(out, driver="GeoJSON")

    print(f"{len(coast)} of {len(gdf)} sub-districts intersect the Odisha coast AOI -> {out.name}")
    print("columns:", list(coast.columns))
    print(coast.drop(columns="geometry").head(10).to_string())


if __name__ == "__main__":
    main()
