# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pystac-client>=0.8",
#   "planetary-computer>=1.0",
#   "odc-stac>=0.3.10",
#   "rasterio>=1.3",
#   "pillow>=10",
# ]
# ///
"""Download Sentinel-2 L2A scenes over Puri, before and after Cyclone Fani (landfall 2019-05-03).

Source: Microsoft Planetary Computer STAC (free, no account; asset URLs are signed automatically).
Output per date in data/raw/sentinel2/:
  S2_<date>_puri.tif      4-band COG (B02, B03, B04, B08), uint16 reflectance x10000, EPSG:32645, 10 m
  S2_<date>_puri_scl.tif  scene classification layer (clouds, water, ...), same grid
  S2_<date>_puri_rgb.png  true-colour preview for eyeballing

Usage:  uv run scripts/download_sentinel2.py
"""

from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
from odc.stac import load
from PIL import Image

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Puri town + surrounding coast. Contains EMS AOI08 (Puri grading, 85.775-85.883 E, 19.780-19.857 N)
# with margin, so retrieval also sees tiles that have no ground truth (needed for refusal evals).
BBOX = (85.65, 19.72, 86.00, 19.95)

# Both on relative orbit R076 (same view geometry). Orbit R033 passes (e.g. 04-20, 05-15) are
# low-cloud but clip ~70% of the bbox at the swath edge, including Puri town.
DATES = {
    "2019-04-28": "pre",
    "2019-05-18": "post",
}

BANDS = ["B02", "B03", "B04", "B08"]
CLOUD_SCL_CLASSES = [3, 8, 9, 10]  # cloud shadow, cloud medium/high probability, thin cirrus

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "sentinel2"


def rgb_preview(ds) -> Image.Image:
    rgb = np.stack([ds[b].values for b in ("B04", "B03", "B02")], axis=-1).astype("float32")
    rgb = np.clip(rgb / 3000.0, 0, 1) ** (1 / 1.4)  # simple stretch + gamma
    return Image.fromarray((rgb * 255).astype("uint8"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

    for date, phase in DATES.items():
        items = list(
            catalog.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime=date).items()
        )
        if not items:
            print(f"{date}: no scenes found, skipping")
            continue

        ds = load(
            items,
            bands=BANDS + ["SCL"],
            bbox=BBOX,
            crs="EPSG:32645",
            resolution=10,
            groupby="solar_day",  # mosaics adjacent MGRS tiles (T45QUB/T45QUC) from the same pass
        ).isel(time=0)

        scl = ds["SCL"].values
        valid = scl > 0
        cloud_pct = 100 * np.isin(scl, CLOUD_SCL_CLASSES).sum() / max(valid.sum(), 1)
        nodata_pct = 100 * (~valid).mean()

        stem = OUT_DIR / f"S2_{date.replace('-', '')}_puri"
        ds[BANDS].to_array("band").odc.write_cog(f"{stem}.tif", overwrite=True)
        ds["SCL"].odc.write_cog(f"{stem}_scl.tif", overwrite=True)
        rgb_preview(ds).save(f"{stem}_rgb.png")

        print(
            f"{date} ({phase}): {len(items)} item(s) -> {stem.name}.tif "
            f"{ds.sizes['x']}x{ds.sizes['y']} px, no-data {nodata_pct:.1f}%, "
            f"cloud {cloud_pct:.1f}% of valid pixels"
        )


if __name__ == "__main__":
    main()
