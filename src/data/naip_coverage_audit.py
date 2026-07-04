"""Audit NAIP temporal coverage per state (Planetary Computer STAC).

NAIP is flown on ~2-3 year cycles that differ by state, and coverage starts
around 2010 — the valid panel-year range is therefore state-specific. This
utility counts STAC items (and their native GSDs) per state x year so the
temporal split design (#28) can use the real coverage instead of assuming
NYC's 2010-2024 cadence.

    python -m src.data.naip_coverage_audit --states Delaware NewYork \
        --out ~/outputs/naip_coverage.csv

Uses one small representative bbox per state (state-centroid window) rather
than the full state polygon: the goal is the flight-year CADENCE, not wall-to-
wall coverage. Increase --window-km for sparse rural states if needed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.naip_fetcher import get_catalog

# Rough state center points (lon, lat) for the audit windows. Only CONUS states
# the MS footprints cover; extend as needed.
STATE_CENTERS = {
    "Alabama": (-86.8, 32.8), "Arizona": (-111.7, 34.3), "Arkansas": (-92.4, 34.9),
    "California": (-119.5, 37.2), "Colorado": (-105.5, 39.0), "Connecticut": (-72.7, 41.6),
    "Delaware": (-75.5, 39.0), "DistrictofColumbia": (-77.0, 38.9), "Florida": (-81.7, 28.6),
    "Georgia": (-83.4, 32.6), "Idaho": (-114.6, 44.4), "Illinois": (-89.2, 40.0),
    "Indiana": (-86.3, 39.9), "Iowa": (-93.5, 42.1), "Kansas": (-98.4, 38.5),
    "Kentucky": (-85.3, 37.5), "Louisiana": (-92.0, 31.0), "Maine": (-69.2, 45.4),
    "Maryland": (-76.8, 39.0), "Massachusetts": (-71.8, 42.3), "Michigan": (-84.7, 43.3),
    "Minnesota": (-94.3, 46.3), "Mississippi": (-89.7, 32.7), "Missouri": (-92.5, 38.4),
    "Montana": (-109.6, 47.0), "Nebraska": (-99.8, 41.5), "Nevada": (-116.6, 39.3),
    "NewHampshire": (-71.6, 43.7), "NewJersey": (-74.7, 40.2), "NewMexico": (-106.1, 34.4),
    "NewYork": (-75.5, 42.9), "NorthCarolina": (-79.4, 35.5), "NorthDakota": (-100.5, 47.5),
    "Ohio": (-82.8, 40.3), "Oklahoma": (-97.5, 35.6), "Oregon": (-120.6, 44.0),
    "Pennsylvania": (-77.8, 40.9), "RhodeIsland": (-71.5, 41.7), "SouthCarolina": (-80.9, 33.9),
    "SouthDakota": (-100.2, 44.4), "Tennessee": (-86.3, 35.8), "Texas": (-99.4, 31.5),
    "Utah": (-111.7, 39.3), "Vermont": (-72.7, 44.1), "Virginia": (-78.8, 37.5),
    "Washington": (-120.4, 47.4), "WestVirginia": (-80.6, 38.6), "Wisconsin": (-89.7, 44.6),
    "Wyoming": (-107.6, 43.0),
}


def audit_state(state: str, years, window_km: float = 20.0) -> list[dict]:
    """STAC item count + GSD modes per year for one state's center window."""
    lon, lat = STATE_CENTERS[state]
    half_deg = window_km / 2 / 111.0
    bbox = [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]
    rows = []
    for year in years:
        try:
            items = list(get_catalog().search(
                collections=["naip"], bbox=bbox,
                datetime=f"{year}-01-01/{year}-12-31",
                max_items=100,
            ).items())
            gsds = sorted({
                float(i.properties.get("gsd", np.nan)) for i in items
                if i.properties.get("gsd") is not None
            })
            rows.append({"state": state, "year": year, "n_items": len(items),
                         "gsd_modes": ";".join(f"{g:g}" for g in gsds)})
        except Exception as e:
            rows.append({"state": state, "year": year, "n_items": -1,
                         "gsd_modes": f"ERROR: {e!r}"})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", nargs="+", default=None,
                        help="State stems (default: all known CONUS states)")
    parser.add_argument("--years", nargs=2, type=int, default=[2010, 2024],
                        metavar=("FROM", "TO"))
    parser.add_argument("--window-km", type=float, default=20.0)
    parser.add_argument("--out", default=str(Path.home() / "outputs" / "naip_coverage.csv"))
    args = parser.parse_args(argv)

    states = args.states or sorted(STATE_CENTERS)
    years = range(args.years[0], args.years[1] + 1)

    all_rows = []
    for state in states:
        print(f"Auditing {state} ...")
        all_rows.extend(audit_state(state, years, args.window_km))

    df = pd.DataFrame(all_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")

    # Per-state summary: which years have coverage
    pivot = (df[df["n_items"] > 0]
             .groupby("state")["year"].agg(["min", "max", "count"])
             .rename(columns={"min": "first_year", "max": "last_year", "count": "n_years"}))
    print(pivot.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
