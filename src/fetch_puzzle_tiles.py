"""Candidate NAIP tiles for the deck's "puzzle" slide (similar-looking places, NYC vs Chicago).

Each site is a hand-picked point (lon, lat) in a neighbourhood whose built form should
look alike across the two cities (e.g. glass residential towers for "rich"). Tiles are
fetched with the pipeline's own NAIP fetcher at the pipeline's tile size (200 m), saved
as RGB PNGs, plus a contact sheet per wealth class for picking.

    python src/fetch_puzzle_tiles.py [--classes rich] [--year 2021] [--pixels 512]

Output: paper/Presentacion Vives Lab/figures/puzzle_candidates/<class>/
    <city>_<class>_<k>_<slug>.png, sites.csv (coordinates, actual flight year), sheet.png
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.utils.paths import PROJECT_ROOT

OUT_ROOT = PROJECT_ROOT / "paper" / "Presentacion Vives Lab" / "figures" / "puzzle_candidates"
TILE_METERS = 200.0  # same window as training/prediction tiles


@dataclass(frozen=True)
class Site:
    city: str    # "nyc" | "chi"
    wealth: str  # "rich" | "mid" | "poor"
    name: str    # neighbourhood, human readable
    lat: float
    lon: float
    year: int | None = None  # preferred NAIP flight year (good light); None -> CLI default

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")


# Hand-picked points (WGS84), each centred on a building after inspecting 400 m context
# views. Rich = glass residential high-rises in both cities. Years chosen for light:
# NYC 2022 (2021/2023 flights are dark with long shadows), Chicago 2023 (brightest recent).
SITES: list[Site] = [
    Site("nyc", "rich", "Hudson Yards residential", 40.75322, -74.00270, 2022),
    Site("nyc", "rich", "Battery Park City", 40.71677, -74.01509, 2022),
    Site("nyc", "rich", "Billionaires Row 57th St", 40.76593, -73.97822, 2022),
    Site("nyc", "rich", "Riverside South", 40.77651, -73.98949, 2022),
    # Same point, 2015 flight: off-nadir, so tower facades and height are visible.
    Site("nyc", "rich", "Battery Park City oblique 2015", 40.71677, -74.01509, 2015),
    Site("chi", "rich", "Lakeshore East", 41.88686, -87.61838, 2023),
    Site("chi", "rich", "Streeterville", 41.89286, -87.61686, 2023),
    Site("chi", "rich", "Gold Coast Lake Shore Dr", 41.90044, -87.62517, 2023),
    Site("chi", "rich", "Museum Park South Loop", 41.86856, -87.62490, 2023),
    # Mid = attached rowhouses (Brooklyn/Queens) vs two-flats and greystones on narrow lots.
    Site("nyc", "mid", "Crown Heights North", 40.67304, -73.94146, 2022),
    Site("nyc", "mid", "Prospect Lefferts Gardens", 40.65892, -73.95481, 2022),
    Site("nyc", "mid", "Bay Ridge", 40.62900, -74.02700, 2022),
    Site("nyc", "mid", "Ridgewood", 40.70499, -73.90477, 2022),
    Site("chi", "mid", "Logan Square", 41.92346, -87.70666, 2023),
    Site("chi", "mid", "Avondale", 41.93896, -87.71471, 2023),
    Site("chi", "mid", "Bridgeport", 41.83700, -87.65200, 2023),
    Site("chi", "mid", "Albany Park", 41.96714, -87.71800, 2023),
    # Mid, second batch: detached houses on narrow lots in both cities (Queens/Brooklyn
    # detached-house blocks vs Chicago's bungalow belt), closer built form than rowhouses.
    Site("nyc", "mid", "Middle Village", 40.71750, -73.88229, 2022),
    Site("nyc", "mid", "Glendale", 40.70344, -73.87500, 2022),
    Site("nyc", "mid", "Queens Village", 40.72100, -73.74300, 2022),
    Site("nyc", "mid", "Marine Park", 40.60900, -73.93000, 2022),
    Site("chi", "mid", "Portage Park", 41.95500, -87.76814, 2023),
    Site("chi", "mid", "Belmont Cragin", 41.93236, -87.77000, 2023),
    Site("chi", "mid", "Garfield Ridge", 41.79608, -87.77000, 2023),
    Site("chi", "mid", "Dunning", 41.94635, -87.80000, 2023),
    # Poor = public-housing superblocks (towers in the park), plus high-vacancy blocks
    # (Chicago has a fifth, alternative vacancy site).
    Site("nyc", "poor", "Queensbridge Houses", 40.75528, -73.94421, 2022),
    Site("nyc", "poor", "Marcy Houses", 40.69627, -73.95020, 2022),
    Site("nyc", "poor", "Van Dyke Houses Brownsville", 40.66445, -73.90822, 2022),
    Site("nyc", "poor", "Red Hook Houses", 40.67564, -74.00571, 2022),
    Site("chi", "poor", "Dearborn Homes", 41.84100, -87.62800, 2023),
    Site("chi", "poor", "Wentworth Gardens", 41.82550, -87.63233, 2023),
    Site("chi", "poor", "Altgeld Gardens", 41.65618, -87.60048, 2023),
    Site("chi", "poor", "Englewood vacant lots", 41.77450, -87.65050, 2023),
    Site("chi", "poor", "West Garfield Park vacant lots", 41.88028, -87.73000, 2023),
]


def select_sites(sites: list[Site], classes: list[str]) -> list[Site]:
    """Sites of the requested wealth classes, in table order."""
    return [s for s in sites if s.wealth in set(classes)]


def numbered_sites(classes: list[str], names: list[str] | None = None,
                   sites: list[Site] | None = None) -> list[tuple[Site, int]]:
    """(site, k) pairs, k numbering each city x class over the whole table, so a tile keeps
    its number when only some sites are (re)fetched. ``names``: slug substrings to keep."""
    out, counters = [], {}
    for site in select_sites(SITES if sites is None else sites, classes):
        k = counters[(site.city, site.wealth)] = counters.get((site.city, site.wealth), 0) + 1
        if not names or any(n.lower() in site.slug for n in names):
            out.append((site, k))
    return out


def to_rgb_image(crop: np.ndarray) -> Image.Image:
    """(C, H, W) uint8 NAIP crop -> RGB PIL image (first three bands)."""
    if crop.ndim != 3 or crop.shape[0] < 3:
        raise ValueError(f"expected (C>=3, H, W) crop, got {crop.shape}")
    return Image.fromarray(np.ascontiguousarray(np.transpose(crop[:3], (1, 2, 0))).astype(np.uint8))


def tile_path(out_dir: Path, site: Site, k: int) -> Path:
    return Path(out_dir) / site.wealth / f"{site.city}_{site.wealth}_{k}_{site.slug}.png"


def contact_sheet(images: list[tuple[str, Image.Image]], cols: int = 4, size: int = 256) -> Image.Image:
    """Grid of thumbnails with (possibly multi-line) captions, for picking tiles."""
    if not images:
        raise ValueError("no images")
    lines = max(label.count("\n") + 1 for label, _ in images)
    cap = 6 + 12 * lines
    rows = -(-len(images) // cols)
    sheet = Image.new("RGB", (cols * size, rows * (size + cap)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(images):
        x, y = (i % cols) * size, (i // cols) * (size + cap)
        sheet.paste(im.resize((size, size)), (x, y))
        draw.multiline_text((x + 4, y + size + 3), label, fill="black", spacing=2)
    return sheet


# ---------------------------------------------------------------------------
# Training labels (ACS) for each tile
# ---------------------------------------------------------------------------
PANEL = PROJECT_ROOT / "data" / "processed" / "us_metros_panel_2011_2023.feather"
LABEL_VAR = "W2_i_r5pct"      # default training indicator W2_r5 (dollars per person)
LAST_ACS_YEAR = 2023


def acs_year(flight_year: int | None) -> int:
    """ACS vintage matched to a flight year (latest available if the flight is newer)."""
    return min(int(flight_year), LAST_ACS_YEAR) if flight_year else LAST_ACS_YEAR


def load_panel(path: Path = PANEL, years=range(2011, LAST_ACS_YEAR + 1)):
    """Tract polygons + W2 dollars + per-capita income for the given years."""
    import geopandas as gpd
    cols = ["geoid_2023", "cbsa_code", "geometry"]
    for y in years:
        cols += [f"{LABEL_VAR}_{y}", f"per_capita_income_usd_{y}"]
    return gpd.read_feather(path, columns=cols)


def within_cbsa_percentile(values: pd.Series, cbsa: pd.Series) -> pd.Series:
    """Percentile (0-100) of each value among non-missing tracts of its CBSA."""
    return values.groupby(cbsa).rank(pct=True) * 100


def tract_labels(points: pd.DataFrame, panel) -> pd.DataFrame:
    """For rows with lon/lat/year: tract GEOID, W2 dollars, within-metro W2 percentile
    and per-capita income, in the ACS vintage matched to each row's flight year."""
    import geopandas as gpd
    pts = gpd.GeoDataFrame(points.copy(), geometry=gpd.points_from_xy(points.lon, points.lat),
                           crs="EPSG:4326").to_crs(panel.crs)
    hit = gpd.sjoin(pts, panel, how="left", predicate="within")
    out = []
    for idx, r in hit.iterrows():
        y = acs_year(r["year"])
        row = dict(index=idx, geoid=r["geoid_2023"], cbsa=r["cbsa_code"], acs_year=y,
                   w2_usd=r.get(f"{LABEL_VAR}_{y}"), income_usd=r.get(f"per_capita_income_usd_{y}"))
        same = panel[panel.cbsa_code == r["cbsa_code"]]
        pct = within_cbsa_percentile(same[f"{LABEL_VAR}_{y}"], same.cbsa_code)
        match = same.index[same.geoid_2023 == r["geoid_2023"]]
        row["w2_pct"] = float(pct.loc[match[0]]) if len(match) and pd.notna(row["w2_usd"]) else np.nan
        out.append(row)
    lab = pd.DataFrame(out).drop_duplicates("index").set_index("index")
    return points.join(lab)


def label_text(row) -> str:
    """Two-line sheet caption: tile id + training label."""
    head = f"{row['city']} {row['k']}: {row['name']} ({row['year']})"
    if pd.isna(row.get("w2_usd")):
        return head + "\nno ACS label (tract not in panel)"
    return (head + f"\nW2 ${row['w2_usd'] / 1e3:,.0f}k  |  pct {row['w2_pct']:.0f} in metro"
            f"  |  inc ${row['income_usd'] / 1e3:,.0f}k  (ACS {row['acs_year']})")


def labelled_sheets(classes: list[str], out_dir: Path = OUT_ROOT, panel=None) -> pd.DataFrame:
    """Rebuild each class's sheet.png from tiles on disk, captioned with ACS labels;
    also writes the labels into sites.csv."""
    rows = []
    for site, k in numbered_sites(classes):
        path = tile_path(out_dir, site, k)
        if path.exists():
            rows.append(dict(city=site.city, wealth=site.wealth, k=k, name=site.name, lat=site.lat,
                             lon=site.lon, year=site.year, file=path.name))
    if not rows:
        raise ValueError("no tiles on disk: run the fetch first")
    table = tract_labels(pd.DataFrame(rows), load_panel() if panel is None else panel)
    for wealth, grp in table.groupby("wealth", sort=False):
        folder = Path(out_dir) / wealth
        images = [(label_text(r), Image.open(folder / r["file"]).convert("RGB")) for _, r in grp.iterrows()]
        sheet = contact_sheet(images)
        try:
            sheet.save(folder / "sheet.png")
        except OSError:  # e.g. sheet.png open in a Windows viewer (file locked)
            print(f"{folder / 'sheet.png'} is locked; wrote sheet_labelled.png instead", flush=True)
            sheet.save(folder / "sheet_labelled.png")
        grp.to_csv(folder / "sites.csv", index=False)
    return table


def site_year(site: Site, default: int | None) -> int | None:
    """The site's own flight year if set, else the run default."""
    return site.year if site.year is not None else default


def grid_overlay(im: Image.Image, meters: float, step: float = 50.0,
                 box_meters: float = TILE_METERS) -> Image.Image:
    """Copy of a north-up view of ``meters`` per side with a metre grid centred on the
    site (labels: metres east / north of the point) and the final tile outlined, used to
    read off how far to move a point so a building sits in the centre."""
    out = im.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    ppm = w / meters
    cx, cy = w / 2, h / 2
    n = int(meters / 2 // step)
    for k in range(-n, n + 1):
        x, y = cx + k * step * ppm, cy + k * step * ppm
        colour = (255, 0, 0) if k == 0 else (255, 255, 0)
        d.line([(x, 0), (x, h - 1)], fill=colour)
        d.line([(0, y), (w - 1, y)], fill=colour)
        d.text((x + 2, 2), f"{k * step:g}", fill=(255, 255, 0))
        d.text((2, y + 2), f"{-k * step:g}", fill=(255, 255, 0))
    half = box_meters / 2 * ppm
    d.rectangle([cx - half, cy - half, cx + half, cy + half], outline=(0, 255, 255), width=2)
    return out


def context_views(classes: list[str], year: int | None, out_dir: Path = OUT_ROOT,
                  meters: float = 400.0, pixels: int = 640,
                  names: list[str] | None = None) -> list[Path]:
    """Wider gridded views of each site, written to <out>/<class>/context/."""
    from src.data.naip_fetcher import fetch_naip
    written = []
    for site, _ in numbered_sites(classes, names):
        res = fetch_naip(site.lon, site.lat, meters, nbands=3, out_pixels=pixels,
                         year_hint=site_year(site, year))
        if res.crop is None:
            print(f"{site.city} {site.name}: failure={res.failure}", flush=True)
            continue
        path = Path(out_dir) / site.wealth / "context" / f"{site.city}_{site.slug}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        grid_overlay(to_rgb_image(res.crop), meters).save(path)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Building-footprint overlays (Microsoft US Building Footprints, local GeoParquet)
# ---------------------------------------------------------------------------
FOOTPRINTS = PROJECT_ROOT / "data" / "processed" / "buildings_polygons"
STATE_DIR = {"nyc": "NewYork", "chi": "Illinois"}
FP_FILL = (116, 89, 180, 90)      # theme accent (mgAccent #7459B4), translucent
FP_EDGE = (255, 255, 255, 235)


def tile_frame(site: Site, year: int | None) -> tuple:
    """(native CRS, native bounds) of the exact window the fetcher reads for this tile:
    same item choice and the same EPSG:4326 -> native envelope as fetch_naip."""
    from src.data.naip_fetcher import _bbox_to_native, open_naip_src, resolve_naip_item
    ref = resolve_naip_item(site.lon, site.lat, TILE_METERS, site_year(site, year))
    if ref.failure is not None:
        raise RuntimeError(f"{site.name}: {ref.failure}")
    with open_naip_src(ref.href, ref.use_cache) as src:
        return src.crs, _bbox_to_native(ref.bbox, src.crs)


def load_footprints(state: str, bounds_5070: tuple, root: Path = FOOTPRINTS):
    """Footprint polygons (EPSG:5070) intersecting ``bounds_5070``, reading only the
    parquet parts whose stored bbox overlaps it."""
    import json
    import geopandas as gpd
    import pyarrow.parquet as pq
    from shapely.geometry import box
    minx, miny, maxx, maxy = bounds_5070
    parts = []
    for f in sorted((Path(root) / state).glob("*.parquet")):
        geo = json.loads(pq.ParquetFile(f).schema_arrow.metadata[b"geo"])
        bx = geo["columns"]["geometry"].get("bbox")
        if bx and (bx[0] > maxx or bx[2] < minx or bx[1] > maxy or bx[3] < miny):
            continue
        g = gpd.read_parquet(f)
        parts.append(g[g.intersects(box(*bounds_5070))])
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:5070")
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)


def footprint_overlay(im: Image.Image, polygons, bounds: tuple) -> Image.Image:
    """Draw polygons (already in the tile's native CRS) onto the tile; ``bounds`` is the
    native window that was resampled to the image, so the mapping is linear."""
    minx, miny, maxx, maxy = bounds
    w, h = im.size
    to_px = lambda x, y: ((x - minx) / (maxx - minx) * w, (maxy - y) / (maxy - miny) * h)
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    width = max(1, round(w / 256))
    for geom in polygons:
        for poly in getattr(geom, "geoms", [geom]):
            ring = [to_px(x, y) for x, y in poly.exterior.coords]
            d.polygon(ring, fill=FP_FILL)
            d.line(ring, fill=FP_EDGE, width=width)
    return Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")


# The six tiles shown on the puzzle slide: deck slot -> (wealth, site name)
PUZZLE_PICKS = {
    "nyc_rich": ("rich", "Riverside South"), "chi_rich": ("rich", "Lakeshore East"),
    "nyc_mid": ("mid", "Queens Village"), "chi_mid": ("mid", "Belmont Cragin"),
    "nyc_poor": ("poor", "Queensbridge Houses"), "chi_poor": ("poor", "Dearborn Homes"),
}
PUZZLE_DIR = PROJECT_ROOT / "paper" / "Presentacion Vives Lab" / "figures" / "puzzle"


def puzzle_tiles(picks: dict = PUZZLE_PICKS, out_dir: Path = PUZZLE_DIR,
                 cand_dir: Path = OUT_ROOT, year: int | None = None) -> pd.DataFrame:
    """Write <slot>.png (the candidate tile) and <slot>_fp.png (with footprints) for the
    deck, plus labels.csv with each tile's W2 and income."""
    import geopandas as gpd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    numbered = {(s.wealth, s.name): (s, k) for s, k in numbered_sites(["rich", "mid", "poor"])}
    rows = []
    for slot, key in picks.items():
        site, k = numbered[key]
        im = Image.open(tile_path(cand_dir, site, k)).convert("RGB")
        crs, bounds = tile_frame(site, year)
        frame = gpd.GeoSeries([__import__("shapely.geometry", fromlist=["box"]).box(*bounds)], crs=crs)
        fp = load_footprints(STATE_DIR[site.city], tuple(frame.to_crs("EPSG:5070").total_bounds))
        polys = fp.to_crs(crs).geometry
        im.save(out_dir / f"{slot}.png")
        footprint_overlay(im, polys, bounds).save(out_dir / f"{slot}_fp.png")
        rows.append(dict(slot=slot, city=site.city, wealth=site.wealth, k=k, name=site.name, lat=site.lat,
                         lon=site.lon, year=site.year, file=tile_path(cand_dir, site, k).name,
                         n_footprints=len(fp)))
        print(f"{slot}: {site.name}, {len(fp)} footprints", flush=True)
    table = tract_labels(pd.DataFrame(rows), load_panel())
    table.to_csv(out_dir / "labels.csv", index=False)
    return table


def fetch_site(site: Site, year: int | None, pixels: int):
    """Fetch one site's tile with the pipeline fetcher; returns NaipFetchResult."""
    from src.data.naip_fetcher import fetch_naip  # network + GDAL, imported lazily
    return fetch_naip(site.lon, site.lat, TILE_METERS, nbands=3, out_pixels=pixels,
                      year_hint=site_year(site, year))


def run(classes: list[str], year: int | None, pixels: int, out_dir: Path = OUT_ROOT,
        names: list[str] | None = None) -> pd.DataFrame:
    rows, sheets = [], {}
    for site, k in numbered_sites(classes, names):
        res = fetch_site(site, year, pixels)
        row = dict(city=site.city, wealth=site.wealth, k=k, name=site.name, lat=site.lat, lon=site.lon,
                   year=res.actual_year, failure=res.failure, partial=res.partial_coverage, file=None)
        if res.crop is not None:
            path = tile_path(out_dir, site, k)
            path.parent.mkdir(parents=True, exist_ok=True)
            im = to_rgb_image(res.crop)
            im.save(path)
            row["file"] = path.name
            sheets.setdefault(site.wealth, []).append((f"{site.city} {k}: {site.name} ({res.actual_year})", im))
        print(f"{site.city} {site.wealth} {k} {site.name}: year={res.actual_year} failure={res.failure}", flush=True)
        rows.append(row)
    table = pd.DataFrame(rows)
    for wealth, images in sheets.items():
        folder = Path(out_dir) / wealth
        contact_sheet(images).save(folder / "sheet.png")
        table[table.wealth == wealth].to_csv(folder / "sites.csv", index=False)
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--classes", nargs="+", default=["rich"], choices=["rich", "mid", "poor"])
    ap.add_argument("--year", type=int, default=2021,
                    help="NAIP flight year for sites without their own year")
    ap.add_argument("--pixels", type=int, default=512, help="output tile size in pixels")
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--sites", nargs="+", default=None,
                    help="only these sites (substrings of the site slug), e.g. middle_village")
    ap.add_argument("--puzzle", action="store_true",
                    help="write the six chosen tiles (+ footprint overlays, labels.csv) to figures/puzzle")
    ap.add_argument("--labels", action="store_true",
                    help="rebuild sheet.png from tiles on disk with ACS training labels (no fetch)")
    ap.add_argument("--context", action="store_true",
                    help="write 400 m gridded context views (for re-centring points) instead of tiles")
    a = ap.parse_args()
    if a.puzzle:
        cols = ["slot", "name", "w2_usd", "income_usd", "w2_pct", "n_footprints"]
        print(puzzle_tiles()[cols].round(0).to_string())
    elif a.labels:
        cols = ["city", "k", "name", "geoid", "acs_year", "w2_usd", "w2_pct", "income_usd"]
        print(labelled_sheets(a.classes, a.out)[cols].round(1).to_string())
    elif a.context:
        for p in context_views(a.classes, a.year, a.out, names=a.sites):
            print(p)
    else:
        print(run(a.classes, a.year, a.pixels, a.out, a.sites).to_string())
