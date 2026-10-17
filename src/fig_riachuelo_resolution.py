#!/usr/bin/env python3
"""
Slide 02 figure: the same income map at RWI resolution and at 50 m, along the
lower Riachuelo (Buenos Aires).

  Left : Abbate et al. (2025) 50 m predictions averaged to Bing zoom-14 tiles,
         the grid of Meta's Relative Wealth Index (Chi et al. 2022), or the RWI
         itself when --rwi is given.
  Right: the 50 m predictions.

Both panels colour the SAME settled 50 m cells, so only resolution differs.
Colour = percentile in the metro-wide distribution of 50 m cells. Shaded band:
--buffer metres from the Riachuelo.

Usage, from the project root:
  python src/fig_riachuelo_resolution.py \
      --grid "PATH/income_estimates_2022.shp" \
      --out  "docs/Presentacion Vives Lab/figures"
Options:
  --col NAME         income column (auto-detected when unambiguous; columns are printed)
  --grid-crs EPSG    if the grid file has no CRS
  --rwi FILE.csv     Meta RWI for Argentina (latitude, longitude, rwi): left panel = RWI
  --river FILE       river geometry; otherwise fetched once from OSM with osmnx and
                     cached as <out>/riachuelo.geojson
  --buffer 500       riverbank band width (m)
  --accent/--card    colour ramp ends, e.g. the theme's mgAccent / mgCard hex
  --labels           add approximate neighbourhood labels (check positions!)

Writes <out>/riachuelo_resolution.pdf, .png and riachuelo_stats.tex (LaTeX macros).
Needs geopandas, shapely, matplotlib (+ osmnx once, unless --river).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

Z = 14                                        # Bing tile zoom used by the RWI
VIEW = (-58.49, -34.725, -58.325, -34.600)    # lon/lat window: Puerto Madero to Puente La Noria
UTM = "EPSG:32721"                            # UTM 21S, metres
RED = "#b3261e"
PLACES = {                                    # approximate, only drawn with --labels
    "Puerto Madero": (-58.363, -34.612),
    "Villa 21-24": (-58.393, -34.657),
}


# ---------------------------------------------------------------- helpers
def union(geoms):
    return geoms.union_all() if hasattr(geoms, "union_all") else geoms.unary_union


def tile_xy(lon, lat, z=Z):
    n = 2 ** z
    lat_r = np.radians(np.asarray(lat, float))
    x = np.floor((np.asarray(lon, float) + 180.0) / 360.0 * n).astype(int)
    y = np.floor((1.0 - np.arcsinh(np.tan(lat_r)) / np.pi) / 2.0 * n).astype(int)
    return x, y


def tile_box(x, y, z=Z):
    n = 2 ** z
    lon0, lon1 = x / n * 360 - 180, (x + 1) / n * 360 - 180
    lat_n = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))
    lat_s = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 1) / n))))
    return box(lon0, lat_s, lon1, lat_n)


def load_grid(path, col, crs):
    p = Path(path)
    g = gpd.read_parquet(p) if p.suffix.lower() in (".parquet", ".geoparquet") else gpd.read_file(p)
    if g.crs is None:
        if not crs:
            raise SystemExit("Grid has no CRS: pass --grid-crs (e.g. 4326).")
        g = g.set_crs(crs)
    num = [c for c in g.columns
           if c != g.geometry.name and pd.api.types.is_numeric_dtype(g[c])]
    print(f"{len(g):,} cells; numeric columns: {num}")
    if col is None:
        cand = [c for c in num if any(k in c.lower() for k in ("pred", "inc", "ingreso"))]
        if len(cand) != 1:
            raise SystemExit(f"Pick the income column with --col (candidates: {cand or num})")
        col = cand[0]
    print(f"income column: {col}")
    g = g[[col, g.geometry.name]].rename(columns={col: "y"})
    g = g[np.isfinite(g["y"])].to_crs(UTM)
    if g.geom_type.isin(["Point", "MultiPoint"]).all():          # points -> 50 m squares
        g["geometry"] = g.geometry.buffer(25, cap_style=3)
    # Average in logs when the column is a level (geometric mean within a tile).
    y = g["y"].to_numpy()
    g["z"] = np.log(y) if (y.min() > 0 and y.max() > 50) else y
    return g


def load_river(out, river):
    cache = out / "riachuelo.geojson"
    src = Path(river) if river else cache
    if src.exists():
        return gpd.read_file(src).to_crs(4326)
    import osmnx as ox
    W, S, E, N = VIEW[0] - 0.05, VIEW[1] - 0.05, VIEW[2] + 0.02, VIEW[3]
    tags = {"waterway": ["river", "canal"], "natural": "water"}
    try:
        f = ox.features_from_bbox(bbox=(W, S, E, N), tags=tags)          # osmnx >= 2
    except TypeError:
        f = ox.features_from_bbox(N, S, E, W, tags=tags)                 # osmnx 1.x
    name = f["name"].fillna("") if "name" in f else pd.Series("", index=f.index)
    r = f[name.str.contains("Riachuelo|Matanza", case=False, regex=True)]
    if r.empty:
        raise SystemExit("No OSM feature named Riachuelo/Matanza: pass --river.")
    r = gpd.GeoDataFrame(geometry=[union(r.geometry)], crs=f.crs)
    r.to_file(cache, driver="GeoJSON")
    print(f"river cached at {cache}")
    return r


# ---------------------------------------------------------------- main
def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--grid", required=True)
    a.add_argument("--out", default=".")
    a.add_argument("--col")
    a.add_argument("--grid-crs")
    a.add_argument("--rwi")
    a.add_argument("--river")
    a.add_argument("--buffer", type=float, default=500)
    a.add_argument("--accent", default="#2b4c7e")
    a.add_argument("--card", default="#f4f1ea")
    a.add_argument("--labels", action="store_true")
    a.add_argument("--year", default="2022")
    args = a.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    g = load_grid(args.grid, args.col, args.grid_crs)

    # Metro-wide reference: percentile of every 50 m cell.
    ref = np.sort(g["z"].to_numpy())
    pct = lambda v: 100.0 * np.searchsorted(ref, v, side="right") / len(ref)
    g["p_fine"] = pct(g["z"].to_numpy())

    cent = g.geometry.centroid.to_crs(4326)
    g["tx"], g["ty"] = tile_xy(cent.x, cent.y)

    # Left panel: tile means of the same predictions (all metro cells, so edge
    # tiles are complete), or the RWI ranked within its file.
    if args.rwi:
        r = pd.read_csv(args.rwi)
        r["tx"], r["ty"] = tile_xy(r["longitude"], r["latitude"])
        rs = np.sort(r["rwi"].to_numpy())
        r["p_left"] = 100.0 * np.searchsorted(rs, r["rwi"], side="right") / len(rs)
        g = g.merge(r[["tx", "ty", "p_left"]].drop_duplicates(["tx", "ty"]), on=["tx", "ty"], how="left")
        left_title = "Meta RWI · 2.4 km tiles"
        left_note = "percentile within Argentina"
    else:
        tm = g.groupby(["tx", "ty"])["z"].mean().rename("z_tile").reset_index()
        tm["p_left"] = pct(tm["z_tile"].to_numpy())
        g = g.merge(tm[["tx", "ty", "p_left"]], on=["tx", "ty"], how="left")
        left_title = "Averaged to the RWI grid"
        left_note = None

    # Study window.
    view = gpd.GeoSeries([box(*VIEW)], crs=4326).to_crs(UTM).iloc[0]
    xmin, ymin, xmax, ymax = view.bounds
    sub = g[g.geometry.centroid.within(view)].copy()
    print(f"{len(sub):,} settled cells in the window")

    # Riverbank band.
    river = load_river(out, args.river).to_crs(UTM)
    zone = union(river.geometry).buffer(args.buffer).intersection(view)
    zone_gs = gpd.GeoSeries([zone], crs=UTM)
    sub["in_zone"] = sub.geometry.centroid.within(zone)

    # Stats for the slide.
    zb, rest = sub[sub["in_zone"]], sub[~sub["in_zone"]]
    s = {
        "riaBuffer": f"{args.buffer:.0f}",
        "riaFine": f"{100 * (zb['p_fine'] < 20).mean():.0f}",      # % riverbank cells in metro bottom fifth, 50 m
        "riaCoarse": f"{100 * (zb['p_left'] < 20).mean():.0f}",    # same, tile values
        "riaPctFine": f"{zb['p_fine'].median():.0f}",              # median percentile, riverbank, 50 m
        "riaPctCoarse": f"{zb['p_left'].median():.0f}",
        "riaPctRest": f"{rest['p_fine'].median():.0f}",            # rest of the window, 50 m
        "riaTileKm": f"{360 / 2 ** Z * 111.32 * np.cos(np.radians(-34.66)):.1f}",  # tile width here
    }
    for k, v in s.items():
        print(f"  {k:13s} {v}")
    (out / "riachuelo_stats.tex").write_text(
        "% written by fig_riachuelo_resolution.py\n" + "".join(f"\\def\\{k}{{{v}}}\n" for k, v in s.items()))

    # ------------------------------------------------------------ plot
    cmap = LinearSegmentedColormap.from_list("w", [args.card, args.accent])
    norm = Normalize(0, 100)
    plt.rcParams.update({"font.size": 7, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.45), constrained_layout=True)

    tiles = {(x, y) for x, y in zip(sub["tx"], sub["ty"])}
    tile_gs = gpd.GeoSeries([tile_box(x, y) for x, y in tiles], crs=4326).to_crs(UTM)

    panels = [("p_left", left_title, s["riaPctCoarse"]), ("p_fine", "50 m cells", s["riaPctFine"])]
    for ax, (col, title, med) in zip(axes, panels):
        sub.plot(column=col, cmap=cmap, norm=norm, ax=ax, linewidth=0, rasterized=True,
                 missing_kwds={"color": "#dddddd", "rasterized": True})
        if col == "p_left":
            tile_gs.boundary.plot(ax=ax, color="white", lw=0.7)
        zone_gs.plot(ax=ax, color=RED, alpha=0.13, lw=0)
        zone_gs.boundary.plot(ax=ax, color=RED, lw=1.0, ls=(0, (3, 2)))
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(title, fontsize=8, loc="left", fontfamily="monospace", color="#555555")
        ax.text(0.5, -0.03, f"riverbank median: {med}th percentile", transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color=RED)
        if args.labels:
            for lab, (lo, la) in PLACES.items():
                px, py = gpd.GeoSeries([Point(lo, la)], crs=4326).to_crs(UTM).iloc[0].coords[0]
                ax.annotate(lab, (px, py), fontsize=6, ha="center", color="#222222",
                            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.8))

    # 2 km scale bar on the right panel
    ax = axes[1]
    x0, y0 = xmax - 2600, ymin + 600
    ax.plot([x0, x0 + 2000], [y0, y0], color="#222222", lw=1.5, solid_capstyle="butt")
    ax.text(x0 + 1000, y0 + 180, "2 km", ha="center", va="bottom", fontsize=6)
    axes[0].text(0.02, 0.02, f"Riachuelo · {args.buffer:.0f} m band", transform=axes[0].transAxes,
                 fontsize=6, color=RED, fontweight="bold")

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation="horizontal",
                      fraction=0.045, pad=0.06, shrink=0.45, ticks=[0, 20, 40, 60, 80, 100])
    cb.outline.set_linewidth(0.4)
    cb.set_label("percentile of predicted per-capita income, metro area"
                 + (f"  (left: {left_note})" if left_note else ""), fontsize=6.5)
    cb.ax.tick_params(labelsize=6, width=0.4)

    for ext in ("pdf", "png"):
        fig.savefig(out / f"riachuelo_resolution.{ext}", dpi=300)
    print(f"wrote {out / 'riachuelo_resolution.pdf'}")


if __name__ == "__main__":
    main()
