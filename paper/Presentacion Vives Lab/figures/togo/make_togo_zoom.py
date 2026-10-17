"""Full-bleed 16:9 map: Meta Relative Wealth Index (Chi et al. 2022), 2.4 km tiles, zoomed on
southern Togo (Lome and hinterland). Deciles within Togo, Spectral (D1 red = poorest).
Usage: python make_togo_zoom.py togo_rwi_border.json <outdir> [basemap.jpg]
basemap: Sentinel-2 cloudless 2016 mosaic (EOX, CC BY 4.0), Web Mercator tiles, file name
s2cloudless2016_z{Z}_x{X0}-{X1}_y{Y0}-{Y1}.jpg; tiles drawn semi-transparent on top."""
import re
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon, Rectangle

SRC, OUT = sys.argv[1], sys.argv[2]
BASE = sys.argv[3] if len(sys.argv) > 3 else None
ONLYBASE = len(sys.argv) > 4 and sys.argv[4] == "base"   # imagery + border only (pre-slide)
LON0, LAT0_, LON1 = 0.60, 6.150, 1.80
FW, FH = 16 / 2.54, 9 / 2.54                       # beamer 16:9 paper, inches
LATC = LAT0_ + 0.25
LAT1 = LAT0_ + (LON1 - LON0) * np.cos(np.radians(LATC)) * FH / FW
Z = 14

d = json.load(open(SRC))
rows = [l.split(",") for l in d["rwi_csv"].strip().split("\n")]
hdr = rows[0]; rw = np.array(rows[1:], float)
lat, lon, rwi = rw[:, hdr.index("latitude")], rw[:, hdr.index("longitude")], rw[:, hdr.index("rwi")]
n = 2 ** Z
tx = np.floor((lon + 180) / 360 * n).astype(int)
ty = np.floor((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n).astype(int)
o = np.argsort(rwi); pct = np.empty(len(rwi)); pct[o] = (np.arange(len(rwi)) + 0.5) / len(rwi) * 100   # within Togo
x0, x1, y0, y1 = tx.min(), tx.max(), ty.min(), ty.max()
A = np.full((y1 - y0 + 1, x1 - x0 + 1), np.nan); A[ty - y0, tx - x0] = pct
lon_e = np.arange(x0, x1 + 2) / n * 360 - 180
lat_e = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * np.arange(y0, y1 + 2) / n))))

def stitch(ways):
    pool = [list(map(tuple, w)) for w in ways]; chains = []
    while pool:
        ch = pool.pop(0); grown = True
        while grown and ch[0] != ch[-1]:
            grown = False
            for i, w in enumerate(pool):
                if w[0] == ch[-1]: ch += w[1:]
                elif w[-1] == ch[-1]: ch += w[::-1][1:]
                elif w[-1] == ch[0]: ch = w[:-1] + ch
                elif w[0] == ch[0]: ch = w[::-1][:-1] + ch
                else: continue
                pool.pop(i); grown = True; break
        chains.append(np.array(ch))
    return sorted(chains, key=len, reverse=True)
land = stitch(d["border_ways"])[0]; coast = stitch(d["coast_ways"])[0]
near = lambda p: int(np.argmin(((coast - p) ** 2).sum(1)))
ia, ib = near(land[-1]), near(land[0])
seg = coast[ia:ib + 1] if ia <= ib else coast[ib:ia + 1][::-1]
togo = np.vstack([land, seg, land[:1]])
# sea polygon: coastline (west -> east) closed along the bottom of the frame
cw = coast if coast[0, 0] < coast[-1, 0] else coast[::-1]
sea = np.vstack([cw, [[cw[-1, 0], LAT0_ - 1], [cw[0, 0], LAT0_ - 1]]])

cmap = ListedColormap(matplotlib.colormaps["Spectral"](np.linspace(0, 1, 10)))
norm = BoundaryNorm(np.arange(0, 101, 10), cmap.N)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman"],
                     "font.size": 8, "pdf.fonttype": 42})
LANDX, NODATA, SEA, LINE = "#e3e2e0", "#c9c9c9", "#dfe7ec", "#2b2b2b"   # neighbours, no estimate, sea
import matplotlib.patheffects as pe
from matplotlib.collections import PolyCollection
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1])
if BASE:
    from PIL import Image
    Zb, X0b, X1b, Y0b, Y1b = map(int, re.search(r"_z(\d+)_x(\d+)-(\d+)_y(\d+)-(\d+)", BASE).groups())
    img = np.asarray(Image.open(BASE).convert("RGB"))
    nb = 2 ** Zb; Hp = img.shape[0]
    merc_lat = lambda yp: np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * yp / (256 * nb)))))
    lat_top, lat_bot = merc_lat(Y0b * 256), merc_lat((Y1b + 1) * 256)
    # resample rows from Web Mercator to a linear-latitude grid (sub-0.1% shift here, done anyway)
    lat_rows = np.linspace(lat_top, lat_bot, Hp)
    ypix = (1 - np.arcsinh(np.tan(np.radians(lat_rows))) / np.pi) / 2 * 256 * nb - Y0b * 256
    img = img[np.clip(np.rint(ypix).astype(int), 0, Hp - 1)]
    # contrast stretch (1-99.5th pct on land-dominated luminance) so towns and roads stand out
    f = img.astype(float) / 255
    lum = f.mean(2)
    lo, hi = np.percentile(lum, [1, 99.5])
    f = np.clip((f - lo) / (hi - lo), 0, 1) ** 0.9
    g = f.mean(2, keepdims=True); f = np.clip(g + 1.10 * (f - g), 0, 1)   # slight saturation boost
    if not ONLYBASE:   # under the deciles: mostly grey imagery, so tile hues match the key
        g = f.mean(2, keepdims=True); f = 0.30 * f + 0.70 * g
    ax.imshow(f, extent=(X0b / nb * 360 - 180, (X1b + 1) / nb * 360 - 180, lat_bot, lat_top),
              interpolation="lanczos", zorder=0, rasterized=True)
    ALPHA, EDGE = 0.58, (1, 1, 1, 0.50)
else:
    ax.add_patch(Rectangle((-180, -90), 360, 180, fc=LANDX, ec="none", zorder=0))
    ax.add_patch(Polygon(togo, closed=True, fc=NODATA, ec="none", zorder=1))
    ALPHA, EDGE = 1.0, "white"
lw_ = lon_e[tx - x0]; le_ = lon_e[tx - x0 + 1]; ln_ = lat_e[ty - y0]; ls_ = lat_e[ty - y0 + 1]
vis = (le_ > LON0 - 0.05) & (lw_ < LON1 + 0.05) & (ln_ > LAT0_ - 0.05) & (ls_ < LAT1 + 0.05)
verts = np.stack([np.c_[lw_, ls_], np.c_[le_, ls_], np.c_[le_, ln_], np.c_[lw_, ln_]], 1)[vis]
fc = cmap(norm(pct[vis])); fc[:, 3] = ALPHA
if not ONLYBASE:
    ax.add_collection(PolyCollection(verts, facecolors=fc, edgecolors=[EDGE], linewidths=0.3, zorder=2))
if BASE:
    ax.plot(land[:, 0], land[:, 1], color="white", lw=1.1, zorder=5, solid_joinstyle="round",
            path_effects=[pe.Stroke(linewidth=2.3, foreground=(0, 0, 0, 0.55)), pe.Normal()])
else:
    ax.add_patch(Polygon(sea, closed=True, fc=SEA, ec="none", zorder=3))
    ax.plot(cw[:, 0], cw[:, 1], color="#9fb3bf", lw=0.6, zorder=4)
    ax.plot(land[:, 0], land[:, 1], color=LINE, lw=1.1, zorder=5, solid_joinstyle="round")
ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0_, LAT1)
ax.set_aspect(1 / np.cos(np.radians(LATC)), adjustable="datalim")
ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0_, LAT1)
ax.set_axis_off()

if not ONLYBASE:
    # decile key: small horizontal colourbar on a white card, bottom right (over the sea)
    cw_in, ch_in, mr, mb = 1.55, 0.10, 0.20, 0.36
    card = fig.add_axes([mr / FW, (mb - 0.27) / FH, (cw_in + 2 * 0.16) / FW, 0.73 / FH])
    card.set_axis_off()
    card.add_patch(Rectangle((0, 0), 1, 1, fc=(1, 1, 1, 0.62), ec=(0.17, 0.17, 0.17, 0.9), lw=0.6,
                             transform=card.transAxes, clip_on=False))
    if BASE:
        card.text(0.5, 0.07, "Imagery: Sentinel-2 cloudless 2016 by EOX (Copernicus Sentinel data)",
                  transform=card.transAxes, ha="center", va="bottom", fontsize=3.6, color="#333333")
    cax = fig.add_axes([(mr + 0.16) / FW, mb / FH, cw_in / FW, ch_in / FH])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=np.arange(5, 100, 10))
    cb.set_ticklabels([f"D{i}" for i in range(1, 11)])
    cax.set_xticks(np.arange(0, 101, 10), minor=True)
    cax.tick_params(which="major", length=0, labelsize=5.6, pad=2.2)
    cax.tick_params(which="minor", length=2.2, width=0.5)
    cb.outline.set_linewidth(0.6)
    cax.set_title("Relative Wealth Index\ndecile within Togo", fontsize=7, pad=4, linespacing=1.15)
else:
    fig.text(0.2 / FW + 0.004, 0.09 / FH, "Imagery: Sentinel-2 cloudless 2016 by EOX (Copernicus Sentinel data)",
             fontsize=3.8, color="#222222", ha="left", va="bottom",
             bbox=dict(boxstyle="square,pad=0.35", fc=(1, 1, 1, 0.62), ec=(0.17, 0.17, 0.17, 0.9), lw=0.5))
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/togo_rwi_zoom{'_satbase' if ONLYBASE else ('_sat' if BASE else '')}.{ext}", dpi=300)
print("view", LON0, LAT0_, LON1, round(LAT1, 3), "tiles in view",
      int(((lon > LON0) & (lon < LON1) & (lat > LAT0_) & (lat < LAT1)).sum()))
