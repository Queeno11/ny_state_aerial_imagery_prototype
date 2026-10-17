"""Slide 02: Meta RWI (2.4 km tiles) vs Abbate et al. (2025) 50 m predictions,
lower Riachuelo, Buenos Aires. numpy + matplotlib only."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

CELLS, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MODE = sys.argv[4] if len(sys.argv) > 4 else "rwi"     # "rwi" or "own" (own predictions averaged to RWI tiles)
SUF = ("" if MODE == "rwi" else "_own") + "_slide"
BUFFER = 500.0                                  # m
def _tb(x, y, z=14):
    n = 2 ** z
    return (x / n * 360 - 180, np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 1) / n)))),
            (x + 1) / n * 360 - 180, np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))))
TX0, TX1, TY0, TY1 = 5533, 5536, 9874, 9877          # 4 x 4 tiles, lower Riachuelo
VIEW = (_tb(TX0, TY1)[0], _tb(TX0, TY1)[1], _tb(TX1, TY0)[2], _tb(TX1, TY0)[3])
LAT0 = -34.66
KX, KY = 111320 * np.cos(np.radians(LAT0)), 110574.0   # m per degree
STEP = 0.00059
Z = 14
INK, MUTED, RED = "#222222", "#6b6b6b", "#b3261e"
# sequential ramp, light -> dark (one hue), 10 deciles
LO, HI = np.array(to_rgb("#f4f1ea")), np.array(to_rgb("#27477a"))
t = np.linspace(0, 1, 10) ** 0.9
cmap = ListedColormap([tuple(LO + (HI - LO) * v) for v in t])
norm = BoundaryNorm(np.arange(0, 101, 10), cmap.N)

# ---------------------------------------------------------------- data
c = np.load(CELLS)
cx, cy, pred = c["cx"], c["cy"], c["pred"]
d = json.load(open(DATA))
rows = [l.split(",") for l in d["rwi_csv"].strip().split("\n")]
hdr = rows[0]; rw = np.array(rows[1:], float)
rlat, rlon, rwi = rw[:, hdr.index("latitude")], rw[:, hdr.index("longitude")], rw[:, hdr.index("rwi")]

def tile_xy(lon, lat):
    n = 2 ** Z
    x = np.floor((lon + 180) / 360 * n).astype(int)
    y = np.floor((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n).astype(int)
    return x, y

def tile_bounds(x, y):
    n = 2 ** Z
    lon0, lon1 = x / n * 360 - 180, (x + 1) / n * 360 - 180
    latn = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))
    lats = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 1) / n))))
    return lon0, lats, lon1, latn

# 50 m: percentile among all metro cells
order = np.argsort(pred)
p_fine = np.empty_like(pred); p_fine[order] = (np.arange(len(pred)) + 0.5) / len(pred) * 100

# RWI: percentile among RWI tiles that contain predicted cells (same metro footprint)
ctx, cty = tile_xy(cx, cy)
ckey = ctx * 100000 + cty
rtx, rty = tile_xy(rlon, rlat)
rkey = rtx * 100000 + rty
assert len(np.unique(rkey)) == len(rkey), "duplicate RWI tiles"
metro = np.isin(rkey, np.unique(ckey))
print(f"RWI tiles in file window: {len(rkey)}, covering predicted cells: {metro.sum()}")
mk, mv = rkey[metro], rwi[metro]
o = np.argsort(mv); p_r = np.empty_like(mv); p_r[o] = (np.arange(len(mv)) + 0.5) / len(mv) * 100
lut = dict(zip(mk, p_r))
p_left = np.array([lut.get(k, np.nan) for k in ckey])
if MODE == "own":
    # same 50 m predictions (log scale), averaged within each zoom-14 tile over all its cells,
    # then ranked like the 50 m panel: every settled cell gets the percentile of its tile's mean
    uk, inv = np.unique(ckey, return_inverse=True)
    tmean = np.bincount(inv, weights=pred) / np.bincount(inv)
    v = tmean[inv]
    o = np.argsort(v, kind="stable"); r = np.empty(len(v)); r[o] = np.arange(len(v))
    # mid-rank for ties (all cells of a tile share one value)
    lo = np.searchsorted(np.sort(v), v, "left"); hi = np.searchsorted(np.sort(v), v, "right")
    p_left = (lo + hi) / 2 / len(v) * 100
    print(f"own-aggregate tiles: {len(uk)}")
print(f"cells without an RWI tile: {np.isnan(p_left).mean():.2%}")

# rasterize on the cell grid
x0 = cx.min(); y0 = cy.min()
ix = np.rint((cx - x0) / STEP).astype(int); iy = np.rint((cy - y0) / STEP).astype(int)
assert np.allclose(x0 + ix * STEP, cx, atol=1e-7) and np.allclose(y0 + iy * STEP, cy, atol=1e-7)
H, W = iy.max() + 1, ix.max() + 1
def raster(v):
    a = np.full((H, W), np.nan); a[iy, ix] = v; return a
A_fine, A_left = raster(p_fine), raster(p_left)
ext = (x0 - STEP / 2, x0 + (W - .5) * STEP, y0 - STEP / 2, y0 + (H - .5) * STEP)

# river polylines -> distance from each cell centre (metres)
segs = []
for ln in d["river_lines"]:
    p = np.array(ln["coords"])
    segs.append(np.c_[p[:-1], p[1:]])
S = np.vstack(segs)                                     # lon_a, lat_a, lon_b, lat_b
def dist_to_river(lon, lat):
    px, py = lon * KX, lat * KY
    ax_, ay_, bx_, by_ = S[:, 0] * KX, S[:, 1] * KY, S[:, 2] * KX, S[:, 3] * KY
    out = np.full(px.shape, np.inf)
    for i in range(0, len(px), 20000):
        X, Y = px[i:i + 20000, None], py[i:i + 20000, None]
        dx, dy = bx_ - ax_, by_ - ay_
        tt = np.clip(((X - ax_) * dx + (Y - ay_) * dy) / np.maximum(dx * dx + dy * dy, 1e-9), 0, 1)
        out[i:i + 20000] = np.sqrt((X - ax_ - tt * dx) ** 2 + (Y - ay_ - tt * dy) ** 2).min(1)
    return out

inview = (cx >= VIEW[0]) & (cx <= VIEW[2]) & (cy >= VIEW[1]) & (cy <= VIEW[3])
dist = np.full(len(cx), np.inf); dist[inview] = dist_to_river(cx[inview], cy[inview])
band = inview & (dist <= BUFFER)
rest = inview & ~band
print(f"cells in view {inview.sum():,}; in {BUFFER:.0f} m band {band.sum():,}")

# distance field on a regular lattice for the band outline
gx = np.arange(VIEW[0], VIEW[2], STEP); gy = np.arange(VIEW[1], VIEW[3], STEP)
GX, GY = np.meshgrid(gx, gy)
DF = dist_to_river(GX.ravel(), GY.ravel()).reshape(GX.shape)

# ---------------------------------------------------------------- stats
def pw(m):                                              # population weights per cell
    return c["pers"][m]
st = {
    "riaBuffer": f"{BUFFER:.0f}",
    "riaFine": f"{100 * np.mean(p_fine[band] < 20):.0f}",
    "riaCoarse": f"{100 * np.nanmean(p_left[band] < 20):.0f}",
    "riaPctFine": f"{np.median(p_fine[band]):.0f}",
    "riaPctCoarse": f"{np.nanmedian(p_left[band]):.0f}",
    "riaPctRest": f"{np.median(p_fine[rest]):.0f}",
    "riaPctRestCoarse": f"{np.nanmedian(p_left[rest]):.0f}",
    "riaNcells": f"{band.sum():,}".replace(",", "{,}"),
    "riaTiles": f"{len(set(ckey[band]))}",
}
# how many RWI tiles intersect the band, and the spread of 50 m values inside each
for k, v in st.items(): print(f"  {k:16s} {v}")
tb = ckey[band]
spread = [np.subtract(*np.percentile(p_fine[inview & (ckey == k)], [90, 10])) for k in np.unique(tb)]
print("  within-tile P90-P10 of 50 m percentiles, band tiles:", np.round(np.median(spread)))
st["riaSpread"] = f"{np.median(spread):.0f}"
with open(f"{OUT}/riachuelo_stats{SUF}.tex", "w") as f:
    f.write("% riachuelo_resolution figure: numbers for slide 02 (make_fig.py)\n")
    for k, v in st.items(): f.write(f"\\def\\{k}{{{v}}}\n")

# ---------------------------------------------------------------- plot
cmap = ListedColormap(matplotlib.colormaps["Spectral"](np.linspace(0, 1, 10)))
cmap.set_bad("#cccccc")
norm = BoundaryNorm(np.arange(0, 101, 10), cmap.N)
LINE = "#2b2b2b"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman"],
                     "mathtext.fontset": "cm", "font.size": 7, "pdf.fonttype": 42,
                     "axes.unicode_minus": False})
# physical size = size on the slide (right column ~ 9 cm), so fonts render at true size
PW = 1.50                                              # panel width, inches
ASP = (VIEW[3] - VIEW[1]) / ((VIEW[2] - VIEW[0]) * np.cos(np.radians(LAT0)))
PH = PW * ASP
L_, GAPX, CBG, CBW, CBL = 0.02, 0.08, 0.10, 0.09, 0.30 # inches
TOP, BOT = 0.20, 0.03
FW = L_ + PW + GAPX + PW + CBG + CBW + CBL
FH = TOP + PH + BOT
fig = plt.figure(figsize=(FW, FH))
axes = [fig.add_axes([L_ / FW, BOT / FH, PW / FW, PH / FH]),
        fig.add_axes([(L_ + PW + GAPX) / FW, BOT / FH, PW / FW, PH / FH])]
cax = fig.add_axes([(L_ + 2 * PW + GAPX + CBG) / FW, BOT / FH, CBW / FW, PH / FH])

tiles = [(x, y) for x in range(TX0, TX1 + 1) for y in range(TY0, TY1 + 1)]
titles = (["Meta RWI, 2.4 km", "Income prediction, 50 m"] if MODE == "rwi"
          else ["Income prediction, 2.4 km", "Income prediction, 50 m"])
for ax, A, title in zip(axes, [A_left, A_fine], titles):
    ax.imshow(np.ma.masked_invalid(A), origin="lower", extent=ext, cmap=cmap, norm=norm,
              interpolation="nearest", rasterized=True)
    pc = [Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1]) for b in (tile_bounds(a, bb) for a, bb in tiles)]
    ax.add_collection(PatchCollection(pc, facecolor="none", edgecolor="white" if A is A_left else (1, 1, 1, 0.55),
                                      linewidth=0.6 if A is A_left else 0.45))
    ax.contour(GX, GY, DF, levels=[BUFFER], colors=[LINE], linewidths=0.7, linestyles=[(0, (2.5, 1.8))])
    for s_ in segs:
        ax.plot(np.r_[s_[:, 0], s_[-1, 2]], np.r_[s_[:, 1], s_[-1, 3]], color=LINE, lw=1.2,
                solid_capstyle="round", solid_joinstyle="round")
    ax.set_xlim(VIEW[0], VIEW[2]); ax.set_ylim(VIEW[1], VIEW[3])
    ax.set_aspect(1 / np.cos(np.radians(LAT0)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, fontsize=7.5, pad=3)

# north arrow (right panel, top right) and 2 km scale bar (left panel, bottom left)
ax = axes[1]
nx, ny = VIEW[2] - 0.0050, VIEW[3] - 0.0155
ax.add_patch(Rectangle((nx - 0.0032, ny - 0.0018), 0.0064, 0.0158, fc="white", ec="none", alpha=0.85))
ax.annotate("", xy=(nx, ny + 0.0085), xytext=(nx, ny),
            arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.25", color="black", lw=0.9))
ax.text(nx, ny + 0.0095, "N", ha="center", va="bottom", fontsize=6.5)
ax = axes[0]
L = 2000 / KX; bx, by, bh = VIEW[0] + 0.0035, VIEW[1] + 0.0050, 0.0008
ax.add_patch(Rectangle((bx - 0.0022, by - 0.0042), L + 0.0074, 0.0075, fc="white", ec="none", alpha=0.85))
ax.add_patch(Rectangle((bx, by), L / 2, bh, fc="white", ec="black", lw=0.4))
ax.add_patch(Rectangle((bx + L / 2, by), L / 2, bh, fc="black", ec="black", lw=0.4))
ax.text(bx, by - 0.0005, "0", ha="center", va="top", fontsize=5.8)
ax.text(bx + L, by - 0.0005, "2 km", ha="center", va="top", fontsize=5.8)

cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical",
                  ticks=np.arange(5, 100, 10))
cb.set_ticklabels([f"D{i}" for i in range(1, 11)])
cax.set_yticks(np.arange(0, 101, 10), minor=True)
cax.tick_params(which="major", length=0, labelsize=6.3, pad=2)
cax.tick_params(which="minor", length=2, width=0.4)
cb.outline.set_linewidth(0.5)
cax.set_title("decile", fontsize=6.3, pad=3)
for ext_ in ("pdf", "png"):
    fig.savefig(f"{OUT}/riachuelo_resolution{SUF}.{ext_}", dpi=400, transparent=True)
print("saved", round(FW * 2.54, 2), "x", round(FH * 2.54, 2), "cm")
