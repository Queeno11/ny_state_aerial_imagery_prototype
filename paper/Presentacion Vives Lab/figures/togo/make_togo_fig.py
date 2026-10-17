"""Full-screen map: Meta Relative Wealth Index for Togo (Chi et al. 2022), 2.4 km tiles,
deciles within Togo, Spectral (D1 red = poorest). numpy + matplotlib only.
Usage: python make_togo_fig.py togo_rwi_border.json <outdir>  (data: HDX + OSM, fetched 2026-10-16)"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon

SRC, OUT = sys.argv[1], sys.argv[2]
Z = 14
d = json.load(open(SRC))
rows = [l.split(",") for l in d["rwi_csv"].strip().split("\n")]
hdr = rows[0]; rw = np.array(rows[1:], float)
lat, lon, rwi = rw[:, hdr.index("latitude")], rw[:, hdr.index("longitude")], rw[:, hdr.index("rwi")]

n = 2 ** Z
tx = np.floor((lon + 180) / 360 * n).astype(int)
ty = np.floor((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n).astype(int)
assert len(set(zip(tx, ty))) == len(tx), "duplicate tiles"
o = np.argsort(rwi); pct = np.empty(len(rwi)); pct[o] = (np.arange(len(rwi)) + 0.5) / len(rwi) * 100
print(f"{len(rwi)} tiles; rwi range {rwi.min():.2f} to {rwi.max():.2f}")

# tile raster (regular in tile index) drawn with true tile edges
x0, x1, y0, y1 = tx.min(), tx.max(), ty.min(), ty.max()
A = np.full((y1 - y0 + 1, x1 - x0 + 1), np.nan)
A[ty - y0, tx - x0] = pct
lon_e = np.arange(x0, x1 + 2) / n * 360 - 180
lat_e = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * np.arange(y0, y1 + 2) / n))))

# border: OSM land boundary (maritime ways removed) closed with the real coastline
def stitch(ways):
    pool = [list(map(tuple, w)) for w in ways]
    chains = []
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
landc = stitch(d["border_ways"]); coastc = stitch(d["coast_ways"])
print(f"land chains {len(landc)} {[len(c) for c in landc[:3]]}; coast chains {len(coastc)} {[len(c) for c in coastc[:3]]}")
land, coast = landc[0], coastc[0]
def nearest(p):
    return int(np.argmin(((coast - p) ** 2).sum(1)))
ia, ib = nearest(land[-1]), nearest(land[0])
print("land ends", land[0], land[-1], "-> coast idx", ib, ia, "gaps (deg)",
      np.abs(coast[ib] - land[0]).max().round(4), np.abs(coast[ia] - land[-1]).max().round(4))
seg = coast[ia:ib + 1] if ia <= ib else coast[ib:ia + 1][::-1]
main = np.vstack([land, seg, land[:1]])
rings = [main]
LAT0 = 8.6
cmap = ListedColormap(matplotlib.colormaps["Spectral"](np.linspace(0, 1, 10)))
norm = BoundaryNorm(np.arange(0, 101, 10), cmap.N)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman"],
                     "font.size": 8, "pdf.fonttype": 42, "axes.unicode_minus": False})

pad = 0.06
W_, S_, E_, N_ = main[:, 0].min() - pad, main[:, 1].min() - pad, main[:, 0].max() + pad, main[:, 1].max() + pad
asp = (N_ - S_) / ((E_ - W_) * np.cos(np.radians(LAT0)))
MH = 3.15                      # map height, inches (full-screen 16:9 slide is 3.54 in tall)
MW = MH / asp
CBG, CBW, CBL, LM = 0.30, 0.11, 0.45, 0.02
FW, FH = LM + MW + CBG + CBW + CBL, MH + 0.04
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([LM / FW, 0.02 / FH, MW / FW, MH / FH])
cax = fig.add_axes([(LM + MW + CBG) / FW, 0.02 / FH + MH * 0.18 / FH, CBW / FW, MH * 0.64 / FH])

ax.add_patch(Polygon(main, closed=True, fc="#cccccc", ec="none", zorder=0))      # land with no estimate
ax.pcolormesh(lon_e, lat_e, np.ma.masked_invalid(A), cmap=cmap, norm=norm, shading="flat",
              rasterized=True, zorder=1, edgecolors="none")
for r in rings:
    ax.plot(r[:, 0], r[:, 1], color="#2b2b2b", lw=0.9, zorder=2, solid_joinstyle="round")
ax.set_xlim(W_, E_); ax.set_ylim(S_, N_)
ax.set_aspect(1 / np.cos(np.radians(LAT0)))
ax.set_axis_off()

cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical", ticks=np.arange(5, 100, 10))
cb.set_ticklabels([f"D{i}" for i in range(1, 11)])
cax.set_yticks(np.arange(0, 101, 10), minor=True)
cax.tick_params(which="major", length=0, labelsize=7.5, pad=3)
cax.tick_params(which="minor", length=2.5, width=0.5)
cb.outline.set_linewidth(0.6)
cax.set_title("RWI decile", fontsize=7.5, pad=4)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/togo_rwi.{ext}", dpi=400, transparent=True)
print("saved", round(FW * 2.54, 2), "x", round(FH * 2.54, 2), "cm")
