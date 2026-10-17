"""Read the 2022 50 m prediction grid (geoparquet) without pyarrow/geopandas and save cells.npz.
Usage: python build_cells.py <predictions_grid_eph_income_estimates_2022.parquet> cells.npz"""
import sys, struct
import numpy as np
import miniparquet as mp

d, _ = mp.read(sys.argv[1], ["id", "prediction", "personas", "geometry", "link"])
def ring(b):
    bo = "<" if b[0] == 1 else ">"
    _, _, npts = struct.unpack(bo + "III", b[1:13])
    return np.frombuffer(b[13:13 + 16 * npts], dtype=bo + "f8").reshape(-1, 2)
R = np.stack([ring(g) for g in d["geometry"]])
xmin, xmax = R[:, :, 0].min(1), R[:, :, 0].max(1)
ymin, ymax = R[:, :, 1].min(1), R[:, :, 1].max(1)
np.savez(sys.argv[2], id=np.array(d["id"]), pred=np.array(d["prediction"], float),
         pers=np.array(d["personas"], float), cx=(xmin + xmax) / 2, cy=(ymin + ymax) / 2,
         w=xmax - xmin, h=ymax - ymin, link=np.array([l.decode() if l else "" for l in d["link"]]))
print(len(R), "cells")
