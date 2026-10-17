# Slide 02 figure: Riachuelo, 2.4 km vs 50 m

- `riachuelo_resolution.pdf`: the figure on slide 02.
- `riachuelo_stats.tex`: numbers the slide prints (\riaPctCoarse etc.), for the cropped window.
- `make_slide_fig.py`: rebuilds both. Needs numpy + matplotlib only.
      python build_cells.py "<data_out>/predictions_grid_eph_income_estimates_2022.parquet" cells.npz
      python make_slide_fig.py cells.npz riachuelo_rwi_river.json <outdir> own
  (writes riachuelo_resolution_own_slide.pdf / riachuelo_stats_own_slide.tex; rename to the files here)
  `own` = left panel is your own predictions at 2.4 km; `rwi` = left panel is Meta's RWI.
- `riachuelo_rwi_river.json`: Meta RWI rows for the Buenos Aires metro (HDX, arg_relative_wealth_index.csv)
  and Riachuelo/Matanza centrelines (OpenStreetMap, Overpass), fetched 2026-10-16.

Left panel = the same 2022 predictions (Abbate et al. 2025) averaged within Bing zoom-14 tiles (the RWI grid,
~2 km here); each settled cell takes its tile's metro-wide percentile. Right panel = 50 m cells, metro-wide percentile.
Crop = 4 x 4 tiles (x 5533-5536, y 9874-9877): the block with the largest within-tile spread of 50 m values.
