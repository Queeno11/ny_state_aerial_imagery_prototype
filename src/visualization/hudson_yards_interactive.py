"""
src/visualization/hudson_yards_interactive.py

Interactive Folium map for the Hudson Yards case study.

Run from WSL conda environment:
    cd /path/to/project
    conda activate torch_geo_env
    python -m src.visualization.hudson_yards_interactive [--savename NAME]

Output: results/<savename>/evaluation/figures/D_hudson_yards_interactive.html
"""

from __future__ import annotations

import argparse
import base64
import json
import warnings
from io import BytesIO
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from pyproj import Transformer
from shapely.geometry import box as shapely_box

from src.utils.paths import (
    ACS_ROOT_DIR,
    IMAGERY_ROOT,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
)
from src.geo_utils import calculate_exact_tau

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────

YEARS = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
CRS_PROJ = 6539
CRS_GEO = 4326
_M_PER_FT = 0.3048006096

TAU_METERS = 100
IMAGE_SIZE = 224
_EXACT_TAU_M, _ = calculate_exact_tau(TAU_METERS, IMAGE_SIZE)
TAU_FT = _EXACT_TAU_M / _M_PER_FT

CASE_STUDY = {"lon": -74.0015, "lat": 40.7538, "half_km": 0.6}

DEFAULT_SAVENAME = (
    "scalemae_lr0.0001_size224_y2010-2012-2014-2016-2018-2020-2022-2024_ranknet_mining_lambda_s_05"
)

# ─── Coordinate helpers ───────────────────────────────────────────────────────

def _to_6539(lon: float, lat: float) -> tuple[float, float]:
    t = Transformer.from_crs(CRS_GEO, CRS_PROJ, always_xy=True)
    return t.transform(lon, lat)


def _to_4326(x: float, y: float) -> tuple[float, float]:
    t = Transformer.from_crs(CRS_PROJ, CRS_GEO, always_xy=True)
    return t.transform(x, y)


def _norm_geoid(x) -> str:
    return str(int(x)).zfill(11)


# ─── Imagery helpers ──────────────────────────────────────────────────────────

def _slice_zarr(ds, cx: float, cy: float, half_ft: float,
                max_px: int = 800) -> np.ndarray | None:
    """Extract (4, H, W) uint8 tile from zarr Dataset."""
    try:
        x_vals = ds.x.values
        y_vals = ds.y.values  # descending

        xi0 = int(np.searchsorted(x_vals, cx - half_ft, side="left"))
        xi1 = int(np.searchsorted(x_vals, cx + half_ft, side="right"))
        yi0 = int(np.searchsorted(-y_vals, -(cy + half_ft), side="left"))
        yi1 = int(np.searchsorted(-y_vals, -(cy - half_ft), side="right"))

        if xi1 <= xi0 or yi1 <= yi0:
            return None
        step = max(1, int(np.ceil(max(xi1 - xi0, yi1 - yi0) / max_px)))
        tile = ds["value"].isel(
            y=slice(yi0, yi1, step), x=slice(xi0, xi1, step)
        ).compute().values
        return np.nan_to_num(tile, nan=0).clip(0, 255).astype(np.uint8)
    except Exception as e:
        print(f"      zarr slice error: {e}")
        return None


def _stretch_rgb(tile: np.ndarray) -> np.ndarray:
    """(4, H, W) uint8 → (H, W, 3) uint8 with percentile stretch."""
    rgb = np.stack([tile[0], tile[1], tile[2]], axis=-1).astype(float)
    for ch in range(3):
        lo, hi = np.percentile(rgb[:, :, ch], [2, 98])
        rgb[:, :, ch] = np.clip(
            (rgb[:, :, ch] - lo) / max(hi - lo, 1.0) * 255, 0, 255
        )
    return rgb.astype(np.uint8)


def _to_b64_png(rgb: np.ndarray) -> str:
    """(H, W, 3) uint8 → base64 PNG string."""
    img = Image.fromarray(rgb)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _spectral_hex(val: float, vmin: float, vmax: float) -> str:
    """Map val to Spectral colormap hex."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return mcolors.to_hex(cm.Spectral(norm(float(val))))


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(results_dir: Path, processed_dir: Path) -> dict:
    """Load all case-study data into a single dict."""

    lon, lat = CASE_STUDY["lon"], CASE_STUDY["lat"]
    half_km = CASE_STUDY["half_km"]
    cx, cy = _to_6539(lon, lat)
    half_ft = half_km * 1000.0 / _M_PER_FT

    # WGS84 bounding box corners (SW and NE)
    lon0, lat0 = _to_4326(cx - half_ft, cy - half_ft)
    lon1, lat1 = _to_4326(cx + half_ft, cy + half_ft)
    box_4326 = shapely_box(lon0, lat0, lon1, lat1)
    box_proj = shapely_box(cx - half_ft, cy - half_ft, cx + half_ft, cy + half_ft)

    print(f"  Box WGS84: [{lat0:.4f},{lon0:.4f}] → [{lat1:.4f},{lon1:.4f}]")

    # Buildings in box (4326)
    print("  Loading buildings_nyc.parquet …")
    bldg_nyc = gpd.read_parquet(processed_dir / "buildings_nyc.parquet")
    bldg_box = bldg_nyc[bldg_nyc.intersects(box_4326)].copy()
    box_ids = set(bldg_box.index.tolist())
    print(f"    {len(box_ids)} buildings in box")

    # Predictions per year
    print("  Loading predictions by year …")
    pred_by_yr: dict[int, gpd.GeoDataFrame] = {}
    all_vals: list[float] = []
    for yr in YEARS:
        df = gpd.read_parquet(results_dir / f"predictions_{yr}.parquet")
        sub = df[df.index.isin(box_ids)].copy()
        if len(sub):
            pred_by_yr[yr] = sub
            all_vals.extend(sub["predicted_value"].dropna().tolist())
        print(f"    {yr}: {len(sub)} buildings")

    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))
    print(f"  Prediction range: [{vmin:.3f}, {vmax:.3f}]")

    # All-year building trajectories: {str(doitt_id): {str(yr): value}}
    trajectories: dict[str, dict[str, float]] = {}
    for yr, df in pred_by_yr.items():
        for did, row in df.iterrows():
            v = row["predicted_value"]
            if not pd.isna(v):
                trajectories.setdefault(str(did), {})[str(yr)] = round(float(v), 4)

    # Tract geometries overlapping box
    print("  Loading tract splits …")
    splits = gpd.read_feather(processed_dir / "tract_splits.feather")
    splits["GEOID_str"] = splits["GEOID"].astype(str).str.zfill(11)
    splits_proj = splits.to_crs(CRS_PROJ)
    overlap = splits_proj[splits_proj.intersects(box_proj)].copy()
    overlap_geoids = set(overlap["GEOID_str"])
    tracts_4326 = splits[splits["GEOID_str"].isin(overlap_geoids)].to_crs(CRS_GEO)
    print(f"    {len(tracts_4326)} overlapping tracts")

    # ACS panel data for overlapping tracts
    print("  Loading ACS panel …")
    panel = pd.read_feather(processed_dir / "ny_tracts_panel_2009_2014_2019_2024.feather")
    panel["GEOID_str"] = panel["geoid_2024"].astype(str).str.zfill(11)
    acs_by_geoid: dict[str, dict[str, float]] = {}
    for _, row in panel[panel["GEOID_str"].isin(overlap_geoids)].iterrows():
        gid = row["GEOID_str"]
        acs: dict[str, float] = {}
        for yr in [2009, 2014, 2019, 2024]:
            col = f"per_capita_income_usd_{yr}"
            if col in panel.columns:
                v = row[col]
                if not pd.isna(v):
                    acs[str(yr)] = round(float(v), 2)
        acs_by_geoid[gid] = acs

    return dict(
        cx=cx, cy=cy, half_ft=half_ft,
        lon0=lon0, lat0=lat0, lon1=lon1, lat1=lat1,
        bldg_box=bldg_box,
        pred_by_yr=pred_by_yr,
        vmin=vmin, vmax=vmax,
        trajectories=trajectories,
        tracts_4326=tracts_4326,
        acs_by_geoid=acs_by_geoid,
        overlap_geoids=overlap_geoids,
    )


def extract_images(cx: float, cy: float, half_ft: float) -> dict[int, str]:
    """Extract zarr tiles for all years as base64 PNG strings."""
    images: dict[int, str] = {}
    for yr in YEARS:
        zarr_path = Path(str(IMAGERY_ROOT)) / f"nyc_{yr}.zarr"
        if not zarr_path.exists():
            print(f"    zarr {yr}: not found")
            continue
        try:
            ds = xr.open_zarr(str(zarr_path), chunks="auto")
            tile = _slice_zarr(ds, cx, cy, half_ft, max_px=1000)
            if tile is None or tile.ndim != 3 or tile.shape[0] < 3:
                print(f"    zarr {yr}: bad tile")
                continue
            rgb = _stretch_rgb(tile)
            images[yr] = _to_b64_png(rgb)
            print(f"    zarr {yr}: {rgb.shape[1]}×{rgb.shape[0]} px")
        except Exception as e:
            print(f"    zarr {yr}: {e}")
    return images


def build_geojsons(bldg_box: gpd.GeoDataFrame,
                   pred_by_yr: dict[int, gpd.GeoDataFrame],
                   vmin: float, vmax: float) -> dict[int, dict]:
    """GeoJSON FeatureCollection per year (4326), with style properties."""
    geojsons: dict[int, dict] = {}
    for yr in YEARS:
        if yr not in pred_by_yr:
            continue
        df = pred_by_yr[yr].to_crs(CRS_GEO)
        df = df.join(bldg_box[["CONSTRUCTION_YEAR", "DEMOLITION_YEAR"]], how="left")

        cy_col = df["CONSTRUCTION_YEAR"].fillna(0)
        dy_col = df["DEMOLITION_YEAR"].fillna(9999)
        df = df[(cy_col <= yr) & (dy_col > yr)].copy()

        df["fill_color"] = df["predicted_value"].apply(
            lambda v: _spectral_hex(v, vmin, vmax) if pd.notna(v) else "#888888"
        )
        df["is_new"] = df["CONSTRUCTION_YEAR"].apply(
            lambda v: bool(not pd.isna(v) and float(v) > 2009)
        )
        df["doitt_id"] = df.index.astype(int)
        df["geoid_str"] = df["GEOID"].apply(_norm_geoid)
        df["pred_val"] = df["predicted_value"].round(4)
        df["const_yr"] = df["CONSTRUCTION_YEAR"].apply(
            lambda v: int(v) if pd.notna(v) else None
        )

        keep = ["geometry", "fill_color", "is_new", "doitt_id",
                "geoid_str", "pred_val", "const_yr"]
        gj = json.loads(df[keep].to_json())
        geojsons[yr] = gj
        print(f"    {yr}: {len(gj['features'])} features")
    return geojsons


# ─── Map builder ──────────────────────────────────────────────────────────────

_CSS = """
<style>
.year-selector-ctrl {
    background: white;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    padding: 10px 12px;
    font-family: Arial, sans-serif;
    min-width: 200px;
}
.year-selector-ctrl h4 {
    margin: 0 0 8px 0;
    font-size: 12px;
    font-weight: 700;
    color: #333;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.year-btn-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 4px;
}
.year-btn {
    padding: 4px 0;
    font-size: 11px;
    font-weight: 600;
    border: 1.5px solid #aaa;
    border-radius: 4px;
    background: #f5f5f5;
    color: #555;
    cursor: pointer;
    text-align: center;
    transition: all 0.15s;
}
.year-btn:hover { background: #e0e8f5; border-color: #4a8cca; color: #2255aa; }
.year-btn.active { background: #2255aa; border-color: #2255aa; color: white; }
.img-toggle-row {
    display: flex;
    align-items: center;
    margin-top: 8px;
    font-size: 11px;
    color: #555;
    gap: 6px;
}
.img-toggle-row input { cursor: pointer; }

.bldg-popup {
    font-family: Arial, sans-serif;
    font-size: 12px;
    width: 400px;
}
.bldg-popup h4 {
    margin: 0 0 6px 0;
    font-size: 13px;
    color: #222;
}
.bldg-popup .meta {
    color: #555;
    margin-bottom: 10px;
    font-size: 11px;
    line-height: 1.6;
}
.bldg-popup .new-tag {
    display: inline-block;
    background: #c0392b;
    color: white;
    font-size: 9px;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 3px;
    vertical-align: middle;
    letter-spacing: 0.5px;
}
.chart-title {
    font-size: 11px;
    font-weight: 700;
    color: #333;
    margin: 8px 0 2px 0;
}
.chart-container { width: 400px; overflow-x: hidden; }

/* ── Draggable floating panel ── */
.bldg-panel {
    position: absolute;
    z-index: 1200;
    background: white;
    border-radius: 8px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
    width: 440px;
    max-width: 90%;
    overflow: hidden;
    font-family: Arial, sans-serif;
}
.bldg-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #2255aa;
    color: white;
    padding: 6px 10px;
    cursor: move;
    user-select: none;
    -webkit-user-select: none;
    touch-action: none;
}
.bldg-panel-header .drag-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 6px;
}
.bldg-panel-header .drag-title::before {
    content: "\2630";
    font-size: 12px;
    opacity: 0.8;
}
.bldg-panel-close {
    cursor: pointer;
    font-size: 16px;
    line-height: 1;
    font-weight: 700;
    padding: 0 4px;
    border-radius: 3px;
}
.bldg-panel-close:hover { background: rgba(255,255,255,0.25); }
.bldg-panel-body {
    padding: 10px 12px;
    max-height: 70vh;
    overflow-y: auto;
}
</style>
"""

_JS_CHARTS = r"""
function svgLineChart(entries, yLabel, color, width, height, isDots) {
    if (!entries || entries.length === 0) return '<p style="color:#aaa;font-size:11px">No data</p>';
    var mL = 58, mR = 12, mT = 8, mB = 32;
    var w = width - mL - mR, h = height - mT - mB;

    var xs = entries.map(function(e){return e.x;});
    var ys = entries.map(function(e){return e.y;});
    var xMin = Math.min.apply(null,xs), xMax = Math.max.apply(null,xs);
    var ySpan = Math.max.apply(null,ys) - Math.min.apply(null,ys);
    var yPad  = ySpan * 0.12 || 0.5;
    var yMin  = Math.min.apply(null,ys) - yPad;
    var yMax  = Math.max.apply(null,ys) + yPad;

    function px(v){ return mL + (xMax===xMin ? w/2 : (v-xMin)/(xMax-xMin)*w); }
    function py(v){ return mT + h - (yMax===yMin ? h/2 : (v-yMin)/(yMax-yMin)*h); }

    var pathD = entries.map(function(e,i){
        return (i===0?'M':'L')+px(e.x).toFixed(1)+','+py(e.y).toFixed(1);
    }).join(' ');

    var dots = entries.map(function(e){
        return '<circle cx="'+px(e.x).toFixed(1)+'" cy="'+py(e.y).toFixed(1)+
               '" r="4" fill="'+color+'" stroke="white" stroke-width="1.2"/>';
    }).join('');

    // x-axis labels
    var xLabels = entries.map(function(e){
        var lbl = String(e.x).slice(-2); // last 2 digits of year
        return '<text x="'+px(e.x).toFixed(1)+'" y="'+(mT+h+18)+
               '" text-anchor="middle" font-size="9" fill="#555">'+lbl+'</text>'+
               '<line x1="'+px(e.x).toFixed(1)+'" y1="'+(mT+h)+
               '" x2="'+px(e.x).toFixed(1)+'" y2="'+(mT+h+4)+'" stroke="#aaa"/>';
    }).join('');

    // y-axis ticks (4)
    var yTicks = '';
    for (var i=0; i<=3; i++) {
        var tv = yMin + (yMax-yMin)*i/3;
        var ty = py(tv);
        var fmt = Math.abs(tv) >= 10000
            ? (tv/1000).toFixed(0)+'k'
            : tv.toFixed(2);
        yTicks += '<line x1="'+(mL-4)+'" y1="'+ty.toFixed(1)+'" x2="'+mL+'" y2="'+ty.toFixed(1)+'" stroke="#aaa"/>'+
                  '<text x="'+(mL-6)+'" y="'+(ty+3).toFixed(1)+
                  '" text-anchor="end" font-size="9" fill="#555">'+fmt+'</text>';
    }

    var line = isDots ? '' :
        '<path d="'+pathD+'" fill="none" stroke="'+color+'" stroke-width="2" stroke-linejoin="round"/>';

    return '<svg width="'+width+'" height="'+height+'" xmlns="http://www.w3.org/2000/svg">'+
        '<line x1="'+mL+'" y1="'+mT+'" x2="'+mL+'" y2="'+(mT+h)+'" stroke="#ccc"/>'+
        '<line x1="'+mL+'" y1="'+(mT+h)+'" x2="'+(mL+w)+'" y2="'+(mT+h)+'" stroke="#ccc"/>'+
        yTicks + xLabels + line + dots +
        '</svg>';
}

function makePopupHtml(props) {
    var did   = props.doitt_id;
    var geoid = props.geoid_str;
    var traj  = TRAJECTORIES[String(did)] || {};
    var acs   = ACS_DATA[String(geoid)] || {};

    var html = '<div class="bldg-popup">';
    html += '<h4>Building ' + did + '</h4>';
    html += '<div class="meta">';
    html += 'Predicted value: <b>' + (props.pred_val !== null ? props.pred_val.toFixed(3) : '—') + '</b><br>';
    if (props.const_yr) {
        html += 'Built: <b>' + props.const_yr + '</b>';
        if (props.is_new) html += ' <span class="new-tag">NEW</span>';
        html += '<br>';
    }
    html += 'Tract: <b>' + geoid + '</b>';
    html += '</div>';

    // Chart 1: prediction trajectory
    html += '<div class="chart-title">Predicted value over time</div>';
    html += '<div class="chart-container">';
    var trajEntries = Object.keys(traj).map(function(k){
        return {x: parseInt(k), y: traj[k]};
    }).sort(function(a,b){return a.x-b.x;});
    html += svgLineChart(trajEntries, 'Predicted value', '#3275b5', 400, 130, false);
    html += '</div>';

    // Chart 2: ACS income
    html += '<div class="chart-title">ACS per-capita income (USD) &mdash; tract average</div>';
    html += '<div class="chart-container">';
    var acsEntries = Object.keys(acs).map(function(k){
        return {x: parseInt(k), y: acs[k]};
    }).sort(function(a,b){return a.x-b.x;});
    if (acsEntries.length === 0) {
        html += '<p style="color:#aaa;font-size:11px;margin:4px 0">No ACS data for this tract</p>';
    } else {
        html += svgLineChart(acsEntries, 'Income (USD)', '#c0392b', 400, 130, false);
    }
    html += '</div>';
    html += '</div>';
    return html;
}
"""

_JS_MAP = r"""
(function() {
    var map;  // assigned in init() after all scripts have run
    var YEARS_LIST = YEARS_DATA;
    var imgBounds  = [[IMG_LAT0, IMG_LON0], [IMG_LAT1, IMG_LON1]];
    var curYear    = YEARS_LIST[YEARS_LIST.length - 1];
    var showImg    = true;

        /* ── Draggable floating panel (replaces Leaflet popup) ── */
    var bldgPanel = null;

    function closeDraggablePanel() {
        if (bldgPanel && bldgPanel.parentNode) {
            bldgPanel.parentNode.removeChild(bldgPanel);
        }
        bldgPanel = null;
    }

    function showDraggablePanel(bodyHtml, originalEvent) {
        closeDraggablePanel();

        var container = map.getContainer();
        var panel = document.createElement('div');
        panel.className = 'bldg-panel';
        panel.innerHTML =
            '<div class="bldg-panel-header">' +
                '<span class="drag-title">Building detail \u2014 drag to move</span>' +
                '<span class="bldg-panel-close" title="Close">\u2715</span>' +
            '</div>' +
            '<div class="bldg-panel-body"></div>';
        panel.querySelector('.bldg-panel-body').innerHTML = bodyHtml;

        // Stop map from grabbing scroll/clicks inside the panel
        L.DomEvent.disableClickPropagation(panel);
        L.DomEvent.disableScrollPropagation(panel);

        container.appendChild(panel);
        bldgPanel = panel;

        // Initial position: near the click, but clamped inside the map
        var size = map.getSize();
        var pw = panel.offsetWidth || 440;
        var ph = panel.offsetHeight || 300;
        var startLeft = size.x / 2 - pw / 2;
        var startTop  = 60;
        if (originalEvent) {
            var rect = container.getBoundingClientRect();
            startLeft = (originalEvent.clientX - rect.left) + 14;
            startTop  = (originalEvent.clientY - rect.top) + 14;
        }
        startLeft = Math.max(4, Math.min(startLeft, size.x - pw - 4));
        startTop  = Math.max(4, Math.min(startTop,  size.y - 40));
        panel.style.left = startLeft + 'px';
        panel.style.top  = startTop + 'px';

        // Close button
        panel.querySelector('.bldg-panel-close').addEventListener('click', closeDraggablePanel);

        // Drag behaviour on the header
        var header = panel.querySelector('.bldg-panel-header');
        var dragging = false, offX = 0, offY = 0;

        function pointerXY(ev) {
            if (ev.touches && ev.touches.length) {
                return {x: ev.touches[0].clientX, y: ev.touches[0].clientY};
            }
            return {x: ev.clientX, y: ev.clientY};
        }

        function onDown(ev) {
            dragging = true;
            var p = pointerXY(ev);
            offX = p.x - panel.offsetLeft;
            offY = p.y - panel.offsetTop;
            // Disable map dragging while moving the panel
            if (map.dragging) map.dragging.disable();
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
            document.addEventListener('touchmove', onMove, {passive: false});
            document.addEventListener('touchend', onUp);
            ev.preventDefault();
        }

        function onMove(ev) {
            if (!dragging) return;
            var p = pointerXY(ev);
            var s = map.getSize();
            var nl = p.x - offX;
            var nt = p.y - offY;
            nl = Math.max(-panel.offsetWidth + 60, Math.min(nl, s.x - 60));
            nt = Math.max(0, Math.min(nt, s.y - 32));
            panel.style.left = nl + 'px';
            panel.style.top  = nt + 'px';
            if (ev.cancelable) ev.preventDefault();
        }

        function onUp() {
            dragging = false;
            if (map.dragging) map.dragging.enable();
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            document.removeEventListener('touchmove', onMove);
            document.removeEventListener('touchend', onUp);
        }

        header.addEventListener('mousedown', onDown);
        header.addEventListener('touchstart', onDown, {passive: false});
    }

    /* ── Layer stores ── */
    var imgLayers  = {};   // {yr: L.imageOverlay | null}
    var bldgLayers = {};   // {yr: L.geoJSON}
    var tractLayer = null;
    var selectedBldgLayer = null;  // track currently selected building
    var selectedBldgOverlay = null;  // persistent overlay for selected building

    /* ── Color scale legend ── */
    function addLegend() {
        var legend = L.control({position: 'bottomright'});
        legend.onAdd = function() {
            var div = L.DomUtil.create('div');
            div.style.cssText = 'background:white;padding:8px 10px;border-radius:6px;'
                + 'box-shadow:0 2px 8px rgba(0,0,0,0.25);font-family:Arial,sans-serif;font-size:11px;';
            var stops = [];
            for (var i=0; i<=10; i++) {
                stops.push('hsl('+(i*36)+',70%,50%)');
            }
            div.innerHTML = '<div style="font-weight:700;margin-bottom:4px">predicted_value</div>'
                + '<div style="display:flex;align-items:center;gap:4px">'
                + '<span>low</span>'
                + '<div style="width:120px;height:10px;background:linear-gradient(to right,'
                + 'LEGEND_GRADIENT' + ');border-radius:2px"></div>';
                + '<span>high</span></div>';
            return div;
        };
        legend.addTo(map);
    }

    /* ── Tract layer ── */
    function addTractLayer() {
        tractLayer = L.geoJSON(TRACTS_GEOJSON, {
            style: {
                color: '#111',
                weight: 1.8,
                fillOpacity: 0,
                dashArray: null
            },
            onEachFeature: function(feature, layer) {
                var g = feature.properties.GEOID_str || feature.properties.geoid_str || '';
                layer.bindTooltip('Tract ' + g, {sticky: true, className: 'leaflet-tooltip'});
            }
        }).addTo(map);
    }

    /* ── Selected building overlay (persistent) ── */
    function initSelectedBldgOverlay() {
        selectedBldgOverlay = L.geoJSON(null, {
            interactive: false,
            style: {
                fillColor: 'transparent',
                fillOpacity: 0,
                color: '#000',
                weight: 3.5,
                dashArray: null
            }
        }).addTo(map);
        // Keep it on top
        selectedBldgOverlay.bringToFront();
    }

    function updateSelectedBldgOverlay(feature) {
        if (!selectedBldgOverlay) return;
        selectedBldgOverlay.clearLayers();
        if (feature) {
            selectedBldgOverlay.addData(feature);
            selectedBldgOverlay.bringToFront();
        }
    }

    /* ── Building layers ── */
    function buildBldgLayer(yr) {
        var gj = GEOJSONS[yr];
        if (!gj) return null;
        return L.geoJSON(gj, {
            style: function(feature) {
                var p = feature.properties;
                return {
                    fillColor:   p.fill_color || '#888',
                    fillOpacity: 0.7,
                    color:       p.is_new ? '#1a1a1a' : '#555',
                    weight:      p.is_new ? 1.5 : 0.5,
                    dashArray:   p.is_new ? '5,3' : null,
                };
            },
            onEachFeature: function(feature, layer) {
                layer.on('click', function(e) {
                    L.DomEvent.stopPropagation(e);
                    var html = makePopupHtml(feature.properties);
                    showDraggablePanel(html, e.originalEvent);
                    
                    // Update the persistent overlay with the selected building geometry
                    updateSelectedBldgOverlay(feature);
                    selectedBldgLayer = layer;
                });
            }
        });
    }

    function buildImgLayer(yr) {
        var b64 = IMAGES[yr];
        if (!b64) return null;
        return L.imageOverlay('data:image/png;base64,' + b64, imgBounds,
                              {opacity: 1.0, zIndex: 200});
    }

    /* ── Year selection ── */
    function selectYear(yr) {
        // Remove old
        if (imgLayers[curYear])  map.removeLayer(imgLayers[curYear]);
        if (bldgLayers[curYear]) map.removeLayer(bldgLayers[curYear]);
        curYear = yr;
        // Add new
        if (showImg && imgLayers[curYear]) imgLayers[curYear].addTo(map);
        if (bldgLayers[curYear]) bldgLayers[curYear].addTo(map);
        // Bring overlay to front so it stays visible above everything
        if (selectedBldgOverlay) selectedBldgOverlay.bringToFront();
        if (tractLayer) tractLayer.bringToFront();
        // Update buttons
        document.querySelectorAll('.year-btn').forEach(function(btn) {
            btn.classList.toggle('active', parseInt(btn.dataset.yr) === curYear);
        });
    }

    function toggleImage(visible) {
        showImg = visible;
        if (imgLayers[curYear]) {
            if (showImg) imgLayers[curYear].addTo(map);
            else         map.removeLayer(imgLayers[curYear]);
        }
    }

    /* ── Year selector control ── */
    function addYearSelector() {
        var ctrl = L.control({position: 'topright'});
        ctrl.onAdd = function() {
            var div = L.DomUtil.create('div', 'year-selector-ctrl leaflet-bar');
            L.DomEvent.disableClickPropagation(div);
            L.DomEvent.disableScrollPropagation(div);

            var btnHtml = YEARS_LIST.map(function(yr) {
                var active = yr === curYear ? ' active' : '';
                return '<div class="year-btn'+active+'" data-yr="'+yr+'" onclick="selectYearGlobal('+yr+')">'+yr+'</div>';
            }).join('');

            div.innerHTML = '<h4>Year</h4>'
                + '<div class="year-btn-grid">' + btnHtml + '</div>'
                + '<div class="img-toggle-row">'
                + '<input type="checkbox" id="imgToggle" checked onchange="toggleImageGlobal(this.checked)">'
                + '<label for="imgToggle" style="cursor:pointer">Show aerial imagery</label>'
                + '</div>';
            return div;
        };
        ctrl.addTo(map);
    }

    /* ── Expose to global scope for onclick handlers ── */
    window.selectYearGlobal = selectYear;
    window.toggleImageGlobal = toggleImage;

    /* ── Initialise after all folium scripts have run ── */
    function init() {
        map = MAP_OBJ;  // now safe: map variable is defined in a later script block

        // Build all layers (lazy — only create, don't add yet)
        YEARS_LIST.forEach(function(yr) {
            imgLayers[yr]  = buildImgLayer(yr);
            bldgLayers[yr] = buildBldgLayer(yr);
        });

        addTractLayer();
        initSelectedBldgOverlay();
        addYearSelector();
        addLegend();

        // Show the default (last) year
        if (showImg && imgLayers[curYear]) imgLayers[curYear].addTo(map);
        if (bldgLayers[curYear]) bldgLayers[curYear].addTo(map);
    }

    window.addEventListener('load', init);
})();
"""


def build_map(data: dict, images_b64: dict[int, str], geojsons: dict[int, dict]) -> folium.Map:
    """Assemble the Folium map with all interactive layers."""

    center_lat = (data["lat0"] + data["lat1"]) / 2
    center_lon = (data["lon0"] + data["lon1"]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="Cartodb Positron",
        prefer_canvas=True,
    )

    # Build Spectral legend gradient (CSS linear-gradient stops)
    grad_stops = [mcolors.to_hex(cm.Spectral(i / 9)) for i in range(10)]
    legend_gradient = ",".join(grad_stops)

    # Tracts GeoJSON (4326)
    tracts_json = data["tracts_4326"][["GEOID_str", "type", "geometry"]].to_json()

    # Only include years that have both a GeoJSON and (optionally) an image
    active_years = [yr for yr in YEARS if yr in geojsons]

    map_name = m.get_name()

    # Images JS object: {yr: b64_string}
    images_js = "{" + ",".join(
        f"{yr}: '{b64}'" for yr, b64 in images_b64.items()
    ) + "}"

    # GeoJSONs JS object (compact)
    geojsons_js = "{" + ",".join(
        f"{yr}: {json.dumps(gj, separators=(',', ':'))}"
        for yr, gj in geojsons.items()
    ) + "}"

    # Assemble custom JS: replace placeholders
    js_body = _JS_MAP \
        .replace("MAP_OBJ",      map_name) \
        .replace("YEARS_DATA",   json.dumps(active_years)) \
        .replace("IMG_LAT0",     f"{data['lat0']:.6f}") \
        .replace("IMG_LON0",     f"{data['lon0']:.6f}") \
        .replace("IMG_LAT1",     f"{data['lat1']:.6f}") \
        .replace("IMG_LON1",     f"{data['lon1']:.6f}") \
        .replace("LEGEND_GRADIENT", legend_gradient)

    full_script = f"""
var TRAJECTORIES = {json.dumps(data['trajectories'], separators=(',', ':'))};
var ACS_DATA     = {json.dumps(data['acs_by_geoid'],  separators=(',', ':'))};
var IMAGES       = {images_js};
var GEOJSONS     = {geojsons_js};
var TRACTS_GEOJSON = {tracts_json};

{_JS_CHARTS}
{js_body}
"""

    m.get_root().html.add_child(folium.Element(_CSS))
    m.get_root().script.add_child(folium.Element(full_script))

    return m


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build interactive Hudson Yards Folium map."
    )
    parser.add_argument("--savename", default=DEFAULT_SAVENAME)
    args = parser.parse_args()

    results_dir = RESULTS_DIR / args.savename
    out_dir = results_dir / "evaluation" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading case-study data ===")
    data = load_data(results_dir, PROCESSED_DATA_DIR)

    print("\n=== Extracting zarr imagery ===")
    images_b64 = extract_images(data["cx"], data["cy"], data["half_ft"])

    print("\n=== Building GeoJSONs ===")
    geojsons = build_geojsons(data["bldg_box"], data["pred_by_yr"],
                              data["vmin"], data["vmax"])

    print("\n=== Building Folium map ===")
    m = build_map(data, images_b64, geojsons)

    out_path = out_dir / "D_hudson_yards_interactive.html"
    m.save(str(out_path))
    print(f"\n=== Done ===")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
