"""Descriptive figures & tables for the train/val/test split (paper annex).

Produces, from the built whole-city split + tract overrides:

* **Maps** of the within-city split for NYC and Chicago (train / val / test /
  dead-zone tracts), on a clean Science-style template — single-panel per city
  plus a combined two-panel figure.
* **Tables**: (a) every CBSA with its bracket, split, tract count and shares;
  (b) per-bracket × split totals; (c) an overall totals row. Written as both CSV
  (for inspection) and LaTeX booktabs (drop straight into the annex).

Colors are the validated data-viz categorical slots (blue/orange/aqua, all-pairs
CVD-safe) plus a recessive neutral for the dropped dead-zone. Identity is never
color-alone: every map carries a labelled legend and thin polygon edges.

Everything here is reporting only — it never changes an assignment. Call
:func:`report_splits` at the end of the split build; it is wrapped so a plotting
failure cannot abort a training run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import cbsa_brackets
from src.data import us_split

# Validated categorical slots (see dataviz skill: blue/orange/aqua pass all-pairs
# CVD ΔE 9.2, normal-vision 24.0). Dead-zone is a recessive neutral, not a hue.
SPLIT_MAP_COLORS = {
    "train": "#2a78d6",       # slot 1 blue
    "test": "#eb6834",        # slot 2 orange
    "val": "#1baf7a",         # slot 3 aqua  (tract_splits folds val_within_* -> val)
    "dead_zone": "#d9d8d2",   # recessive neutral (dropped from every split)
    "unassigned": "#f2f1ec",
}
_SPLIT_ORDER = ["train", "val", "test", "dead_zone", "unassigned"]
_SPLIT_LABEL = {
    "train": "Train", "val": "Validation", "test": "Test",
    "dead_zone": "Dead zone (dropped)", "unassigned": "Unassigned",
}

# Map cities: cbsa_code -> display name.
MAP_CITIES = {
    us_split.NYC_CBSA: "New York City",
    "16980": "Chicago",
}

# Ink / chrome (dataviz reference: light chart chrome).
_INK = "#0b0b0b"
_MUTED = "#898781"
_EDGE = "#fcfcfb"          # thin surface-colored gap between tracts


def _style_map_ax(ax, title):
    import matplotlib.pyplot as plt  # noqa: F401
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, color=_INK, pad=8,
                 fontfamily="sans-serif", fontweight="semibold"
                 if "semibold" in _mpl_weights() else "bold")


def _mpl_weights():
    # Guard against backends lacking 'semibold'.
    try:
        from matplotlib import font_manager  # noqa: F401
        return {"semibold"}
    except Exception:
        return set()


def _legend_handles(present_types, counts):
    import matplotlib.patches as mpatches
    handles = []
    for t in _SPLIT_ORDER:
        if t not in present_types:
            continue
        n = counts.get(t, 0)
        handles.append(mpatches.Patch(
            facecolor=SPLIT_MAP_COLORS[t], edgecolor="none",
            label=f"{_SPLIT_LABEL[t]}  ({n:,} tracts)"))
    return handles


def plot_city_split_map(city_gdf, city_title, out_path, type_col="type", ax=None):
    """Single-city split map (train/val/test/dead-zone tracts). Saves if ``ax`` is None.

    ``city_gdf``: GeoDataFrame of one city's tracts with a ``type`` column and
    geometry. Returns the Axes.
    """
    import matplotlib.pyplot as plt

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=200)

    counts = city_gdf[type_col].value_counts().to_dict()
    present = [t for t in _SPLIT_ORDER if t in counts]
    for t in present:
        sub = city_gdf[city_gdf[type_col] == t]
        sub.plot(ax=ax, color=SPLIT_MAP_COLORS[t], edgecolor=_EDGE, linewidth=0.15)

    _style_map_ax(ax, city_title)
    ax.legend(handles=_legend_handles(present, counts), loc="lower left",
              frameon=False, fontsize=9, labelcolor=_INK,
              title="Split", title_fontsize=9)

    if own_fig:
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        fig.savefig(str(Path(out_path).with_suffix(".pdf")), bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"Created figure: {out_path}")
    return ax


def plot_split_maps(tract_gdf, out_dir, cities=MAP_CITIES, type_col="type",
                    cbsa_col="cbsa_code"):
    """Per-city maps + a combined multi-panel figure, for every city in ``cities``.

    ``tract_gdf``: GeoDataFrame with ``cbsa_code``, ``type`` and geometry (any
    CRS; drawn as-is). Cities absent from the frame are skipped.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = tract_gdf[cbsa_col].astype(str)

    present_cities = [(c, name) for c, name in cities.items() if (codes == c).any()]
    if not present_cities:
        print("⚠️ split maps: none of the map cities are present in the frame.")
        return

    for code, name in present_cities:
        city = tract_gdf[codes == code]
        plot_city_split_map(city, f"{name} — train / validation / test split",
                            out_dir / f"split_map_{code}.png", type_col=type_col)

    # Combined figure (one row).
    n = len(present_cities)
    fig, axes = plt.subplots(1, n, figsize=(7.2 * n, 7.2), dpi=200)
    axes = np.atleast_1d(axes)
    for ax, (code, name) in zip(axes, present_cities):
        city = tract_gdf[codes == code]
        plot_city_split_map(city, name, None, type_col=type_col, ax=ax)
    fig.suptitle("Within-city train / validation / test split",
                 fontsize=15, color=_INK, fontweight="bold")
    fig.tight_layout()
    combo = out_dir / "split_maps_nyc_chicago.png"
    fig.savefig(combo, bbox_inches="tight", facecolor="white")
    fig.savefig(str(combo.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Created figure: {combo}")


def per_cbsa_table(city_split_df: pd.DataFrame,
                   nyc_row: dict | None = None) -> pd.DataFrame:
    """Every CBSA with bracket, split, tract count and shares (of bracket & total).

    ``nyc_row``: optional dict describing the within-city NYC entry (it is absent
    from ``city_split_df`` because NYC has no whole-city split) — appended as a
    ``split="within_city"`` row so the annex lists NYC too.
    """
    df = city_split_df.copy()
    df["cbsa_code"] = df["cbsa_code"].astype(str)
    if nyc_row is not None:
        df = pd.concat([df, pd.DataFrame([nyc_row])], ignore_index=True)

    total = df["n_tracts"].sum()
    bracket_tot = df.groupby("bracket")["n_tracts"].transform("sum")
    df["share_of_bracket"] = (df["n_tracts"] / bracket_tot).round(4)
    df["share_of_total"] = (df["n_tracts"] / total).round(4)
    df["is_forced_test"] = df["cbsa_code"].isin(us_split.FORCED_TEST_CBSAS)

    bracket_order = {b: i for i, b in enumerate(cbsa_brackets.BRACKETS)}
    df["_b"] = df["bracket"].map(bracket_order).fillna(99)
    cols = ["cbsa_code", "cbsa_title", "bracket", "split", "n_tracts",
            "share_of_bracket", "share_of_total", "is_forced_test"]
    if "holdout_year" in df.columns:
        cols.append("holdout_year")
    return df.sort_values(["_b", "n_tracts"], ascending=[True, False])[cols].reset_index(drop=True)


def totals_table(per_cbsa: pd.DataFrame) -> pd.DataFrame:
    """Per-bracket × split city/tract counts and tract shares, plus an ALL row."""
    g = (per_cbsa.groupby(["bracket", "split"])
         .agg(n_cities=("cbsa_code", "size"), n_tracts=("n_tracts", "sum"))
         .reset_index())
    tot = per_cbsa["n_tracts"].sum()
    g["tract_share"] = (g["n_tracts"] / tot).round(4)

    overall = (per_cbsa.groupby("split")
               .agg(n_cities=("cbsa_code", "size"), n_tracts=("n_tracts", "sum"))
               .reset_index())
    overall.insert(0, "bracket", "ALL")
    overall["tract_share"] = (overall["n_tracts"] / tot).round(4)
    return pd.concat([g, overall], ignore_index=True)


def _write_table(df: pd.DataFrame, out_dir: Path, stem: str, caption: str, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    try:
        latex = df.to_latex(index=False, escape=True, longtable=(len(df) > 40),
                            caption=caption, label=label,
                            column_format="l" * df.shape[1])
        (out_dir / f"{stem}.tex").write_text(latex)
    except Exception as exc:                          # pragma: no cover
        print(f"⚠️ LaTeX export for {stem} failed ({exc!r}); CSV still written.")
    print(f"Created table: {out_dir / stem}.csv (+ .tex)")


def write_split_tables(city_split_df, out_dir, nyc_row=None):
    """Write per-CBSA and totals tables (CSV + LaTeX) to ``out_dir``."""
    out_dir = Path(out_dir)
    per_cbsa = per_cbsa_table(city_split_df, nyc_row=nyc_row)
    totals = totals_table(per_cbsa)
    _write_table(per_cbsa, out_dir, "split_by_cbsa",
                 "Train/validation/test assignment by CBSA, with population "
                 "bracket and tract shares.", "tab:split_by_cbsa")
    _write_table(totals, out_dir, "split_totals",
                 "Train/validation/test totals by population bracket "
                 "(tract counts and shares).", "tab:split_totals")
    return per_cbsa, totals


def report_splits(tract_gdf, city_split_df, fig_dir, tab_dir, nyc_row=None):
    """Produce every split descriptive (maps + tables). Never raises.

    ``tract_gdf``: tract GeoDataFrame with cbsa_code/type/geometry (the same frame
    persisted to tract_splits.feather). ``city_split_df``: the whole-city split.
    """
    try:
        plot_split_maps(tract_gdf, fig_dir)
    except Exception as exc:                          # pragma: no cover
        print(f"⚠️ split maps skipped ({exc!r}).")
    try:
        write_split_tables(city_split_df, tab_dir, nyc_row=nyc_row)
    except Exception as exc:                          # pragma: no cover
        print(f"⚠️ split tables skipped ({exc!r}).")
