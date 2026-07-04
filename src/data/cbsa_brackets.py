"""CBSA population brackets and the whole-city (CBSA) train/val/test split (#28).

Khachiyan et al. (2022, AER: Insights)-style assignment: whole urban areas
(CBSAs) are randomized into train/val/test so no city ever straddles splits,
stratified by population bracket so every size class is represented in each
split. Targets ~50/20/30 (train/val/test) measured in TRACT COUNT — the
cross-sectional data share — with cities as atoms (greedy deterministic
balancing within brackets). The mega bracket (top-4 CBSAs by population,
expected NY/LA/Chicago/Houston) is fixed at 2 train / 1 val / 1 test via a
seeded permutation.

Each TRAIN city additionally gets one temporal-holdout year: the imagery year
closest to the middle of its state's NAIP coverage window (from the
naip_coverage_audit.py CSV when available, else the middle of the panel
years), so the holdout is always an interpolation test.

Shared by build_dataset.py (split construction; persists cbsa_splits.feather)
and main.py (#31 per-bracket validation metrics). Population per CBSA comes
from process_acs.get_large_metros() — the same aggregation that enforces
MIN_METRO_POP — with a panel-derived fallback (summing total_population per
CBSA) when the raw ACS store is unreachable (tests / sandboxed sessions).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR

SPLIT_SEED = 825
BRACKET_TARGETS = {"train": 0.50, "val": 0.20, "test": 0.30}
# Tie-break priority when two splits have the same tract deficit.
_SPLIT_PRIORITY = ("train", "test", "val")
N_MEGA = 4
MEGA_MIN_UNIVERSE = 8          # mega bracket only forms when the universe has >= 8 CBSAs
LARGE_MIN_POP = 1_000_000
MEDIUM_MIN_POP = 750_000
BRACKETS = ("mega", "large", "medium", "small")
CITY_SPLIT_PATH = PROCESSED_DATA_DIR / "cbsa_splits.feather"
DEFAULT_COVERAGE_CSV = Path.home() / "outputs" / "naip_coverage.csv"

# State FIPS prefix -> MS-footprints / naip_coverage_audit state stem.
FIPS_TO_STATE = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "DistrictofColumbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
    "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
    "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
    "32": "Nevada", "33": "NewHampshire", "34": "NewJersey", "35": "NewMexico",
    "36": "NewYork", "37": "NorthCarolina", "38": "NorthDakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "RhodeIsland",
    "45": "SouthCarolina", "46": "SouthDakota", "47": "Tennessee", "48": "Texas",
    "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "WestVirginia", "55": "Wisconsin", "56": "Wyoming",
}


def cbsa_populations(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """CBSA populations ``[cbsa_code, cbsa_title, population]``.

    Primary source: :func:`process_acs.get_large_metros` (never recomputed).
    Fallback (raw ACS store unreachable): sum ``total_population_{BASE_YEAR}``
    per ``cbsa_code`` from the supplied tract ``panel`` — the same carried
    column, just restricted to the panel universe.
    """
    from src.data import process_acs

    try:
        pop = process_acs.get_large_metros()
    except Exception as exc:
        pop_col = f"total_population_{process_acs.BASE_YEAR}"
        if panel is None or pop_col not in panel.columns or "cbsa_code" not in panel.columns:
            raise
        warnings.warn(
            f"get_large_metros() unavailable ({exc!r}); falling back to summing "
            f"{pop_col} per CBSA from the tract panel."
        )
        pop = (
            panel.groupby("cbsa_code", as_index=False)[pop_col]
            .sum()
            .rename(columns={pop_col: "population"})
        )
        pop["cbsa_title"] = pop["cbsa_code"].astype(str)
    pop = pop.copy()
    pop["cbsa_code"] = pop["cbsa_code"].astype(str)
    pop = pop.sort_values("population", ascending=False).reset_index(drop=True)
    return pop[["cbsa_code", "cbsa_title", "population"]]


def assign_brackets(pop_df: pd.DataFrame, n_mega: int = N_MEGA) -> pd.DataFrame:
    """Add a ``bracket`` column: mega (top ``n_mega`` by population, only when
    the universe has >= MEGA_MIN_UNIVERSE CBSAs), large (>1M),
    medium (750K-1M), small (the 500K-750K remainder)."""
    df = pop_df.sort_values(
        ["population", "cbsa_code"], ascending=[False, True]
    ).reset_index(drop=True)
    df["bracket"] = np.where(
        df["population"] > LARGE_MIN_POP, "large",
        np.where(df["population"] >= MEDIUM_MIN_POP, "medium", "small"),
    )
    if len(df) >= MEGA_MIN_UNIVERSE:
        df.loc[df.index[:n_mega], "bracket"] = "mega"
    return df


def build_city_split(tract_counts: pd.Series, pop_df: pd.DataFrame,
                     seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Assign whole cities to train/val/test, stratified by population bracket.

    Args:
        tract_counts: Series indexed by cbsa_code -> number of unique tracts
            (the split universe: CBSAs actually present in the buildings data).
        pop_df: output of :func:`cbsa_populations`.
        seed: RNG seed (mega permutation only; the rest is fully deterministic).

    Returns ``[cbsa_code, cbsa_title, population, n_tracts, bracket, split]``.
    Deterministic given seed. Raises when the universe has < 3 CBSAs (a
    whole-city split needs at least one city per set).
    """
    counts = tract_counts.copy()
    counts.index = counts.index.astype(str)
    universe = list(counts.index)
    if len(universe) < 3:
        raise ValueError(
            f"Whole-city split needs >= 3 CBSAs; got {len(universe)}. "
            "Single/dual-CBSA runs cannot hold out entire cities — pass more "
            "states or widen the buildings universe."
        )

    df = pop_df[pop_df["cbsa_code"].isin(universe)].copy()
    missing = sorted(set(universe) - set(df["cbsa_code"]))
    if missing:
        warnings.warn(
            f"{len(missing)} CBSAs in the tract universe lack a population row "
            f"({missing[:5]}...); treating them as 'small'."
        )
        df = pd.concat([
            df,
            pd.DataFrame({"cbsa_code": missing,
                          "cbsa_title": missing,
                          "population": 0}),
        ], ignore_index=True)
    df = assign_brackets(df)
    df["n_tracts"] = df["cbsa_code"].map(counts).astype(int)

    rng = np.random.default_rng(seed)
    split_of: dict[str, str] = {}

    for bracket in BRACKETS:
        cities = df[df["bracket"] == bracket]
        if cities.empty:
            continue
        if bracket == "mega":
            # Fixed 2 train / 1 val / 1 test, seeded permutation.
            codes = list(cities.sort_values(
                ["population", "cbsa_code"], ascending=[False, True])["cbsa_code"])
            perm = [codes[i] for i in rng.permutation(len(codes))]
            assignment = ["train", "train", "val", "test"][: len(perm)]
            split_of.update(dict(zip(perm, assignment)))
            continue

        cities = cities.sort_values(
            ["n_tracts", "cbsa_code"], ascending=[False, True])
        if len(cities) == 1:
            split_of[cities["cbsa_code"].iloc[0]] = "train"
            continue
        if len(cities) == 2:
            split_of[cities["cbsa_code"].iloc[0]] = "train"
            split_of[cities["cbsa_code"].iloc[1]] = "test"
            continue

        # Greedy: assign each city (largest first) to the split with the
        # largest remaining tract deficit vs the 50/20/30 targets.
        total = cities["n_tracts"].sum()
        targets = {s: BRACKET_TARGETS[s] * total for s in _SPLIT_PRIORITY}
        assigned = {s: 0 for s in _SPLIT_PRIORITY}
        for _, city in cities.iterrows():
            deficits = {s: targets[s] - assigned[s] for s in _SPLIT_PRIORITY}
            best = max(_SPLIT_PRIORITY, key=lambda s: (deficits[s], -_SPLIT_PRIORITY.index(s)))
            split_of[city["cbsa_code"]] = best
            assigned[best] += city["n_tracts"]

    df["split"] = df["cbsa_code"].map(split_of)

    # Global post-check: every split must own at least one city. When a split
    # is empty (tiny universes), move the smallest train city into it.
    for s in ("val", "test"):
        if (df["split"] == s).any():
            continue
        train_pool = df[df["split"] == "train"].sort_values(
            ["n_tracts", "cbsa_code"], ascending=[True, True])
        if len(train_pool) <= 1:
            raise ValueError(
                f"Cannot populate the '{s}' split without emptying train "
                f"({len(train_pool)} train cities). Widen the CBSA universe."
            )
        df.loc[df["cbsa_code"] == train_pool["cbsa_code"].iloc[0], "split"] = s

    _print_split_summary(df)
    return df[["cbsa_code", "cbsa_title", "population", "n_tracts",
               "bracket", "split"]].reset_index(drop=True)


def _print_split_summary(df: pd.DataFrame) -> None:
    print("\n--- Whole-city (CBSA) split ---")
    summary = (
        df.groupby(["bracket", "split"])
        .agg(cities=("cbsa_code", "size"), tracts=("n_tracts", "sum"))
        .reset_index()
    )
    print(summary.to_string(index=False))
    shares = df.groupby("split")["n_tracts"].sum()
    shares = (shares / shares.sum()).round(3)
    print(f"Tract shares: {shares.to_dict()} (targets {BRACKET_TARGETS})")
    print("-" * 31)


def load_naip_coverage(csv_path=None) -> pd.DataFrame | None:
    """Load the naip_coverage_audit.py CSV (state, year, n_items, ...) if present."""
    csv_path = Path(csv_path) if csv_path is not None else DEFAULT_COVERAGE_CSV
    if not csv_path.exists():
        return None
    cov = pd.read_csv(csv_path)
    if not {"state", "year", "n_items"}.issubset(cov.columns):
        warnings.warn(f"{csv_path} lacks state/year/n_items columns; ignoring it.")
        return None
    return cov


def pick_holdout_year(state_stem: str | None, years, coverage_df=None) -> int:
    """Imagery year closest to the middle of the state's NAIP coverage window.

    Falls back to the middle of ``years`` when the state has no coverage rows
    (or no coverage CSV was supplied). Ties break toward the earlier year.
    """
    years = sorted(int(y) for y in years)
    mid = (years[0] + years[-1]) / 2.0
    if coverage_df is not None and state_stem is not None:
        cov = coverage_df[(coverage_df["state"] == state_stem)
                          & (coverage_df["n_items"] > 0)]
        if len(cov):
            mid = (cov["year"].min() + cov["year"].max()) / 2.0
    return min(years, key=lambda y: (abs(y - mid), y))


def attach_holdout_years(city_split_df: pd.DataFrame, cbsa_state_map: dict,
                         years, coverage_df=None) -> pd.DataFrame:
    """Add ``holdout_year`` (NaN for non-train cities) to the city split table.

    ``cbsa_state_map``: cbsa_code -> state stem (modal state of the city's
    tracts, via FIPS_TO_STATE).
    """
    df = city_split_df.copy()
    df["holdout_year"] = [
        float(pick_holdout_year(cbsa_state_map.get(code), years, coverage_df))
        if split == "train" else np.nan
        for code, split in zip(df["cbsa_code"], df["split"])
    ]
    return df


def holdout_year_map(city_split_df: pd.DataFrame) -> dict[int, int]:
    """{cbsa_code (int) -> holdout year (int)} for train cities — the shard
    cache managers use int CBSA keys (see main.py ``_cbsa_int``)."""
    out = {}
    for code, year in zip(city_split_df["cbsa_code"], city_split_df.get("holdout_year", [])):
        if pd.notna(year):
            try:
                out[int(code)] = int(year)
            except (TypeError, ValueError):
                continue
    return out


def save_city_split(city_split_df: pd.DataFrame, path=CITY_SPLIT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    city_split_df.reset_index(drop=True).to_feather(path)
    print(f"Created file: {path}")


def load_city_split(path=CITY_SPLIT_PATH) -> pd.DataFrame:
    return pd.read_feather(Path(path))


def top_cbsas(city_split_df: pd.DataFrame, n: int = 10) -> list:
    """The n most populous cbsa_codes (for the #31 per-city metrics)."""
    return list(
        city_split_df.sort_values("population", ascending=False)["cbsa_code"].head(n)
    )
