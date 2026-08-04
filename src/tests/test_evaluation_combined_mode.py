"""Tests for the combined US+NYC evaluation run (--mode both).

Covers the orchestration only — part internals are exercised by
test_evaluation_us_parts.py (US) and test_csa_event_study.py (the CSA engine).
The NYC parts are monkeypatched to record calls, so these tests stay fast and do
not need imagery, footprints or the csa package.
"""

import json

import numpy as np
import pandas as pd
import pytest

import src.evaluation as ev


# ─── part selection ───────────────────────────────────────────────────────────

def test_default_parts_per_mode():
    assert ev._split_parts(None, "us") == (list(ev._DEFAULT_US_PARTS), [])
    assert ev._split_parts(None, "nyc") == ([], list(ev._DEFAULT_NYC_PARTS))
    us, nyc = ev._split_parts(None, "both")
    # 'both' is an explicit request for the full set: the 3-panel main figure and
    # the CSA / Hudson Yards parts are included, unlike the us-only default.
    assert "main_figure" in us
    assert nyc == list(ev._NYC_PARTS) == ["A", "B", "C", "D", "E"]
    assert "main_figure" not in ev._DEFAULT_US_PARTS


def test_mixed_part_tokens_are_routed():
    us, nyc = ev._split_parts(["cross", "D", "main_figure", "e"], "both")
    assert us == ["cross", "main_figure"]
    assert nyc == ["D", "E"]          # lower-case NYC letters are accepted


def test_unknown_part_tokens_are_dropped(capsys):
    us, nyc = ev._split_parts(["cross", "banana", "Z"], "both")
    assert us == ["cross"] and nyc == []
    out = capsys.readouterr().out
    assert "banana" in out and "Z" in out


def test_resolve_mode_accepts_both(tmp_path):
    assert ev._resolve_mode(tmp_path, {"footprints_source": "ms_us"}, "both") == "both"


# ─── NYC directory resolution ─────────────────────────────────────────────────

def _touch_nyc(dirpath, years=(2010, 2012)):
    dirpath.mkdir(parents=True, exist_ok=True)
    for y in years:
        pd.DataFrame({"GEOID": ["36061000100"], "year": [y],
                      "predicted_value": [0.1], "Rel_Score": [0.2],
                      "building_id": [1], "type": ["test"]}
                     ).to_csv(dirpath / f"{y}_predictions.csv", index=False)
    return dirpath


def test_prefers_the_nyc_subdir_of_the_run(tmp_path):
    run = tmp_path / "run_x"
    _touch_nyc(run / ev.NYC_SUBDIR)
    assert ev._nyc_results_dir(run) == run / ev.NYC_SUBDIR


def test_falls_back_to_the_run_itself_for_a_legacy_nyc_run(tmp_path):
    run = tmp_path / "run_legacy"
    run.mkdir(parents=True)
    (run / "predictions_2016.parquet").write_bytes(b"")   # doitt_nyc-only artifact
    assert ev._nyc_results_dir(run) == run


def test_explicit_nyc_dir_wins(tmp_path):
    run = tmp_path / "run_y"
    _touch_nyc(run / ev.NYC_SUBDIR)
    other = _touch_nyc(tmp_path / "other")
    assert ev._nyc_results_dir(run, nyc_results_dir=other) == other


def test_returns_none_when_there_is_no_nyc_pass(tmp_path):
    run = tmp_path / "run_z"
    run.mkdir(parents=True)
    assert ev._nyc_results_dir(run) is None


# ─── year grid ────────────────────────────────────────────────────────────────

def test_nyc_year_grid_follows_the_directory(tmp_path, monkeypatch):
    """A NAIP-cadence NYC run (odd years) must not be evaluated on the biennial
    grid the module defaults to."""
    monkeypatch.setattr(ev, "YEARS", [2010, 2012, 2014, 2016])
    odd = _touch_nyc(tmp_path / "odd", years=(2017, 2019, 2021))
    assert ev._set_nyc_years(odd) == [2017, 2019, 2021]
    assert ev.YEARS == [2017, 2019, 2021]


def test_nyc_year_grid_left_alone_when_it_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "YEARS", [2010, 2012])
    d = _touch_nyc(tmp_path / "even", years=(2010, 2012))
    assert ev._set_nyc_years(d) == [2010, 2012]


def test_nyc_year_grid_reads_tract_parquets_when_csvs_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "YEARS", [2010])
    d = tmp_path / "tractonly"
    d.mkdir()
    for y in (2018, 2020):
        pd.DataFrame({"GEOID": ["36061000100"], "Rel_Score": [0.1],
                      "predicted_value": [0.2], "predicted_value_std": [0.0]}
                     ).to_parquet(d / f"predictions_by_tract_{y}.parquet")
    assert ev._set_nyc_years(d) == [2018, 2020]


# ─── dispatch + fault isolation ───────────────────────────────────────────────

@pytest.fixture
def stub_nyc_parts(monkeypatch):
    calls = []

    def rec(name, ret=None):
        def _f(*a, **kw):
            calls.append((name, kw))
            return ret
        return _f

    monkeypatch.setattr(ev, "part_a", rec("A"))
    monkeypatch.setattr(ev, "part_b", rec("B"))
    monkeypatch.setattr(ev, "part_c", rec("C", pd.DataFrame({"gb2_map": [1.0]})))
    monkeypatch.setattr(ev, "part_d", rec("D", {"csa/nyc/pretrend_p": 0.42}))
    monkeypatch.setattr(ev, "part_e", rec("E"))
    return calls


def test_nyc_parts_run_in_order_and_c_feeds_e(tmp_path, stub_nyc_parts):
    d = _touch_nyc(tmp_path / "nyc")
    summary = ev._run_nyc_parts(d, tmp_path, tmp_path / "out", ["E", "C", "A"])
    names = [c[0] for c in stub_nyc_parts]
    assert names == ["A", "C", "E"], "C must precede E so the GB2 map can flow"
    e_kwargs = dict(stub_nyc_parts[-1][1])
    assert e_kwargs["qmap_long"] is not None
    assert summary["years"] == [2010, 2012]


def test_part_d_headline_flows_into_the_summary(tmp_path, stub_nyc_parts):
    d = _touch_nyc(tmp_path / "nyc")
    summary = ev._run_nyc_parts(d, tmp_path, tmp_path / "out", ["D"])
    assert summary["csa/nyc/pretrend_p"] == 0.42


def test_one_failing_nyc_part_does_not_stop_the_others(tmp_path, monkeypatch,
                                                       stub_nyc_parts):
    monkeypatch.setattr(
        ev, "part_c", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    d = _touch_nyc(tmp_path / "nyc")
    summary = ev._run_nyc_parts(d, tmp_path, tmp_path / "out", ["A", "C", "D", "E"])
    names = [c[0] for c in stub_nyc_parts]
    assert names == ["A", "D", "E"]          # C blew up, the rest still ran
    assert summary["part_C/status"] == "failed"
    # E must still run, falling back to raw predictions
    assert dict(stub_nyc_parts[-1][1])["qmap_long"] is None


def test_both_mode_runs_the_nyc_half_from_the_subdir(tmp_path, monkeypatch,
                                                     stub_nyc_parts):
    """The whole point of --mode both: one call, two output trees."""
    run = tmp_path / "run_both"
    _touch_nyc(run / ev.NYC_SUBDIR)
    # no US predictions in `run` itself -> the US half is skipped, not fatal
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path)

    summary = ev.run_evaluation(
        "run_both", params={"footprints_source": "ms_us"},
        results_dir=run, processed_dir=tmp_path, mode="both",
    )
    assert summary["us/status"] == "no_predictions"
    assert [c[0] for c in stub_nyc_parts] == ["A", "B", "C", "D", "E"]
    assert summary["nyc/results_dir"] == str(run / ev.NYC_SUBDIR)
    # NYC results land in their own evaluation tree, not the US one
    assert (run / ev.NYC_SUBDIR / "evaluation" / "figures").is_dir()
    saved = json.loads((run / "evaluation" / "US_summary.json").read_text())
    assert saved["mode"] == "both"


def test_both_mode_reports_a_missing_nyc_pass_without_failing(tmp_path, capsys,
                                                              stub_nyc_parts):
    run = tmp_path / "run_no_nyc"
    run.mkdir(parents=True)
    summary = ev.run_evaluation(
        "run_no_nyc", params={"footprints_source": "ms_us"},
        results_dir=run, processed_dir=tmp_path, mode="both",
    )
    assert summary == {}                     # nothing ran at all
    assert not stub_nyc_parts
    assert "NYC parts skipped" in capsys.readouterr().out


# ─── usetex safety ────────────────────────────────────────────────────────────
# paper.mplstyle sets text.usetex, where a bare '%' opens a LaTeX comment and
# swallows the rest of the string: "5% of baseline area" renders as "5". This was
# a real (silent) figure-title bug, so the guards get tests.

def test_tex_pct_escapes_the_percent():
    assert ev._tex_pct(0.05) == r"5\%"
    assert ev._tex_pct(0.10) == r"10\%"
    assert ev._tex_pct(0.01) == r"1\%"
    assert "%" not in ev._tex_pct(0.05).replace(r"\%", "")


def test_tex_escape_neutralises_latex_specials():
    out = ev._tex_escape("only 19% of _units_ & $x$ #1 {a}")
    for raw, esc in (("%", r"\%"), ("_", r"\_"), ("&", r"\&"),
                     ("$", r"\$"), ("#", r"\#")):
        assert esc in out
    # no unescaped special is left behind
    assert "%" not in out.replace(r"\%", "")


def test_city_labels_are_ascii_for_usetex():
    """City labels flow straight into figure titles; a raw en-dash breaks usetex."""
    from src import csa_event_study as ces
    for spec in ces.CSA_CITIES.values():
        assert spec.label.isascii(), f"{spec.key}: {spec.label!r} is not ASCII"


# ─── part_a must stay out-of-sample now that NYC predicts the whole city ─────

def _synth_nyc_processed(tmp_path, n_train=400, n_heldout=300):
    """A tract_splits.feather with both train and held-out NYC tracts."""
    import geopandas as gpd
    from shapely.geometry import box

    rows, geoms = [], []
    for i in range(n_train + n_heldout):
        rows.append({"GEOID": f"36061{i:06d}",
                     "type": "train" if i < n_train else "test"})
        geoms.append(box(i, 0, i + 1, 1))
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=5070)
    gdf.to_feather(tmp_path / "tract_splits.feather")
    return gdf


def test_part_a_headline_is_heldout_not_pooled(tmp_path, capsys):
    """REGRESSION: the NYC pass is now unsampled, so the tract parquet contains
    train tracts too. part_a used to pool every row, which would silently turn
    the paper's headline cross-section into a partly in-sample number.
    """
    gdf = _synth_nyc_processed(tmp_path)
    rng = np.random.default_rng(0)
    t = gdf["type"].to_numpy()
    label = rng.normal(0, 1, len(gdf))
    # train tracts fit much better than held-out ones
    pred = label + rng.normal(0, np.where(t == "train", 0.15, 1.1))

    rd = tmp_path / "run"
    rd.mkdir()
    out = tmp_path / "eval"
    ev._make_dirs(out)
    pd.DataFrame({"GEOID": gdf["GEOID"], "Rel_Score": label,
                  "predicted_value": pred, "predicted_value_std": 0.1}
                 ).to_parquet(rd / "predictions_by_tract_2016.parquet")

    ev.part_a(rd, tmp_path, out)

    tab = pd.read_csv(out / "tables" / "A_cross_sectional_2016.csv")
    head = tab[tab["headline"] & (tab["metric"] == "Spearman_rho")]
    assert len(head) == 1
    assert head["split"].iloc[0] == "heldout"
    assert head["n"].iloc[0] == 300, "headline must use held-out tracts only"

    by_split = tab[tab["metric"] == "Spearman_rho"].set_index("split")["value"]
    # train fits better by construction; the headline must NOT be the pooled value
    assert by_split["train"] > by_split["heldout"]
    assert "all" in by_split.index                       # reported, but not headline
    assert by_split["heldout"] < by_split["all"]
    assert "HEADLINE" in capsys.readouterr().out


def test_part_a_says_so_when_no_holdout_can_be_resolved(tmp_path, capsys):
    """With no split file, part_a still runs but must not claim to be holdout."""
    rd = tmp_path / "run"
    rd.mkdir()
    out = tmp_path / "eval"
    ev._make_dirs(out)
    rng = np.random.default_rng(1)
    label = rng.normal(0, 1, 200)
    pd.DataFrame({"GEOID": [f"36061{i:06d}" for i in range(200)],
                  "Rel_Score": label, "predicted_value": label + rng.normal(0, .5, 200),
                  "predicted_value_std": 0.1}
                 ).to_parquet(rd / "predictions_by_tract_2016.parquet")

    ev.part_a(rd, tmp_path, out)   # tmp_path has no tract_splits.feather

    printed = capsys.readouterr().out
    assert "NOT out-of-sample" in printed
    tab = pd.read_csv(out / "tables" / "A_cross_sectional_2016.csv")
    assert tab[tab["headline"]]["split"].iloc[0] == "all"


def test_panel_b_weights_by_tracts_not_buildings():
    """Labels are tract-level, so a city sampled at more buildings per tract must
    not gain weight in the main figure's Panel B. This matched
    metrics.weighted_within_spearman only after a fix — pinning it here.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # City A: few tracts but a huge building count (full-universe style) and a
    # very different rho. City B: many tracts, capped buildings.
    cells = pd.DataFrame([
        {"cbsa": "A", "year": 2020, "n": 100_000, "n_tracts": 10, "rho": 0.10},
        {"cbsa": "B", "year": 2020, "n": 1_000, "n_tracts": 90, "rho": 0.90},
    ])
    fig, ax = plt.subplots()
    ev._panel_b_kill_shot(ax, cells)
    drawn = [l for l in ax.get_lines() if l.get_label().startswith("Ordinal")]
    assert drawn, "the real ordinal curve must be drawn"
    y = float(drawn[0].get_ydata()[0])
    plt.close(fig)

    tract_weighted = (0.10 * 10 + 0.90 * 90) / 100          # = 0.82
    bldg_weighted = (0.10 * 100_000 + 0.90 * 1_000) / 101_000  # = 0.108
    assert y == pytest.approx(tract_weighted, abs=1e-9)
    assert abs(y - bldg_weighted) > 0.5, "building-weighting was used"


def test_within_city_val_types_count_as_heldout():
    """val_within_nyc / val_within_chicago are spatial holdouts; if they were
    missing from the group map those tracts would vanish from both groups."""
    heldout = set(ev.CSA_SPLIT_GROUPS["heldout"])
    assert {"val_within_nyc", "val_within_chicago", "test", "dead_zone"} <= heldout
    assert "train" not in heldout


def test_us_mode_does_not_touch_the_nyc_half(tmp_path, stub_nyc_parts):
    run = tmp_path / "run_us"
    _touch_nyc(run / ev.NYC_SUBDIR)
    ev.run_evaluation(
        "run_us", params={"footprints_source": "ms_us"},
        results_dir=run, processed_dir=tmp_path, mode="us",
    )
    assert not stub_nyc_parts
