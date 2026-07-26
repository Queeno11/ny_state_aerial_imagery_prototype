"""Unit tests (synthetic data) for the pure logic in eval_small_city_offline."""
import numpy as np
import pandas as pd
import pytest

from src.eval_small_city_offline import (
    build_eval_frame,
    compute_report,
    dedupe_effective_rows,
    label_for_effective_year,
)


def _synthetic_buildings(n_tracts=6, per_tract=4, cbsa="11111"):
    rows = []
    for t in range(n_tracts):
        for b in range(per_tract):
            rows.append({
                "building_id": t * 100 + b,
                "GEOID": f"{cbsa}{t:06d}",
                "cbsa_code": cbsa,
                "centroid_x": 1000.0 * t,
                "centroid_y": 2000.0 * t,
            })
    return pd.DataFrame(rows)


class TestBuildEvalFrame:
    def test_one_building_per_tract_all_years(self):
        bld = _synthetic_buildings(n_tracts=6, per_tract=4)
        frame = build_eval_frame(bld, {11111: [2018, 2020]})
        assert len(frame) == 6 * 2
        assert frame.groupby("GEOID")["building_id"].nunique().eq(1).all()
        assert set(frame["year"]) == {2018, 2020}
        # every tract appears with both years
        assert frame.groupby("GEOID")["year"].count().eq(2).all()

    def test_deterministic_under_seed(self):
        bld = _synthetic_buildings()
        f1 = build_eval_frame(bld, {11111: [2020]}, seed=7)
        f2 = build_eval_frame(bld, {11111: [2020]}, seed=7)
        pd.testing.assert_frame_equal(f1, f2)

    def test_tract_subsample(self):
        bld = _synthetic_buildings(n_tracts=10)
        frame = build_eval_frame(bld, {11111: [2020]}, n_tracts_per_cbsa=4)
        assert frame["GEOID"].nunique() == 4

    def test_multiple_cbsas_and_missing_cbsa(self):
        bld = pd.concat([_synthetic_buildings(cbsa="11111"),
                         _synthetic_buildings(cbsa="22222")], ignore_index=True)
        frame = build_eval_frame(bld, {11111: [2020], 22222: [2018], 99999: [2020]})
        assert set(frame["cbsa"]) == {11111, 22222}
        assert (frame.loc[frame["cbsa"] == 22222, "year"] == 2018).all()

    def test_empty_input(self):
        frame = build_eval_frame(_synthetic_buildings().iloc[:0], {11111: [2020]})
        assert frame.empty


class TestLabelForEffectiveYear:
    lookup = {("g1", 2018): 0.5, ("g1", 2020): 0.7, ("g1", 2022): np.nan}
    panel = [2018, 2020, 2022]

    def test_exact_hit(self):
        assert label_for_effective_year("g1", 2020, self.lookup, self.panel) == (0.7, 2020)

    def test_nearest_fallback(self):
        # 2019 not in panel -> nearest is 2018 (min with abs-distance tie rule)
        assert label_for_effective_year("g1", 2019, self.lookup, self.panel) == (0.5, 2018)

    def test_nan_exact_falls_back_to_nearest(self):
        # (g1, 2022) is NaN -> nearest OTHER panel year is 2022 itself... the
        # fallback picks 2022 again and returns None: substituted imagery never
        # silently inherits a farther year's label.
        label, year = label_for_effective_year("g1", 2022, self.lookup, self.panel)
        assert label is None and year is None

    def test_unknown_geoid(self):
        assert label_for_effective_year("nope", 2020, self.lookup, self.panel) == (None, None)


class TestDedupeEffectiveRows:
    def test_substitution_collision_dropped(self):
        df = pd.DataFrame({
            "building_id": [1, 1, 2],
            "year": [2019, 2020, 2020],
            "eff_year": [2020, 2020, 2020],
        })
        out = dedupe_effective_rows(df)
        assert len(out) == 2
        assert out.groupby(["building_id", "eff_year"]).size().eq(1).all()


class TestComputeReport:
    def _frame(self, rho_sign=1.0, n_tracts=30, years=(2018, 2020), cbsa=11111):
        rng = np.random.default_rng(0)
        rows = []
        for year in years:
            labels = rng.normal(size=n_tracts)
            for i, lab in enumerate(labels):
                rows.append({"cbsa": cbsa, "year": year, "building_id": i,
                             "label": lab, "pred": rho_sign * lab})
        return pd.DataFrame(rows)

    def test_perfect_ranking(self):
        rep = compute_report(self._frame(rho_sign=1.0))
        assert rep["bracket"]["within_spearman"] == pytest.approx(1.0)
        assert rep["bracket"]["within_cells"] == 2
        assert rep["per_city"][11111]["within_spearman"] == pytest.approx(1.0)

    def test_anti_ranking(self):
        rep = compute_report(self._frame(rho_sign=-1.0))
        assert rep["bracket"]["within_spearman"] == pytest.approx(-1.0)

    def test_tract_weighting_pools_two_cities(self):
        a = self._frame(rho_sign=1.0, n_tracts=30, cbsa=1)
        b = self._frame(rho_sign=-1.0, n_tracts=10, cbsa=2)
        rep = compute_report(pd.concat([a, b], ignore_index=True))
        # (30*1 + 10*(-1)) / 40 per year -> +0.5 overall
        assert rep["bracket"]["within_spearman"] == pytest.approx(0.5)
        assert set(rep["per_city"]) == {1, 2}
