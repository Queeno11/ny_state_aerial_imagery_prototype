"""Unit tests for src.data.nyc_building_heights (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from src.data import nyc_building_heights as nbh


def test_clean_heights_drops_bad_ids_and_marks_bad_heights_missing():
    raw = pd.DataFrame({"DOITT_ID": [1, 2, 3, 0, None, 1, 4],
                        "HEIGHT_ROOF": [30.0, 0.0, -5.0, 10.0, 12.0, 45.0, 5000.0]})
    c = nbh.clean_heights(raw).set_index("DOITT_ID")["height_roof"]
    assert list(c.index) == [1, 2, 3, 4]            # id 0 and missing id dropped
    assert c[1] == 45.0                              # duplicate keeps the larger
    assert np.isnan(c[2]) and np.isnan(c[3]) and np.isnan(c[4])   # 0, negative, implausible


def test_combine_layers_prefers_current_then_historic():
    cur = pd.DataFrame({"DOITT_ID": [1, 2], "height_roof": [10.0, np.nan]})
    hist = pd.DataFrame({"DOITT_ID": [2, 3], "height_roof": [20.0, 30.0]})
    out = nbh.combine_layers(cur, hist).set_index("DOITT_ID")["height_roof"]
    assert out[1] == 10.0 and out[2] == 20.0 and out[3] == 30.0
    assert nbh.combine_layers(cur, None).equals(cur)


def test_knn_fill_uses_mean_of_k_nearest_known():
    xy = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [100, 0]], float)
    h = np.array([10, 20, 30, 40, 50, np.nan, 999], float)
    filled, imputed = nbh.knn_fill(xy, h, k=5)
    assert imputed.tolist() == [False] * 5 + [True, False]
    assert filled[5] == pytest.approx(np.mean([10, 20, 30, 40, 50]))   # 999 is far away
    assert np.isfinite(filled).all()


def test_knn_fill_edge_cases():
    xy = np.zeros((3, 2))
    f, imp = nbh.knn_fill(xy, np.array([1.0, 2.0, 3.0]))
    assert not imp.any() and f.tolist() == [1, 2, 3]
    f, imp = nbh.knn_fill(xy, np.array([np.nan, np.nan, np.nan]))
    assert not imp.any() and np.isnan(f).all()                # nothing to borrow from
    f, imp = nbh.knn_fill(np.array([[0, 0], [1, 0], [2, 0]], float), np.array([7.0, np.nan, 9.0]), k=5)
    assert imp.tolist() == [False, True, False] and f[1] == pytest.approx(8.0)   # k capped at known count


def test_build_height_table_covers_every_footprint():
    cen = pd.DataFrame({"DOITT_ID": [1, 2, 3, 4], "x": [0.0, 1.0, 2.0, 3.0], "y": 0.0})
    hts = pd.DataFrame({"DOITT_ID": [1, 2, 9], "height_roof": [10.0, np.nan, 50.0]})
    t = nbh.build_height_table(hts, cen, k=1).set_index("DOITT_ID")
    assert list(t.index) == [1, 2, 3, 4]                       # id 9 has no footprint
    assert t["height_roof"].notna().all()
    assert t.loc[1, "height_imputed"] == False and t.loc[2, "height_imputed"] == True  # noqa: E712
    assert t.loc[2, "height_roof"] == 10.0                      # nearest known
