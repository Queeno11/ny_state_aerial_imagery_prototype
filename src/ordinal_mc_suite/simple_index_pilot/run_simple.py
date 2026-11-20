"""One bounded five-replication run of the three-map linear-index ladder."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import time
import traceback

import numba
import numpy as np
import pandas as pd
import scipy
import statsmodels
from threadpoolctl import threadpool_limits

from data import CASES, external_views, test_panels, training
from estimators import estimators, tune
from evaluate import evaluate
from losses import prepare

HERE = Path(__file__).resolve().parent


def json_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def write(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, default=json_numpy, allow_nan=False)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def create_data(cfg, rep):
    seeds, panel, valid = training(cfg, rep)
    fit_links = external_views(panel, seeds["fit_links"], cfg["stability_views"])
    val_links = external_views(valid, seeds["validation_links"], cfg["stability_views"])
    return seeds, panel, valid, fit_links, val_links


def case_data(case, panel, valid, fit_links, val_links):
    fit = prepare(panel, case, fit_links)
    validation = prepare(valid, case, val_links, (fit.denom_w, fit.denom_d))
    for source, observed in ((panel, fit), (valid, validation)):
        mean, _ = source.outcomes(case)
        standardized = (observed.w-mean)/source.cities.sigma[:, None, None]
        np.testing.assert_allclose(standardized, source.epsilon, atol=2e-14, rtol=1e-14)
    return fit, validation


def calibration_stage(cfg, out):
    rep = cfg["calibration_replication"]
    seeds, panel, valid, fit_links, val_links = create_data(cfg, rep)
    weights, rows = {}, []
    for case in CASES:
        start = time.perf_counter()
        data, validation = case_data(case, panel, valid, fit_links, val_links)
        weights[case], candidates = tune(data, validation, cfg)
        rows.extend(dict(case=case, **item) for item in candidates)
        print(f"Calibration {case}: {weights[case]['grid_winner']}, "
              f"positive {weights[case]['full']}, {time.perf_counter()-start:.2f}s", flush=True)
    write(out / "frozen_selection.json", dict(rep=rep, fitting_and_validation_seeds=seeds,
                                              weights=weights, evaluated_tests=False))
    pd.DataFrame(rows).to_csv(out / "calibration_candidates.csv", index=False)
    return weights, set(seeds.values()), rows


def run(cfg, out):
    start = time.perf_counter()
    tuned, seed_set, candidate_rows = calibration_stage(cfg, out)
    combined = {name: [] for name in ("structural", "outcomes", "coefficients", "ranks")}
    diagnostics, val_rows = [], []
    for rep in cfg["replications"]:
        t0 = time.perf_counter()
        dest = out / f"rep_{rep}"
        dest.mkdir()
        seeds, panel, valid, fit_links, val_links = create_data(cfg, rep)
        assert not set(seeds.values()) & seed_set
        seed_set.update(seeds.values())
        first = None
        fitted = {}
        denoms = {}
        for case in CASES:
            t1 = time.perf_counter()
            fit, validation = case_data(case, panel, valid, fit_links, val_links)
            if first is None:
                first = fit.x
            else:
                assert np.array_equal(first, fit.x)
            models, variants, values, audits = estimators(fit, validation, cfg, tuned[case])
            fitted[case] = (models, variants)
            denoms[case] = dict(W=fit.denom_w, D=fit.denom_d)
            for method, info in audits.items():
                diagnostics.append(dict(rep=rep, case=case, method=method, **info))
            for method, info in values.items():
                for label in ("own", "common_library"):
                    val_rows.append(dict(rep=rep, case=case, method=method, head=label,
                                         library_choice=info["library_choice"], **info[label]))
            print(f"Evaluation fit {rep}/{case}, {len(models)} methods, {time.perf_counter()-t1:.2f}s", flush=True)
        cutoff, tests = test_panels(cfg, seeds, panel.cities)
        write(dest / "design.json", dict(seeds=seeds, fit_tract_id_sha256=digest(panel.tract_id.tobytes()),
                                         validation_tract_id_sha256=digest(valid.tract_id.tobytes()),
                                         fit_tracts=int(panel.tract_id.size), validation_tracts=int(valid.tract_id.size),
                                         auxiliary_fit_links=int(fit_links.size), denominator=denoms))
        for case, (models, variants) in fitted.items():
            results = evaluate(rep, case, models, variants, cutoff, tests, cfg)
            for name, rows in results.items():
                combined[name].extend(rows)
            write(dest / f"case_{case}_models.json",
                  {method: dict(own=asdict(model), equal_library=asdict(variants[method]))
                   for method, model in models.items()})
        print(f"Replication {rep} complete in {time.perf_counter()-t0:.2f}s", flush=True)
    combined["optimizer"] = diagnostics
    combined["validation"] = val_rows
    combined["calibration_candidates"] = candidate_rows
    with sqlite3.connect(out / "results.sqlite") as db:
        for name, rows in combined.items():
            table = pd.DataFrame(rows)
            if name == "optimizer":
                for key in ("training_weights", "random_index_direction"):
                    if key in table.columns:
                        table[key] = table[key].map(lambda x: json.dumps(x) if isinstance(x, list) else None)
            table.to_sql(name, db, if_exists="fail", index=False)
            if name != "calibration_candidates":
                table.to_csv(out / f"{name}.csv", index=False)
    count = len(combined["coefficients"])
    assert count == 175 and len(candidate_rows) == 39
    write(out / "completion.json", dict(fitted_estimators=count, evaluation_replications=cfg["replications"],
                                        tuning_scorer_fits=len(candidate_rows), elapsed_seconds=time.perf_counter()-start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config_path = HERE / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["replications"] == [1, 2, 3, 4, 5] and cfg["calibration_replication"] not in cfg["replications"]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / "manifest.json", dict(config=cfg, config_sha256=digest(config_path.read_bytes()),
                                      source_sha256={p.name: digest(p.read_bytes()) for p in HERE.glob("*.py")},
                                      versions=dict(python=platform.python_version(), numpy=np.__version__,
                                                    scipy=scipy.__version__, numba=numba.__version__,
                                                    statsmodels=statsmodels.__version__, pandas=pd.__version__),
                                      selection="actual observed rank + .3 external stability + level MSE/V_W "
                                                "+ paired change MSE/V_D on independent validation tracts"))
    try:
        with threadpool_limits(limits=1):
            run(cfg, out)
    except Exception:
        (out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
