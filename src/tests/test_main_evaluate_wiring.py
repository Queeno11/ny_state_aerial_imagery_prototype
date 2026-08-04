"""The evaluate flag wiring in main.run().

Verifies that run(evaluate=True) calls src.evaluation.run_evaluation once with
the generated savename + params, and that an exception raised there is swallowed
(prediction artifacts must stay intact) rather than propagating.
"""

import inspect

import src.main as main


def test_run_has_evaluate_flag():
    assert "evaluate" in inspect.signature(main.run).parameters


def test_evaluate_calls_run_evaluation(monkeypatch):
    calls = []

    import src.evaluation as ev
    monkeypatch.setattr(
        ev, "run_evaluation",
        lambda savename, params, **kw: calls.append((savename, params, kw)),
    )
    # Neutralize the heavy machinery so only the evaluate branch executes.
    monkeypatch.setattr(main, "generate_parameters_log", lambda *a, **k: None)

    params = {"model_name": "scalemae", "run_id": "unit_eval", "footprints_source": "ms_us"}
    main.run(params, train=False, compute_loss=False,
             generate_predictions=False, evaluate=True)

    assert len(calls) == 1
    savename, passed, kwargs = calls[0]
    assert savename == main.generate_savename("unit_eval")
    assert passed["footprints_source"] == "ms_us"
    # A pipeline run must evaluate BOTH halves: the US parts and the NYC parts
    # (CSA event study + Hudson Yards) off this run's NYC prediction pass.
    assert kwargs["mode"] == "both"


def test_evaluate_swallows_errors(monkeypatch):
    import src.evaluation as ev

    def _boom(savename, params, **kw):
        raise RuntimeError("evaluation exploded")

    monkeypatch.setattr(ev, "run_evaluation", _boom)
    monkeypatch.setattr(main, "generate_parameters_log", lambda *a, **k: None)

    # Must not raise.
    main.run({"model_name": "scalemae", "run_id": "unit_eval2"},
             train=False, compute_loss=False,
             generate_predictions=False, evaluate=True)
