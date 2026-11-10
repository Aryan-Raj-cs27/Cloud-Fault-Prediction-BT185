"""Metrics match scikit-learn; thresholds; statistics; calibration and fusion."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from faultaware.eval.metrics import (
    best_threshold,
    bootstrap_ci,
    confusion_counts,
    evaluate_scores,
    ranking_metrics,
    reliability_curve,
    threshold_metrics,
)
from faultaware.eval.stats import fmt_mean_std, mean_std, paired_tests
from faultaware.eval.tables import to_markdown
from faultaware.fusion.risk import Calibrator, RiskFusion, fuse

RNG = np.random.default_rng(0)
Y = (RNG.random(2000) < 0.1).astype(int)
S = np.clip(0.6 * Y + 0.5 * RNG.random(2000), 0, 1)


def test_threshold_metrics_match_sklearn() -> None:
    pred = S >= 0.5
    m = threshold_metrics(Y, S, 0.5)
    assert m["precision"] == pytest.approx(precision_score(Y, pred))
    assert m["recall"] == pytest.approx(recall_score(Y, pred))
    assert m["f1"] == pytest.approx(f1_score(Y, pred))
    c = confusion_counts(Y, pred)
    assert sum(c.values()) == len(Y)
    assert m["fpr"] == pytest.approx(c["fp"] / (c["fp"] + c["tn"]))


def test_ranking_metrics_match_sklearn_and_degenerate() -> None:
    assert ranking_metrics(Y, S)["pr_auc"] == pytest.approx(average_precision_score(Y, S))
    assert np.isnan(ranking_metrics(np.zeros(5), np.zeros(5))["pr_auc"])


def test_best_threshold_maximises_f1() -> None:
    thr = best_threshold(Y, S, "f1")
    best = threshold_metrics(Y, S, thr)["f1"]
    for t in np.linspace(0, 1, 41):
        assert threshold_metrics(Y, S, t)["f1"] <= best + 1e-9


def test_best_threshold_cost_objective() -> None:
    thr = best_threshold(Y, S, "cost", fp_cost=1.0, fn_cost=100.0)
    assert threshold_metrics(Y, S, thr)["recall"] >= threshold_metrics(Y, S, 0.9)["recall"]
    with pytest.raises(ValueError):
        best_threshold(Y, S, "bad")


def test_evaluate_scores_keys() -> None:
    m = evaluate_scores(Y, S, 0.5)
    assert {"pr_auc", "roc_auc", "brier", "f1", "threshold"} <= set(m)


def test_bootstrap_ci_contains_point_estimate() -> None:
    lo, hi = bootstrap_ci(Y, S, average_precision_score, n_boot=200)
    assert lo <= average_precision_score(Y, S) <= hi


def test_reliability_curve_shapes() -> None:
    mp, obs, cnt = reliability_curve(Y, S, 5)
    assert len(mp) == len(obs) == len(cnt) and cnt.sum() == len(Y)
    mp1, _, _ = reliability_curve(Y, np.full(len(Y), 0.3), 5)
    assert len(mp1) == 1


def test_stats_helpers() -> None:
    assert mean_std([1.0, 3.0]) == (2.0, pytest.approx(np.sqrt(2)))
    assert mean_std([5.0]) == (5.0, 0.0)
    assert fmt_mean_std([1.0, 1.0]) == "1.000 ± 0.000"
    same = paired_tests([1, 2, 3], [1, 2, 3])
    assert same["wilcoxon_p"] == 1.0
    better = paired_tests(np.arange(10) + 5.0, np.arange(10) + RNG.random(10))
    assert better["mean_diff"] > 0 and better["wilcoxon_p"] < 0.05
    with pytest.raises(ValueError):
        paired_tests([1, 2], [1])


def test_markdown_table() -> None:
    import pandas as pd

    md = to_markdown(pd.DataFrame({"a": [1.0], "b": ["x"]}))
    assert md.splitlines()[0] == "| a | b |" and "1.0000" in md


@pytest.mark.parametrize("method", ["isotonic", "sigmoid", "none"])
def test_calibrator_outputs_probabilities(method: str) -> None:
    cal = Calibrator(method).fit(S, Y)
    p = cal(S)
    assert p.min() >= 0 and p.max() <= 1 and p.shape == S.shape


def test_isotonic_calibration_improves_brier() -> None:
    raw = np.clip(S * 0.3 + 0.6, 0, 1)  # badly over-confident scores
    cal = Calibrator("isotonic").fit(raw, Y)
    assert ranking_metrics(Y, cal(raw))["brier"] < ranking_metrics(Y, raw)["brier"]


def test_calibrator_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        Calibrator("magic")


def test_fuse_bounds_and_errors() -> None:
    r = fuse(np.array([0.2, 1.0]), np.array([1.0, 0.0]), 0.5)
    np.testing.assert_allclose(r, [0.6, 0.5])
    with pytest.raises(ValueError):
        fuse(np.array([0.1]), np.array([0.1]), 1.5)
    with pytest.raises(ValueError):
        fuse(np.array([0.1]), np.array([0.1, 0.2]), 0.5)


def test_risk_fusion_picks_alpha_and_roundtrips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    noise = RNG.random(len(Y))
    fusion = RiskFusion("sup", "anom").fit(S, noise, Y, [0.0, 0.5, 1.0])
    assert fusion.alpha == 1.0  # pure-noise anomaly score must not be selected
    r = fusion(S, noise)
    assert r.min() >= 0 and r.max() <= 1
    fusion.save(tmp_path / "f.joblib")
    np.testing.assert_allclose(RiskFusion.load(tmp_path / "f.joblib")(S, noise), r)


def test_early_warning_restricts_to_alive_sessions() -> None:
    from faultaware.eval.metrics import alive_mask, evaluate_early_warning

    y = np.array([1, 1, 0, 0, 1, 0])
    score = np.array([0.9, 0.1, 0.2, 0.8, 0.7, 0.1])
    lead = np.array([0.0, 30.0, 10.0, 0.0, 5.0, 3.0])
    assert alive_mask(lead).tolist() == [False, True, True, False, True, True]
    m = evaluate_early_warning(y, score, lead, 0.5)
    assert m["n_alive"] == 4 and m["n_alive_faults"] == 2
    # alive subset: y=[1,0,1,0], s=[0.1,0.2,0.7,0.1] -> one TP, one FN, no FP
    assert m["recall"] == 0.5 and m["precision"] == 1.0
