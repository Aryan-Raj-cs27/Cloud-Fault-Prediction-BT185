"""Train one model on one observation window and seed, store its scores and run record."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from faultaware.eval.metrics import best_threshold, evaluate_scores
from faultaware.features.store import DataBundle, get_features, window_tag
from faultaware.models.base import FaultPredictor, check_proba, create_model
from faultaware.utils.config import CONFIG_DIR, project_path, to_dict
from faultaware.utils.logging import get_logger
from faultaware.utils.seed import set_seed
from faultaware.utils.tracking import Run

log = get_logger(__name__)


def model_params(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Hyperparameters of ``name`` (variants inherit their ``base``; tuned values override)."""
    all_params = OmegaConf.to_container(OmegaConf.load(config_dir / "model" / "models.yaml"))
    assert isinstance(all_params, dict)
    params = dict(all_params[name])
    if "base" in params:
        params = {**all_params[params["base"]], **params}
    tuned_file = config_dir / "model" / "tuned.yaml"
    if tuned_file.is_file():
        tuned = OmegaConf.to_container(OmegaConf.load(tuned_file))
        assert isinstance(tuned, dict)
        best = tuned.get(params.get("base", name), {}) or {}
        params.update({k: v for k, v in best.items() if not str(k).startswith("_")})
    return params


def preds_path(cfg: DictConfig, tag: str, model: str, seed: int) -> Path:
    """Where validation/test scores of a run are stored."""
    return project_path(cfg.paths.models_dir) / "preds" / tag / model / f"seed{seed}.npz"


def artefact_dir(cfg: DictConfig, tag: str, model: str) -> Path:
    """Directory holding the saved (seed-0 / serving) model."""
    return project_path(cfg.paths.models_dir) / "artefacts" / tag / model


def train_one(
    cfg: DictConfig,
    bundle: DataBundle,
    model_name: str,
    seed: int,
    observe_seconds: float | None,
    save_model: bool,
) -> dict[str, float]:
    """Fit a model, score val/test, record the run.

    Test scores are stored for the single final evaluation (``scripts/evaluate.py``); only
    VALIDATION metrics are logged here, and the decision threshold is chosen on validation.

    Returns:
        Validation metrics at the validation-optimal threshold.
    """
    tag = window_tag(observe_seconds)
    train = get_features(cfg, bundle, "train", observe_seconds)
    val = get_features(cfg, bundle, "val", observe_seconds)
    test = get_features(cfg, bundle, "test", observe_seconds)
    params = model_params(model_name)
    set_seed(seed)
    model: FaultPredictor = create_model(
        model_name,
        params,
        seed,
        train_cfg=to_dict(cfg.train),
        vocab_size=bundle.vocab.size,
        max_len=bundle.max_len,
    )
    started = time.perf_counter()
    model.fit(train, train.y, val, val.y)
    fit_seconds = time.perf_counter() - started
    val_scores = check_proba(model.predict_proba(val), len(val.y))
    started = time.perf_counter()
    test_scores = check_proba(model.predict_proba(test), len(test.y))
    latency_ms = 1000 * (time.perf_counter() - started) / max(1, len(test.y))
    threshold = best_threshold(
        val.y,
        val_scores,
        cfg.fusion.threshold_objective,
        cfg.fusion.cost.false_positive,
        cfg.fusion.cost.false_negative,
    )
    val_metrics = evaluate_scores(val.y, val_scores, threshold)

    out = preds_path(cfg, tag, model_name, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    history = getattr(getattr(model, "history", None), "epochs", [])
    np.savez_compressed(
        out,
        val=val_scores.astype(np.float32),
        test=test_scores.astype(np.float32),
        threshold=threshold,
        fit_seconds=fit_seconds,
        latency_ms=latency_ms,
        train_loss=np.array([h["train_loss"] for h in history], dtype=np.float32),
        val_curve=np.array([h["val_score"] for h in history], dtype=np.float32),
    )
    run = Run(
        name=f"{model_name}-{tag}",
        config={**to_dict(cfg), "model_params": params},
        seed=seed,
        runs_dir=project_path(cfg.paths.runs_dir),
    )
    run.log_metrics(
        {f"val_{k}": v for k, v in val_metrics.items()}
        | {"fit_seconds": fit_seconds, "latency_ms_per_sample": latency_ms}
    )
    run.log_artefact("predictions", out)
    if save_model:
        adir = artefact_dir(cfg, tag, model_name)
        model.save(adir)
        run.log_artefact("model", adir)
    run.save()
    log.info(
        "%s [%s] seed=%d val PR-AUC=%.4f F1=%.4f (%.1fs)",
        model_name,
        tag,
        seed,
        val_metrics["pr_auc"],
        val_metrics["f1"],
        fit_seconds,
    )
    return val_metrics
