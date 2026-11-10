"""Train the model ladder over seeds and observation windows.

Usage:
    python scripts/train.py                              # all models, all windows, 5 seeds
    python scripts/train.py --models xgboost lstm --windows full --seeds 0
    python scripts/train.py --skip-existing train.max_epochs=5
"""

from __future__ import annotations

import argparse
import json

from faultaware.features.store import load_bundle, save_feature_spec, window_tag
from faultaware.training.runner import preds_path, train_one
from faultaware.utils.config import load_config, project_path
from faultaware.utils.logging import get_logger

log = get_logger("train")

DEFAULT_MODELS = [
    "always_zero",
    "random_rate",
    "rule_warn",
    "logreg",
    "random_forest",
    "xgboost",
    "lstm",
    "gru",
    "cnn_lstm",
    "transformer",
    "iforest",
    "lstm_ae",
    "random_forest_smote",
    "lstm_focal",
]


def parse_window(text: str) -> float | None:
    """``full`` -> None, ``60``/``w60`` -> 60.0."""
    return None if text == "full" else float(text.lstrip("w"))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--windows",
        nargs="+",
        default=None,
        help="observation windows: full, 60, 10 (default: from config)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args, overrides = parser.parse_known_args()
    cfg = load_config(overrides)
    seeds = args.seeds if args.seeds is not None else list(cfg.seeds_for_reporting)
    windows = (
        [parse_window(w) for w in args.windows]
        if args.windows
        else list(cfg.data.observation_windows)
    )
    bundle = load_bundle(cfg)
    levels = json.loads(
        project_path(cfg.data.templates_file).with_name("hdfs_levels.json").read_text("utf-8")
    )
    save_feature_spec(
        bundle, levels, project_path(cfg.paths.models_dir) / "serving" / "feature_spec.json"
    )
    log.info("vocabulary size %d (incl. PAD/UNK)", bundle.vocab.size)
    for window in windows:
        for model in args.models:
            for seed in seeds:
                if (
                    args.skip_existing
                    and preds_path(cfg, window_tag(window), model, seed).is_file()
                ):
                    continue
                train_one(cfg, bundle, model, seed, window, save_model=(seed == seeds[0]))


if __name__ == "__main__":
    main()
