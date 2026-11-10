"""Determinism, configuration and tracking utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from faultaware.utils.config import load_config, project_path, to_dict
from faultaware.utils.seed import make_rng, set_seed
from faultaware.utils.tracking import Run, git_commit


def test_same_seed_same_outputs() -> None:
    set_seed(7)
    a = (np.random.rand(3), torch.rand(3))
    set_seed(7)
    b = (np.random.rand(3), torch.rand(3))
    np.testing.assert_array_equal(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])
    assert make_rng(1).random() == make_rng(1).random()


def test_negative_seed_rejected() -> None:
    with pytest.raises(ValueError):
        set_seed(-1)


def test_training_is_deterministic(tiny_data: dict) -> None:
    from faultaware.models.base import create_model
    from tests.unit.test_models import CASES, TRAIN_CFG

    f = tiny_data["features"]
    outs = []
    for _ in range(2):
        m = create_model(
            "gru",
            CASES["gru"],
            seed=3,
            train_cfg={**TRAIN_CFG, "max_epochs": 3},
            vocab_size=tiny_data["vocab"].size,
            max_len=16,
        )
        m.fit(f["train"], f["train"].y, f["val"], f["val"].y)
        outs.append(m.predict_proba(f["test"]))
    np.testing.assert_allclose(outs[0], outs[1], rtol=1e-6)


def test_load_config_groups_and_overrides() -> None:
    cfg = load_config(["train.max_epochs=2"])
    assert cfg.train.max_epochs == 2
    assert {"data", "train", "fusion", "scheduler", "simulator"} <= set(cfg)
    assert isinstance(to_dict(cfg), dict)
    assert project_path("x").is_absolute()
    assert project_path(Path("/abs")).is_absolute()


def test_load_config_missing_group(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("defaults: {data: missing}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_config(config_dir=tmp_path)


def test_run_record(tmp_path: Path) -> None:
    run = Run(name="t", config={"a": 1}, seed=0, runs_dir=tmp_path)
    run.log_metrics({"f1": 0.5})
    run.log_artefact("m", tmp_path / "m")
    record = json.loads(run.save().read_text(encoding="utf-8"))
    assert record["metrics"]["f1"] == 0.5 and record["seed"] == 0
    assert record["git_commit"] == git_commit()
