"""Model smoke tests: output shape/range, learning on a tiny set, save/load round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

from faultaware.models.anomaly import ScoreNormaliser, sequence_nll
from faultaware.models.base import FaultPredictor, check_proba, create_model, model_class
from faultaware.models.deep import build_network, gather_last
from faultaware.training.torch_trainer import FocalLoss, make_classification_loss

TRAIN_CFG = {
    "batch_size": 32,
    "eval_batch_size": 256,
    "max_epochs": 25,
    "early_stopping_patience": 25,
    "lr": 5e-3,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "lr_scheduler": {"factor": 0.5, "patience": 10},
    "imbalance": "class_weight",
    "focal_gamma": 2.0,
    "device": "cpu",
    "num_threads": 2,
}

SEQ = {"embed_dim": 8, "hidden_dim": 16, "num_layers": 1, "dropout": 0.0}
CASES = {
    "always_zero": {},
    "random_rate": {},
    "rule_warn": {},
    "logreg": {"C": 1.0, "max_iter": 500},
    "random_forest": {"n_estimators": 20, "n_jobs": 1},
    "xgboost": {"n_estimators": 30, "max_depth": 3, "n_jobs": 1},
    "lstm": {"arch": "lstm", **SEQ},
    "gru": {"arch": "gru", **SEQ},
    "cnn_lstm": {
        "arch": "cnn_lstm",
        "embed_dim": 8,
        "conv_channels": 8,
        "kernel_size": 3,
        "hidden_dim": 16,
        "num_layers": 1,
        "dropout": 0.0,
    },
    "transformer": {
        "arch": "transformer",
        "embed_dim": 16,
        "num_heads": 2,
        "num_layers": 1,
        "ff_dim": 32,
        "dropout": 0.0,
    },
    "iforest": {"n_estimators": 50, "max_samples": 64, "n_jobs": 1},
    "lstm_ae": {**SEQ, "max_epochs": 5},
}
LEARNERS = ["logreg", "random_forest", "xgboost", "lstm", "gru", "cnn_lstm", "transformer"]


def _fit(name: str, tiny_data: dict) -> tuple[FaultPredictor, dict]:
    f = tiny_data["features"]
    model = create_model(
        name,
        CASES[name],
        seed=0,
        train_cfg=TRAIN_CFG,
        vocab_size=tiny_data["vocab"].size,
        max_len=16,
    )
    model.fit(f["train"], f["train"].y, f["val"], f["val"].y)
    return model, f


@pytest.mark.parametrize("name", list(CASES))
def test_output_shape_range_and_roundtrip(name: str, tiny_data: dict, tmp_path: Path) -> None:
    model, f = _fit(name, tiny_data)
    p = check_proba(model.predict_proba(f["test"]), len(f["test"].y))
    model.save(tmp_path / name)
    loaded = model_class(name).load(tmp_path / name)
    np.testing.assert_allclose(loaded.predict_proba(f["test"]), p, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", LEARNERS)
def test_supervised_models_learn_tiny_task(name: str, tiny_data: dict) -> None:
    model, f = _fit(name, tiny_data)
    p = model.predict_proba(f["train"])
    assert average_precision_score(f["train"].y, p) > 0.9


def test_lstm_ae_ranks_anomalies_higher(tiny_data: dict) -> None:
    model, f = _fit("lstm_ae", tiny_data)
    p = model.predict_proba(f["train"])
    assert p[f["train"].y == 1].mean() > p[f["train"].y == 0].mean()


def test_iforest_flags_clear_outlier(tiny_data: dict) -> None:
    from faultaware.features.build import FeatureSet

    model, f = _fit("iforest", tiny_data)
    tr = f["train"]
    normal = tr.y == 0
    outlier = FeatureSet(
        tr.counts[:1] * 0 + 50,
        tr.stats[:1] * 0 + 50,
        tr.sequences[:1],
        tr.lengths[:1],
        tr.y[:1],
        tr.lead_seconds[:1],
    )
    typical = float(np.median(model.predict_proba(tr)[normal]))
    assert float(model.predict_proba(outlier)[0]) > typical


def test_lstm_overfits_one_batch() -> None:
    torch.manual_seed(0)
    net = build_network({"arch": "lstm", **SEQ}, vocab_size=10, max_len=8)
    tokens = torch.randint(2, 10, (16, 8))
    lengths = torch.full((16,), 8)
    y = (tokens[:, 0] > 5).float()
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(tokens, lengths), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_gather_last_picks_last_valid_step() -> None:
    out = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
    got = gather_last(out, torch.tensor([2, 4]))
    torch.testing.assert_close(got, torch.stack([out[0, 1], out[1, 3]]))


def test_rnn_ignores_padding_after_length() -> None:
    torch.manual_seed(0)
    net = build_network({"arch": "gru", **SEQ}, vocab_size=10, max_len=6).eval()
    a = torch.tensor([[3, 4, 5, 0, 0, 0]])
    b = torch.tensor([[3, 4, 5, 7, 8, 9]])
    ln = torch.tensor([3])
    torch.testing.assert_close(net(a, ln), net(b, ln))


def test_cnn_lstm_is_causal() -> None:
    torch.manual_seed(0)
    net = build_network(CASES["cnn_lstm"], vocab_size=10, max_len=6).eval()
    a = torch.tensor([[3, 4, 5, 2, 2, 2]])
    b = torch.tensor([[3, 4, 5, 9, 9, 9]])
    torch.testing.assert_close(net(a, torch.tensor([3])), net(b, torch.tensor([3])))


def test_bidirectional_rnn_rejected() -> None:
    with pytest.raises(ValueError):
        build_network({"arch": "lstm", **SEQ, "bidirectional": True}, 10, 8)


def test_unknown_model_and_arch() -> None:
    with pytest.raises(KeyError):
        create_model("nope", {}, 0)
    with pytest.raises(ValueError):
        build_network({"arch": "nope"}, 10, 8)


def test_check_proba_validation() -> None:
    with pytest.raises(ValueError):
        check_proba(np.array([0.5, 1.5]), 2)
    with pytest.raises(ValueError):
        check_proba(np.array([0.5]), 2)


def test_losses() -> None:
    logits = torch.tensor([2.0, -2.0])
    target = torch.tensor([1.0, 0.0])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    assert FocalLoss(gamma=0.0, alpha=0.5)(logits, target) == pytest.approx(0.5 * bce.item())
    assert FocalLoss(gamma=2.0)(logits, target) < FocalLoss(gamma=0.0)(logits, target)
    for kind in ("class_weight", "focal", "none"):
        make_classification_loss(kind, 10.0, 2.0, torch.device("cpu"))
    with pytest.raises(ValueError):
        make_classification_loss("bad", 1.0, 2.0, torch.device("cpu"))


def test_sequence_nll_ignores_padding() -> None:
    logits = torch.zeros(1, 4, 5)
    tokens = torch.tensor([[2, 3, 0, 0]])
    assert sequence_nll(logits, tokens).item() == pytest.approx(np.log(5), rel=1e-5)


def test_score_normaliser_monotone() -> None:
    norm = ScoreNormaliser().fit(np.random.default_rng(0).gamma(2.0, size=1000))
    x = np.linspace(0, 20, 50)
    y = norm(x)
    assert np.all(np.diff(y) >= 0) and y.min() >= 0 and y.max() <= 1
