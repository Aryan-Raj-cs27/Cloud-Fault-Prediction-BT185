"""Deep sequence classifiers over event-template sequences (model ladder steps 4–5).

All networks take ``tokens`` of shape ``(batch, max_len)`` (0 = padding) and ``lengths`` of
shape ``(batch,)`` and return logits of shape ``(batch,)``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from faultaware.features.build import FeatureSet
from faultaware.models.base import register
from faultaware.training.torch_trainer import (
    Batch,
    TrainHistory,
    fit_model,
    make_classification_loss,
    resolve_device,
)
from faultaware.utils.seed import set_seed


def trim(tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Drop trailing padding columns beyond the longest sequence in the batch.

    Outputs are unchanged because every network is causal/masked and reads position
    ``length - 1``; this only removes wasted computation on padding.
    """
    return tokens[:, : int(lengths.max().item())]


def gather_last(outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Select the output at position ``length - 1`` for each sequence.

    Args:
        outputs: ``(batch, time, features)``.
        lengths: ``(batch,)`` with values in ``[1, time]``.

    Returns:
        ``(batch, features)``.
    """
    idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, outputs.size(-1))
    return outputs.gather(1, idx).squeeze(1)


class RNNClassifier(nn.Module):
    """Embedding -> (LSTM|GRU) -> last valid hidden state -> linear."""

    def __init__(
        self,
        vocab_size: int,
        arch: str,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if bidirectional:
            raise ValueError("bidirectional RNNs would read future events; not allowed")
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[arch]
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(self.embed(tokens))
        return self.head(self.drop(gather_last(out, lengths))).squeeze(-1)


class CNNLSTMClassifier(nn.Module):
    """Embedding -> causal 1-D convolution -> LSTM -> linear.

    The convolution is left-padded (causal) so position ``t`` only sees events ``<= t``.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        conv_channels: int,
        kernel_size: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(embed_dim, conv_channels, kernel_size)
        self.lstm = nn.LSTM(
            conv_channels,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens).transpose(1, 2)  # (B, E, T)
        x = torch.relu(self.conv(nn.functional.pad(x, (self.pad, 0))))
        out, _ = self.lstm(x.transpose(1, 2))  # (B, T, H)
        return self.head(self.drop(gather_last(out, lengths))).squeeze(-1)


class TransformerClassifier(nn.Module):
    """Token + learned positional embeddings -> Transformer encoder -> masked mean pool."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(embed_dim, 1)
        self.scale = math.sqrt(embed_dim)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        pad_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)  # True = padding
        x = self.embed(tokens) * self.scale + self.pos(positions)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        keep = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


def build_network(params: dict[str, Any], vocab_size: int, max_len: int) -> nn.Module:
    """Construct the network described by ``params["arch"]``."""
    arch = params["arch"]
    if arch in {"lstm", "gru"}:
        return RNNClassifier(
            vocab_size,
            arch,
            params["embed_dim"],
            params["hidden_dim"],
            params["num_layers"],
            params["dropout"],
            params.get("bidirectional", False),
        )
    if arch == "cnn_lstm":
        return CNNLSTMClassifier(
            vocab_size,
            params["embed_dim"],
            params["conv_channels"],
            params["kernel_size"],
            params["hidden_dim"],
            params["num_layers"],
            params["dropout"],
        )
    if arch == "transformer":
        return TransformerClassifier(
            vocab_size,
            max_len,
            params["embed_dim"],
            params["num_heads"],
            params["num_layers"],
            params["ff_dim"],
            params["dropout"],
        )
    raise ValueError(f"unknown arch {arch!r}")


def to_tensors(X: FeatureSet, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Move padded sequences and lengths to ``device``."""
    return (torch.as_tensor(X.sequences, device=device), torch.as_tensor(X.lengths, device=device))


@torch.no_grad()
def predict_logits(
    net: nn.Module, X: FeatureSet, device: torch.device, batch_size: int = 4096
) -> np.ndarray:
    """Run ``net`` over all sequences; returns logits of shape ``(n,)``."""
    net.eval()
    tokens, lengths = to_tensors(X, device)
    out = [
        net(
            trim(tokens[i : i + batch_size], lengths[i : i + batch_size]),
            lengths[i : i + batch_size],
        )
        for i in range(0, len(tokens), batch_size)
    ]
    return torch.cat(out).float().cpu().numpy() if out else np.zeros(0)


class SequenceModel:
    """FaultPredictor wrapper around a PyTorch sequence classifier."""

    def __init__(
        self,
        name: str,
        params: dict[str, Any],
        seed: int,
        train_cfg: dict[str, Any] | None = None,
        vocab_size: int = 2,
        max_len: int = 64,
        **_: Any,
    ) -> None:
        self.name = name
        self.params = dict(params)
        self.seed = seed
        self.train_cfg = dict(train_cfg or {})
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.device = resolve_device(self.train_cfg.get("device", "auto"))
        self.net: nn.Module | None = None
        self.history = TrainHistory()

    def fit(
        self, X_train: FeatureSet, y_train: np.ndarray, X_val: FeatureSet, y_val: np.ndarray
    ) -> SequenceModel:
        set_seed(self.seed)
        torch.set_num_threads(int(self.train_cfg.get("num_threads", 8)))
        self.net = build_network(self.params, self.vocab_size, self.max_len).to(self.device)
        pos = max(1, int(y_train.sum()))
        loss_fn = make_classification_loss(
            self.params.get("imbalance", self.train_cfg.get("imbalance", "class_weight")),
            (len(y_train) - pos) / pos,
            self.train_cfg.get("focal_gamma", 2.0),
            self.device,
        )
        tokens, lengths = to_tensors(X_train, self.device)
        target = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)

        def loss_step(net: nn.Module, batch: Batch) -> torch.Tensor:
            tok, ln, yb = batch
            return loss_fn(net(trim(tok, ln), ln), yb)

        def validate(net: nn.Module) -> float:
            logits = predict_logits(
                net, X_val, self.device, self.train_cfg.get("eval_batch_size", 4096)
            )
            return float(average_precision_score(y_val, logits))

        self.history = fit_model(
            self.net,
            (tokens, lengths, target),
            loss_step,
            validate,
            self.train_cfg,
            self.seed,
            self.params.get("max_epochs"),
        )
        return self

    def predict_proba(self, X: FeatureSet) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("model is not fitted")
        logits = predict_logits(
            self.net, X, self.device, self.train_cfg.get("eval_batch_size", 4096)
        )
        return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

    def save(self, path: Path) -> None:
        if self.net is None:
            raise RuntimeError("model is not fitted")
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "weights.pt")
        meta = {
            "name": self.name,
            "params": self.params,
            "seed": self.seed,
            "train_cfg": self.train_cfg,
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "history": self.history.epochs,
            "best_epoch": self.history.best_epoch,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> SequenceModel:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        train_cfg = dict(meta["train_cfg"])
        if device is not None:
            train_cfg["device"] = device
        obj = cls(
            meta["name"],
            meta["params"],
            meta["seed"],
            train_cfg,
            meta["vocab_size"],
            meta["max_len"],
        )
        obj.net = build_network(obj.params, obj.vocab_size, obj.max_len).to(obj.device)
        obj.net.load_state_dict(
            torch.load(path / "weights.pt", map_location=obj.device, weights_only=True)
        )
        obj.net.eval()
        obj.history.epochs = meta.get("history", [])
        obj.history.best_epoch = meta.get("best_epoch", -1)
        return obj


for _name in ("lstm", "gru", "cnn_lstm", "transformer"):
    register(_name)(SequenceModel)
