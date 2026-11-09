"""Unsupervised / semi-supervised anomaly detectors (model ladder step 6).

Both detectors are fitted on NORMAL training sessions only, so they can flag fault types never
seen as labelled anomalies. Raw anomaly scores are mapped to ``[0, 1]`` with a monotone
logistic transform whose location/scale come from the training-normal score distribution
(rank metrics such as PR-AUC are unaffected by this transform).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from torch import nn

from faultaware.features.build import FeatureSet
from faultaware.models.base import register
from faultaware.models.deep import to_tensors, trim
from faultaware.training.torch_trainer import Batch, TrainHistory, fit_model, resolve_device
from faultaware.utils.seed import set_seed


class ScoreNormaliser:
    """Monotone map of raw anomaly scores to ``[0, 1]``: ``sigmoid((log1p(s) - mu) / sd)``."""

    def __init__(self, mu: float = 0.0, sd: float = 1.0, shift: float = 0.0) -> None:
        self.mu, self.sd, self.shift = mu, sd, shift

    def fit(self, normal_scores: np.ndarray) -> ScoreNormaliser:
        """Fit on raw scores of normal training samples."""
        self.shift = float(-min(0.0, normal_scores.min()))
        z = np.log1p(normal_scores + self.shift)
        # Centre on the 99th percentile so that typical normal samples score well below 0.5.
        self.mu = float(np.percentile(z, 99))
        self.sd = float(max(z.std(), 1e-6))
        return self

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        z = (np.log1p(np.maximum(raw + self.shift, 0.0)) - self.mu) / self.sd
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def to_dict(self) -> dict[str, float]:
        return {"mu": self.mu, "sd": self.sd, "shift": self.shift}


@register("iforest")
class IsolationForestDetector:
    """Isolation Forest on ``log1p`` tabular features of normal training sessions."""

    def __init__(self, name: str, params: dict[str, Any], seed: int, **_: Any) -> None:
        self.name = name
        self.params = dict(params)
        self.seed = seed
        self.forest: IsolationForest | None = None
        self.norm = ScoreNormaliser()

    def _raw(self, X: FeatureSet) -> np.ndarray:
        assert self.forest is not None
        return -self.forest.score_samples(np.log1p(X.tabular))  # higher = more anomalous

    def fit(
        self, X_train: FeatureSet, y_train: np.ndarray, X_val: FeatureSet, y_val: np.ndarray
    ) -> IsolationForestDetector:
        normal = np.log1p(X_train.tabular[y_train == 0])
        self.forest = IsolationForest(
            n_estimators=self.params.get("n_estimators", 200),
            max_samples=min(self.params.get("max_samples", 4096), len(normal)),
            contamination=self.params.get("contamination", "auto"),
            n_jobs=self.params.get("n_jobs", 8),
            random_state=self.seed,
        ).fit(normal)
        self.norm.fit(-self.forest.score_samples(normal))
        return self

    def predict_proba(self, X: FeatureSet) -> np.ndarray:
        return self.norm(self._raw(X))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.forest, path / "forest.joblib")
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "name": self.name,
                    "params": self.params,
                    "seed": self.seed,
                    "norm": self.norm.to_dict(),
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> IsolationForestDetector:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        obj = cls(meta["name"], meta["params"], meta["seed"])
        obj.forest = joblib.load(path / "forest.joblib")
        obj.norm = ScoreNormaliser(**meta["norm"])
        return obj


class LSTMAutoencoder(nn.Module):
    """Sequence autoencoder: LSTM encoder summary -> teacher-forced LSTM decoder -> tokens.

    Shapes: ``tokens (B, T)``, ``lengths (B,)`` -> logits ``(B, T, vocab)``.
    """

    def __init__(
        self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed(tokens)
        enc_out, _ = self.encoder(emb)
        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, enc_out.size(-1))
        summary = enc_out.gather(1, idx).squeeze(1)  # (B, H)
        h0 = summary.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1).contiguous()
        c0 = torch.zeros_like(h0)
        # Decoder input is the sequence shifted right by one (PAD acts as <BOS>).
        dec_in = nn.functional.pad(emb[:, :-1], (0, 0, 1, 0))
        dec_out, _ = self.decoder(dec_in, (h0, c0))
        return self.out(dec_out)


def sequence_nll(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Mean per-token negative log-likelihood of each sequence, ignoring padding.

    Args:
        logits: ``(B, T, V)``.
        tokens: ``(B, T)``.

    Returns:
        ``(B,)`` reconstruction errors.
    """
    nll = nn.functional.cross_entropy(logits.transpose(1, 2), tokens, reduction="none")
    mask = (tokens != 0).float()
    return (nll * mask).sum(1) / mask.sum(1).clamp(min=1.0)


@register("lstm_ae")
class LSTMAutoencoderDetector:
    """LSTM-autoencoder anomaly detector; score = reconstruction NLL."""

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
        self.net: LSTMAutoencoder | None = None
        self.norm = ScoreNormaliser()
        self.history = TrainHistory()

    def _build(self) -> LSTMAutoencoder:
        return LSTMAutoencoder(
            self.vocab_size,
            self.params["embed_dim"],
            self.params["hidden_dim"],
            self.params["num_layers"],
            self.params["dropout"],
        ).to(self.device)

    @torch.no_grad()
    def raw_scores(self, X: FeatureSet) -> np.ndarray:
        """Reconstruction NLL per session, shape ``(n,)``."""
        assert self.net is not None
        self.net.eval()
        tokens, lengths = to_tensors(X, self.device)
        bs = self.train_cfg.get("eval_batch_size", 4096)
        out = [
            sequence_nll(
                self.net(t := trim(tokens[i : i + bs], lengths[i : i + bs]), lengths[i : i + bs]), t
            )
            for i in range(0, len(tokens), bs)
        ]
        return torch.cat(out).cpu().numpy().astype(np.float64)

    def fit(
        self, X_train: FeatureSet, y_train: np.ndarray, X_val: FeatureSet, y_val: np.ndarray
    ) -> LSTMAutoencoderDetector:
        set_seed(self.seed)
        self.net = self._build()
        tokens, lengths = to_tensors(X_train, self.device)
        normal = torch.as_tensor(y_train == 0, device=self.device)
        tokens, lengths = tokens[normal], lengths[normal]
        val_tok, val_len = to_tensors(X_val, self.device)
        val_normal = torch.as_tensor(y_val == 0, device=self.device)
        val_tok, val_len = val_tok[val_normal], val_len[val_normal]

        def loss_step(net: nn.Module, batch: Batch) -> torch.Tensor:
            tok, ln = batch
            tok = trim(tok, ln)
            return sequence_nll(net(tok, ln), tok).mean()

        def validate(net: nn.Module) -> float:
            bs = self.train_cfg.get("eval_batch_size", 4096)
            losses = [
                sequence_nll(
                    net(t := trim(val_tok[i : i + bs], val_len[i : i + bs]), val_len[i : i + bs]), t
                )
                for i in range(0, len(val_tok), bs)
            ]
            return -float(torch.cat(losses).mean())  # maximise negative loss

        self.history = fit_model(
            self.net,
            (tokens, lengths),
            loss_step,
            validate,
            self.train_cfg,
            self.seed,
            self.params.get("max_epochs"),
        )
        self.norm.fit(self.raw_scores(_subset(X_train, y_train == 0)))
        return self

    def predict_proba(self, X: FeatureSet) -> np.ndarray:
        return self.norm(self.raw_scores(X))

    def save(self, path: Path) -> None:
        assert self.net is not None
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "weights.pt")
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "name": self.name,
                    "params": self.params,
                    "seed": self.seed,
                    "train_cfg": self.train_cfg,
                    "vocab_size": self.vocab_size,
                    "max_len": self.max_len,
                    "norm": self.norm.to_dict(),
                    "history": self.history.epochs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> LSTMAutoencoderDetector:
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
        obj.net = obj._build()
        obj.net.load_state_dict(
            torch.load(path / "weights.pt", map_location=obj.device, weights_only=True)
        )
        obj.norm = ScoreNormaliser(**meta["norm"])
        obj.history.epochs = meta.get("history", [])
        return obj


def _subset(X: FeatureSet, mask: np.ndarray) -> FeatureSet:
    return FeatureSet(
        X.counts[mask],
        X.stats[mask],
        X.sequences[mask],
        X.lengths[mask],
        X.y[mask],
        X.lead_seconds[mask],
    )
