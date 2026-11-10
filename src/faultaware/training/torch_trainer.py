"""Generic PyTorch training with early stopping, LR scheduling and gradient clipping.

Hygiene rules implemented (CLAUDE.md §7.4): early stopping on a validation metric with
best-checkpoint restore, ReduceLROnPlateau, gradient clipping, per-epoch train/val history.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from faultaware.utils.logging import get_logger

log = get_logger(__name__)

Batch = tuple[torch.Tensor, ...]


def resolve_device(device: str) -> torch.device:
    """Map ``"auto" | "cpu" | "cuda"`` to a torch device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class FocalLoss(nn.Module):
    """Binary focal loss on logits (Lin et al., 2017), mean-reduced.

    Args:
        gamma: Focusing parameter (0 = BCE).
        alpha: Weight of the positive class in ``[0, 1]``.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


def make_classification_loss(
    imbalance: str, pos_weight: float, gamma: float, device: torch.device
) -> nn.Module:
    """Build the loss for a binary classifier.

    Args:
        imbalance: ``"class_weight"`` (weighted BCE), ``"focal"`` or ``"none"``.
        pos_weight: Negative/positive ratio in the training set.
        gamma: Focal-loss gamma.
        device: Device for the weight tensor.

    Returns:
        A loss module taking ``(logits, targets)``.
    """
    if imbalance == "class_weight":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    if imbalance == "focal":
        alpha = pos_weight / (1.0 + pos_weight)  # up-weight the rare class
        return FocalLoss(gamma=gamma, alpha=alpha)
    if imbalance == "none":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"unknown imbalance strategy {imbalance!r}")


@dataclass
class TrainHistory:
    """Per-epoch metrics and the selected checkpoint."""

    epochs: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_score: float = -np.inf


def iterate_batches(
    tensors: tuple[torch.Tensor, ...],
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> list[Batch]:
    """Split tensors (same first dim) into mini-batches.

    Shuffling permutes *training samples* (independent sessions) — not time order within a
    sequence, and never across splits, so it does not cause temporal leakage.
    """
    n = tensors[0].shape[0]
    idx = torch.randperm(n, generator=generator, device="cpu") if shuffle else torch.arange(n)
    idx = idx.to(tensors[0].device)
    return [tuple(t[idx[i : i + batch_size]] for t in tensors) for i in range(0, n, batch_size)]


def fit_model(
    model: nn.Module,
    train_tensors: tuple[torch.Tensor, ...],
    loss_step: Callable[[nn.Module, Batch], torch.Tensor],
    validate: Callable[[nn.Module], float],
    cfg: dict[str, Any],
    seed: int,
    max_epochs: int | None = None,
) -> TrainHistory:
    """Train ``model`` and restore the weights of the best validation epoch.

    Args:
        model: Module already moved to its device.
        train_tensors: Training tensors, batched together.
        loss_step: Computes the scalar loss of one batch.
        validate: Returns the validation score to MAXIMISE (e.g. PR-AUC or -loss).
        cfg: Training config (``configs/train/default.yaml``).
        seed: Seed for batch shuffling.
        max_epochs: Overrides ``cfg["max_epochs"]`` when given.

    Returns:
        The training history.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=cfg["lr_scheduler"]["factor"],
        patience=cfg["lr_scheduler"]["patience"],
    )
    gen = torch.Generator().manual_seed(seed)
    hist = TrainHistory()
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    for epoch in range(max_epochs or cfg["max_epochs"]):
        started = time.perf_counter()
        model.train()
        losses = []
        for batch in iterate_batches(train_tensors, cfg["batch_size"], True, gen):
            opt.zero_grad(set_to_none=True)
            loss = loss_step(model, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            losses.append(loss.detach())
        train_loss = float(torch.stack(losses).mean())
        model.eval()
        with torch.no_grad():
            score = float(validate(model))
        sched.step(score)
        hist.epochs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_score": score,
                "lr": opt.param_groups[0]["lr"],
                "seconds": time.perf_counter() - started,
            }
        )
        log.info("epoch %02d train_loss=%.4f val=%.4f", epoch, train_loss, score)
        if score > hist.best_score + 1e-5:
            hist.best_score, hist.best_epoch, bad_epochs = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["early_stopping_patience"]:
                log.info("early stopping at epoch %d (best %d)", epoch, hist.best_epoch)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist
