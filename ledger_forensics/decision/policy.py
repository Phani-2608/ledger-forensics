"""Calibration and the review policy.

A raw model output is not a risk score. If the system says 0.70 it should be
wrong about three times in ten, and nothing guarantees that until you check.
So we calibrate, then report reliability and expected calibration error, and
only then build the abstention threshold on top.

The threshold itself is an economic decision, not a round number. A missed
forgery costs the amount claimed. A false alarm costs an analyst's time. The
operating point is the one that minimises expected dollars lost, and the
coverage curve shows what each review budget actually buys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass
class CostModel:
    """Dollars. Defaults are stated in the README, not hidden in code."""
    review_cost: float = 8.0          # analyst time per reviewed case
    recovery_rate: float = 0.9        # fraction of exposure saved when caught
    false_accuse_cost: float = 25.0   # cost of wrongly escalating a good customer


class Calibrator:
    """Platt scaling or isotonic regression, fitted on a held-out split."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.model = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        p = np.asarray(p, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=int).reshape(-1)
        if len(np.unique(y)) < 2:
            self.model = None
            return self
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        else:
            self.model = LogisticRegression(max_iter=1000).fit(
                p.reshape(-1, 1), y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float).reshape(-1)
        if self.model is None:
            return np.clip(p, 0, 1)
        if self.method == "isotonic":
            return np.clip(self.model.predict(p), 0, 1)
        return np.clip(self.model.predict_proba(p.reshape(-1, 1))[:, 1], 0, 1)


def expected_calibration_error(p: np.ndarray, y: np.ndarray,
                               bins: int = 10) -> Tuple[float, List[Dict]]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        n = int(sel.sum())
        if n == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": 0,
                         "confidence": float("nan"), "accuracy": float("nan")})
            continue
        conf, acc = float(p[sel].mean()), float(y[sel].mean())
        ece += (n / len(p)) * abs(conf - acc)
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": n,
                     "confidence": conf, "accuracy": acc})
    return float(ece), rows


def ranking_metrics(p: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    out: Dict[str, float] = {}
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    out["brier"] = float(brier_score_loss(y, np.clip(p, 0, 1)))
    return out


def expected_loss(p: np.ndarray, y: np.ndarray, exposure: np.ndarray,
                  threshold: float, cost: CostModel) -> Dict[str, float]:
    """Dollars lost at one operating point.

    Above threshold the case is reviewed: we pay review cost, recover most of
    the exposure if it really was fraud, and pay a false-accusation cost if it
    was not. Below threshold it is paid out: full exposure lost if it was
    fraud, nothing otherwise.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    exposure = np.asarray(exposure, dtype=float)

    flagged = p >= threshold
    loss = 0.0
    loss += flagged.sum() * cost.review_cost
    caught = flagged & (y == 1)
    loss -= caught.sum() and float((exposure[caught] * cost.recovery_rate).sum())
    false_alarm = flagged & (y == 0)
    loss += false_alarm.sum() * cost.false_accuse_cost
    missed = (~flagged) & (y == 1)
    loss += float(exposure[missed].sum())

    baseline = float(exposure[y == 1].sum())  # review nothing, lose everything
    return {
        "threshold": float(threshold),
        "expected_loss": float(loss),
        "net_saving": float(baseline - loss),
        "coverage": float(flagged.mean()),
        "caught": int(caught.sum()),
        "missed": int(missed.sum()),
        "false_alarms": int(false_alarm.sum()),
        "recall": float(caught.sum() / max(1, (y == 1).sum())),
        "precision": float(caught.sum() / max(1, flagged.sum())),
    }


def sweep_thresholds(p: np.ndarray, y: np.ndarray, exposure: np.ndarray,
                     cost: CostModel, n: int = 60) -> List[Dict[str, float]]:
    return [expected_loss(p, y, exposure, t, cost)
            for t in np.linspace(0.01, 0.99, n)]


def optimal_threshold(p: np.ndarray, y: np.ndarray, exposure: np.ndarray,
                      cost: CostModel) -> Dict[str, float]:
    rows = sweep_thresholds(p, y, exposure, cost)
    return max(rows, key=lambda r: r["net_saving"])


@dataclass
class AbstentionPolicy:
    """Three-way decision: pay, review, or reject.

    The band between the two thresholds is where the system declines to decide
    and asks a human. Reporting the width of that band honestly is more useful
    than pretending to a confident answer everywhere.
    """
    review_threshold: float
    reject_threshold: float

    def decide(self, p: float) -> str:
        if p >= self.reject_threshold:
            return "reject"
        if p >= self.review_threshold:
            return "review"
        return "pay"

    def apply(self, p: np.ndarray) -> np.ndarray:
        return np.array([self.decide(float(v)) for v in p])
