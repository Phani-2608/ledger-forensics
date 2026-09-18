"""The model ladder.

Nothing here is allowed to skip the comparison. A two-channel fused model is
only interesting if it beats the obvious alternatives, so the obvious
alternatives are built and reported alongside it:

    always_benign     predict nothing is fraud (the base rate)
    single_signal     one threshold on the strongest forensic scalar
    rules_only        fire if any semantic check fails
    forensics_only    learned model, pixel features only
    semantics_only    learned model, semantic features only
    fused             learned model, both channels plus quality

If ``fused`` does not beat ``rules_only``, that is the headline and it gets
reported as the headline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..forensics.detectors import FEATURE_NAMES as FORENSIC_FEATURES
from ..semantics.checks import CHECK_NAMES


def feature_names() -> List[str]:
    sem: List[str] = []
    for n in CHECK_NAMES:
        sem += [f"sem_{n}", f"sem_{n}_mag"]
    return list(FORENSIC_FEATURES) + sem + ["quality_score", "quality_flags"]


ALL_FEATURES = feature_names()
N_FORENSIC = len(FORENSIC_FEATURES)
N_SEMANTIC = len(CHECK_NAMES) * 2
FORENSIC_SLICE = slice(0, N_FORENSIC)
SEMANTIC_SLICE = slice(N_FORENSIC, N_FORENSIC + N_SEMANTIC)
QUALITY_SLICE = slice(N_FORENSIC + N_SEMANTIC, len(ALL_FEATURES))


@dataclass
class FittedModel:
    name: str
    predict_proba: object
    channel: str

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)


class AlwaysBenign:
    def fit(self, X, y):
        self.p = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p, dtype=np.float64)


class SingleSignal:
    """One threshold on the single most discriminative forensic scalar."""

    def __init__(self) -> None:
        self.idx = 0
        self.lo = 0.0
        self.hi = 1.0

    def fit(self, X, y):
        best, best_idx = -1.0, 0
        for j in range(N_FORENSIC):
            col = X[:, j]
            if col.std() < 1e-9:
                continue
            # point-biserial correlation with the label
            c = abs(float(np.corrcoef(col, y)[0, 1])) if len(set(y)) > 1 else 0.0
            if np.isfinite(c) and c > best:
                best, best_idx = c, j
        self.idx = best_idx
        col = X[:, self.idx]
        self.lo, self.hi = float(col.min()), float(col.max()) + 1e-9
        self.sign = 1.0
        if len(set(y)) > 1 and float(np.corrcoef(col, y)[0, 1]) < 0:
            self.sign = -1.0
        return self

    def predict_proba(self, X):
        col = X[:, self.idx]
        p = (col - self.lo) / (self.hi - self.lo)
        if self.sign < 0:
            p = 1.0 - p
        return np.clip(p, 0.0, 1.0)


class RulesOnly:
    """Fire if any semantic check failed. No fitting, no thresholds."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        sem = X[:, SEMANTIC_SLICE]
        flags = sem[:, 0::2]
        n = flags.sum(axis=1)
        return np.clip(n / 3.0, 0.0, 1.0)


class LearnedModel:
    def __init__(self, cols: slice | None = None, kind: str = "logreg"):
        self.cols = cols
        self.kind = kind
        self.scaler = StandardScaler()
        if kind == "logreg":
            self.clf = LogisticRegression(max_iter=2000, C=1.0,
                                          class_weight="balanced")
        else:
            self.clf = HistGradientBoostingClassifier(
                max_depth=4, max_iter=180, learning_rate=0.08,
                l2_regularization=1.0, random_state=0)

    def _sub(self, X: np.ndarray) -> np.ndarray:
        return X if self.cols is None else X[:, self.cols]

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(self._sub(X))
        self.clf.fit(Xs, y)
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(self._sub(X))
        return self.clf.predict_proba(Xs)[:, 1]


def build_ladder() -> Dict[str, object]:
    return {
        "always_benign": AlwaysBenign(),
        "single_signal": SingleSignal(),
        "rules_only": RulesOnly(),
        "forensics_only": LearnedModel(FORENSIC_SLICE, "logreg"),
        "semantics_only": LearnedModel(SEMANTIC_SLICE, "logreg"),
        "fused_logreg": LearnedModel(None, "logreg"),
        "fused_gbm": LearnedModel(None, "gbm"),
    }
