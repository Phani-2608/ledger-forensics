"""Cross-stage error attribution.

This is the part of the project that is not standard practice, so the method
needs stating rather than asserting.

THE PROBLEM
Most pipelines report a single end-to-end error rate. When a decision is wrong,
nobody can say whether the reading stage misread a digit, the pixel forensics
missed an edit, the semantic checks were structurally blind to it, or the
fusion layer weighed correct evidence badly. Teams then spend money improving
whichever stage is most fashionable rather than whichever stage is at fault.

THE METHOD: counterfactual repair
We have ground truth at every stage, which almost no real pipeline does. So for
each wrong decision we repair one stage at a time, substituting that stage's
oracle output while holding everything else fixed, and re-run the decision.

  If repairing stage S alone flips the decision to correct, S is a SUFFICIENT
  CAUSE of the error.

THE MULTI-CAUSE RULE
Errors frequently have several sufficient causes, so "which stage caused it" is
ill-posed without a tie-break. We record every sufficient cause, and assign
PRIMARY blame to the earliest one in pipeline order. Rationale: fixing an
upstream stage removes the error without any downstream change, so the earliest
sufficient cause is the cheapest true fix.

When no single repair flips the decision but repairing two together does, the
cause is FUSION_ERROR: both channels supplied usable evidence and the combiner
still got it wrong. When not even a full repair flips it, the cause is
NO_EVIDENCE, and that is the honest floor of the system.

ORACLE DEFINITIONS (stated so they can be argued with)
  OCR       : a perfect reader. Semantic checks re-run on error-free values.
  FORENSICS : an ideal local-edit detector. Feature values set to the training
              distribution's 95th percentile when the document really has an
              edited region, 5th percentile when it does not.
  QUALITY   : the gate's decision inverted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..fusion.models import FORENSIC_SLICE, SEMANTIC_SLICE

# Pipeline order. Primary blame goes to the earliest sufficient cause.
CAUSE_ORDER = [
    "QUALITY_FALSE_REJECT",
    "OCR_ERROR",
    "FORENSICS_MISS",
    "FORENSICS_FALSE_ALARM",
    "SEMANTIC_BLIND",
    "FUSION_ERROR",
    "NO_EVIDENCE",
]


@dataclass
class OracleProfile:
    """Forensic feature values an ideal detector would have produced."""
    high: np.ndarray
    low: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "OracleProfile":
        forensic = X[:, FORENSIC_SLICE]
        return cls(high=np.percentile(forensic, 95, axis=0),
                   low=np.percentile(forensic, 5, axis=0))


@dataclass
class Attribution:
    doc_id: str
    label: int
    tamper_class: str
    predicted: int
    probability: float
    primary_cause: str
    sufficient_causes: List[str] = field(default_factory=list)


def _repair_forensics(x: np.ndarray, has_local_edit: bool,
                      oracle: OracleProfile) -> np.ndarray:
    out = x.copy()
    out[FORENSIC_SLICE] = oracle.high if has_local_edit else oracle.low
    return out


def _repair_ocr(x: np.ndarray, oracle_semantic_vector: Sequence[float]
                ) -> np.ndarray:
    out = x.copy()
    out[SEMANTIC_SLICE] = np.asarray(oracle_semantic_vector, dtype=np.float32)
    return out


def attribute(submissions: Sequence, X: np.ndarray, y: np.ndarray,
              predict: Callable[[np.ndarray], np.ndarray],
              threshold: float, oracle: OracleProfile) -> List[Attribution]:
    """Attribute every wrong decision to the stage responsible for it."""
    probs = predict(X)
    preds = (probs >= threshold).astype(int)
    results: List[Attribution] = []

    for i, sub in enumerate(submissions):
        if preds[i] == y[i]:
            continue

        x = X[i]
        target = int(y[i])
        causes: List[str] = []

        # ---- quality gate ------------------------------------------------
        if sub.quality.rejected and target == 0:
            causes.append("QUALITY_FALSE_REJECT")

        # ---- OCR ---------------------------------------------------------
        if sub.semantics_oracle is not None:
            x_ocr = _repair_ocr(x, sub.semantics_oracle.vector())
            if int(predict(x_ocr.reshape(1, -1))[0] >= threshold) == target:
                causes.append("OCR_ERROR")

        # ---- forensics ---------------------------------------------------
        has_edit = bool(sub.doc.has_local_mask)
        x_for = _repair_forensics(x, has_edit, oracle)
        if int(predict(x_for.reshape(1, -1))[0] >= threshold) == target:
            causes.append("FORENSICS_MISS" if target == 1
                          else "FORENSICS_FALSE_ALARM")

        # ---- semantic blindness -----------------------------------------
        # A miss where a perfect reader still produces no semantic failure and
        # there is no pixel trace to find is not a bug in any stage. It is the
        # attack being invisible to both channels by construction.
        if (target == 1 and not has_edit and sub.semantics_oracle is not None
                and not sub.semantics_oracle.any_failure):
            causes.append("SEMANTIC_BLIND")

        # ---- fusion ------------------------------------------------------
        if not causes and sub.semantics_oracle is not None:
            x_both = _repair_forensics(
                _repair_ocr(x, sub.semantics_oracle.vector()), has_edit, oracle)
            if int(predict(x_both.reshape(1, -1))[0] >= threshold) == target:
                causes.append("FUSION_ERROR")

        if not causes:
            causes.append("NO_EVIDENCE")

        primary = sorted(causes, key=lambda c: CAUSE_ORDER.index(c))[0]
        results.append(Attribution(
            doc_id=sub.doc.doc_id, label=target, tamper_class=sub.tamper_class,
            predicted=int(preds[i]), probability=float(probs[i]),
            primary_cause=primary, sufficient_causes=causes))

    return results


def summarise(attributions: Sequence[Attribution], n_total: int
              ) -> Dict[str, object]:
    primary = Counter(a.primary_cause for a in attributions)
    by_class: Dict[str, Counter] = {}
    for a in attributions:
        by_class.setdefault(a.tamper_class, Counter())[a.primary_cause] += 1
    return {
        "n_errors": len(attributions),
        "n_total": int(n_total),
        "error_rate": len(attributions) / max(1, n_total),
        "primary_cause_counts": dict(primary),
        "primary_cause_share": {k: v / max(1, len(attributions))
                                for k, v in primary.items()},
        "by_tamper_class": {k: dict(v) for k, v in by_class.items()},
    }
