"""Localisation: where was the document edited?

Saying "this document was edited" is a binary classifier. Saying "this region
was edited" is a much stronger and much more falsifiable claim, and because the
tampering engine hands us an exact pixel mask we can score it properly.

Localisation is only defined for classes that have a local edit. Reprints and
full forgeries have no edited region, so they are excluded from these metrics
and reported separately rather than counted as misses.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .detectors import ForensicResult

# Weights reflect which detectors carry spatial information worth trusting.
FUSE_WEIGHTS = {
    "ela": 1.0,
    "noise": 0.7,
    "resampling": 0.8,
    "copy_move": 1.2,
    "glyph": 1.1,
    "geometry": 0.9,
}


def fuse_heatmaps(result: ForensicResult) -> np.ndarray:
    maps = [(FUSE_WEIGHTS.get(k, 1.0), v) for k, v in result.heatmaps.items()
            if v is not None and v.size]
    if not maps:
        return np.zeros((1, 1), dtype=np.float32)
    shape = maps[0][1].shape
    acc = np.zeros(shape, dtype=np.float32)
    wsum = 0.0
    for w, m in maps:
        if m.shape != shape:
            m = cv2.resize(m, (shape[1], shape[0]))
        acc += w * m
        wsum += w
    acc /= max(wsum, 1e-6)
    return cv2.GaussianBlur(acc, (11, 11), 0)


def binarise(heat: np.ndarray, percentile: float = 98.5) -> np.ndarray:
    if heat.max() <= 0:
        return np.zeros_like(heat, dtype=np.uint8)
    thr = float(np.percentile(heat, percentile))
    return (heat >= max(thr, 1e-6)).astype(np.uint8)


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    p = pred.astype(bool)
    t = (truth > 0)
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(p, t).sum() / union)


def pointing_hit(heat: np.ndarray, truth: np.ndarray) -> bool:
    """Does the single most suspicious pixel land inside the true edit?"""
    if heat.size == 0 or truth is None or not (truth > 0).any():
        return False
    idx = int(np.argmax(heat))
    y, x = np.unravel_index(idx, heat.shape)
    return bool(truth[y, x] > 0)


def localisation_metrics(result: ForensicResult, mask: Optional[np.ndarray],
                         percentile: float = 98.5) -> Dict[str, float]:
    heat = fuse_heatmaps(result)
    if mask is None or not (mask > 0).any():
        return {"iou": float("nan"), "pointing": float("nan"),
                "defined": 0.0}
    if heat.shape != mask.shape:
        heat = cv2.resize(heat, (mask.shape[1], mask.shape[0]))
    pred = binarise(heat, percentile)
    return {
        "iou": iou(pred, mask),
        "pointing": float(pointing_hit(heat, mask)),
        "defined": 1.0,
    }
