"""Quality gate.

Decide whether an image is legible enough to judge at all, before spending
effort downstream. The gate is deliberately measured rather than tuned by eye:
the experiment maps gate threshold to downstream decision error so the
rejection rate can be chosen against a cost, not a vibe.

A false reject is its own failure mode and the attribution stage counts it
separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass
class QualityReport:
    sharpness: float       # variance of laplacian, higher is sharper
    contrast: float        # std of luminance
    overexposed: float     # fraction of near-white pixels
    underexposed: float    # fraction of near-black pixels
    resolution: int        # min(h, w)
    skew_deg: float        # estimated text-line skew
    score: float           # 0..1 composite
    rejected: bool
    reasons: tuple

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d


def _skew(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLines(edges, 1, np.pi / 360, threshold=140)
    if lines is None:
        return 0.0
    angles = []
    for rho_theta in lines[:60]:
        theta = float(rho_theta[0][1])
        deg = np.degrees(theta) - 90.0
        if -25 < deg < 25:
            angles.append(deg)
    return float(np.median(angles)) if angles else 0.0


def assess(img: np.ndarray, threshold: float = 0.45) -> QualityReport:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    over = float((gray > 250).mean())
    under = float((gray < 12).mean())
    res = int(min(gray.shape))
    skew = _skew(gray)

    # Each sub-score is 0..1. Bounds chosen from the clean-render distribution,
    # documented in docs/EXPERIMENTS.md rather than hand-tuned per document.
    s_sharp = float(np.clip(sharp / 220.0, 0, 1))
    s_contrast = float(np.clip(contrast / 55.0, 0, 1))
    s_expo = float(np.clip(1.0 - (over * 2.5 + under * 4.0), 0, 1))
    s_res = float(np.clip(res / 480.0, 0, 1))
    s_skew = float(np.clip(1.0 - abs(skew) / 12.0, 0, 1))

    score = float(np.average([s_sharp, s_contrast, s_expo, s_res, s_skew],
                             weights=[0.34, 0.18, 0.18, 0.16, 0.14]))

    reasons = []
    if s_sharp < 0.30:
        reasons.append("blurred")
    if s_expo < 0.55:
        reasons.append("exposure")
    if s_res < 0.55:
        reasons.append("low_resolution")
    if s_skew < 0.55:
        reasons.append("skew")

    return QualityReport(sharp, contrast, over, under, res, skew, score,
                         score < threshold, tuple(reasons))
