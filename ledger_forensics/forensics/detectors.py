"""Deterministic forensic detectors.

Every detector here is an ALGORITHM, not a trained model. None of them has
parameters fitted on our data. That distinction matters for the leave-one-
attack-out experiment: holding out an attack class is only a meaningful test
for the learned fusion layer, because these six have nothing to learn.

PRNU / camera sensor fingerprinting is deliberately absent. It needs many
images from one known physical device to build a reference fingerprint, so on
synthetically rendered documents it is not merely hard, it is inapplicable.
Claiming it would be the least defensible thing in the repo.

Each detector returns (scalar_score, heatmap) where heatmap is float32 HxW in
0..1 and higher means more suspicious.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from .textboxes import detect_words, group_rows

Heatmap = np.ndarray


def _norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(np.percentile(x, 1)), float(np.percentile(x, 99.5))
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def concentration(heat: np.ndarray, top_frac: float = 0.01) -> float:
    """How concentrated is the suspicion?

    Normalising each heatmap to 0..1 destroys absolute scale, so a scalar taken
    from raw magnitudes separates nothing (clean documents get stretched to full
    range too). What actually distinguishes a tampered document is that its
    suspicion is CONCENTRATED in one place rather than spread thinly.

    This returns the ratio of mean heat in the hottest ``top_frac`` of pixels to
    the mean over the whole map, squashed into 0..1. It is scale free by
    construction.
    """
    if heat.size == 0 or not np.isfinite(heat).any():
        return 0.0
    flat = heat.reshape(-1)
    # Measure over the SUPPORT, not the whole image. Several heatmaps are
    # masked to ink, so most pixels are exactly zero; including them makes the
    # denominator tiny and the ratio saturates for every document alike.
    support = flat[flat > 1e-4]
    if support.size < 32:
        return 0.0
    baseline = float(np.median(support))
    if baseline <= 1e-4:
        baseline = float(support.mean())
    if baseline <= 1e-6:
        return 0.0
    k = max(1, int(support.size * top_frac))
    top = float(np.partition(support, -k)[-k:].mean())
    ratio = top / (baseline + 1e-6)
    return float(np.clip((ratio - 1.0) / 6.0, 0.0, 1.0))


def field_outlier(heat: np.ndarray, boxes: List[Tuple[int, int, int, int]]
                  ) -> float:
    """How much does the most suspicious text field stand out from the rest?

    A global statistic over the whole heatmap is the wrong scalar for a local
    edit: changing one total is a few hundred pixels out of three hundred
    thousand, and any image-wide summary drowns it.

    The question that actually matters is comparative. Every field on an
    authentic document was printed in one pass, so their forensic responses
    should look alike. We take the mean heat per detected word box, then report
    the robust z-score of the largest one against the others.
    """
    if heat.size == 0 or len(boxes) < 6:
        return 0.0
    vals = []
    for (x0, y0, x1, y1) in boxes:
        patch = heat[max(0, y0):y1, max(0, x0):x1]
        if patch.size >= 20:
            vals.append(float(patch.mean()))
    if len(vals) < 6:
        return 0.0
    arr = np.array(vals, dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826
    if mad < 1e-5:
        mad = float(arr.std()) + 1e-5
    z = (arr.max() - med) / (mad + 1e-6)
    return float(np.clip(z / 12.0, 0.0, 1.0))


def _blockwise(gray: np.ndarray, fn: Callable[[np.ndarray], float],
               block: int = 16) -> np.ndarray:
    h, w = gray.shape
    bh, bw = max(1, h // block), max(1, w // block)
    out = np.zeros((bh, bw), dtype=np.float32)
    for i in range(bh):
        for j in range(bw):
            patch = gray[i * block:(i + 1) * block, j * block:(j + 1) * block]
            out[i, j] = fn(patch) if patch.size else 0.0
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
def ela(img: np.ndarray, quality: int = 90) -> Tuple[float, Heatmap]:
    """Error Level Analysis.

    A region saved a different number of times than its surroundings responds
    differently to one more compression cycle.
    """
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recomp = np.array(Image.open(buf).convert("RGB")).astype(np.float32)
    diff = np.abs(img.astype(np.float32) - recomp).mean(axis=2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    ink = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) < 160
    ink = cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8))
    # Only ink regions carry usable ELA signal; blank paper is dominated by
    # sensor noise and swamps the statistic.
    heat = _norm(diff) * ink.astype(np.float32)
    if ink.sum() < 50:
        return 0.0, heat
    return concentration(heat), heat


def noise_inconsistency(img: np.ndarray) -> Tuple[float, Heatmap]:
    """Local noise level should be uniform across an authentic capture."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    residual = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    local = _blockwise(np.abs(residual), lambda p: float(np.median(p)), block=16)
    med = float(np.median(local))
    dev = np.abs(local - med) / (med + 1e-3)
    heat = _norm(dev)
    return concentration(heat), heat


def resampling(img: np.ndarray) -> Tuple[float, Heatmap]:
    """Pasted or rescaled content carries different high-frequency energy."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    energy = _blockwise(lap, lambda p: float(p.mean()), block=16)
    ink = _blockwise((gray < 160).astype(np.float32),
                     lambda p: float(p.mean()), block=16)
    # normalise energy by how much ink is present, so blank areas do not fire
    ratio = energy / (ink * 60.0 + 5.0)
    med = float(np.median(ratio[ink > 0.02])) if (ink > 0.02).any() else 0.0
    dev = np.abs(ratio - med) * (ink > 0.02)
    heat = _norm(dev)
    return concentration(heat), heat


def copy_move(img: np.ndarray, corr_threshold: float = 0.93,
              min_cluster: int = 2) -> Tuple[float, Heatmap]:
    """Word-template self-matching with displacement clustering.

    Two approaches were tried and rejected before this one, and the reasons are
    worth keeping:

    - ORB keypoint self-matching fires everywhere on an authentic receipt,
      because printed text repeats the same glyph shapes constantly.
    - Exact block hashing finds nothing, because a pasted region rarely lands on
      the JPEG 8x8 grid, so recompression gives it different pixel values from
      its own source.

    What survives both problems: correlate each detected word against the whole
    image, and require that SEVERAL words share the SAME displacement vector. A
    coincidentally repeated word gives one match at an arbitrary offset. A
    copied region gives a cluster of words at one consistent offset.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    heat = np.zeros((h, w), dtype=np.float32)

    boxes = [b for b in detect_words(img)
             if 14 <= (b[2] - b[0]) < w * 0.5 and (b[3] - b[1]) >= 8]
    if len(boxes) < 3:
        return 0.0, heat

    clusters: Dict[Tuple[int, int], List[Tuple[int, int, int, int, int, int]]] = {}
    for (x0, y0, x1, y1) in boxes:
        tpl = gray[y0:y1, x0:x1]
        if tpl.shape[0] >= h or tpl.shape[1] >= w or tpl.size < 100:
            continue
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        sy0, sy1 = max(0, y0 - 6), min(res.shape[0], y0 + 7)
        sx0, sx1 = max(0, x0 - 6), min(res.shape[1], x0 + 7)
        res[sy0:sy1, sx0:sx1] = -1.0          # suppress the self-match
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv < corr_threshold:
            continue
        dx, dy = maxloc[0] - x0, maxloc[1] - y0
        if abs(dx) + abs(dy) < 20:
            continue
        key = (int(round(dx / 3.0)), int(round(dy / 3.0)))
        clusters.setdefault(key, []).append(
            (x0, y0, x1, y1, maxloc[0], maxloc[1]))

    best = 0
    for pts in clusters.values():
        if len(pts) < min_cluster:
            continue
        best = max(best, len(pts))
        for x0, y0, x1, y1, mx, my in pts:
            bw, bh = x1 - x0, y1 - y0
            heat[y0:y1, x0:x1] = 1.0
            heat[my:min(h, my + bh), mx:min(w, mx + bw)] = 1.0

    if best == 0:
        return 0.0, heat
    heat = cv2.GaussianBlur(heat, (9, 9), 0)
    return float(np.clip(best / 6.0, 0, 1)), heat


def glyph_consistency(img: np.ndarray) -> Tuple[float, Heatmap]:
    """Ink statistics should be consistent across a document printed once."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    heat = np.zeros((h, w), dtype=np.float32)
    boxes = detect_words(img)
    if len(boxes) < 8:
        return 0.0, heat

    feats, keep = [], []
    for (x0, y0, x1, y1) in boxes:
        patch = gray[y0:y1, x0:x1]
        if patch.size < 30:
            continue
        dark = patch < max(90, int(np.percentile(patch, 25)))
        if dark.sum() < 6:
            continue
        density = float(dark.mean())
        ink_val = float(patch[dark].mean())
        dist = cv2.distanceTransform(dark.astype(np.uint8), cv2.DIST_L2, 3)
        stroke = float(dist[dark].mean() * 2.0)
        height = float(y1 - y0)
        feats.append([density, ink_val / 255.0, stroke, height / 30.0])
        keep.append((x0, y0, x1, y1))

    if len(feats) < 8:
        return 0.0, heat

    F = np.array(feats, dtype=np.float32)
    med = np.median(F, axis=0)
    mad = np.median(np.abs(F - med), axis=0) + 1e-3
    z = np.abs(F - med) / (mad * 1.4826)
    per_box = z.max(axis=1)

    for (x0, y0, x1, y1), s in zip(keep, per_box):
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1],
                                        float(np.clip(s / 8.0, 0, 1)))
    heat = cv2.GaussianBlur(heat, (9, 9), 0)
    return concentration(heat), heat


def baseline_geometry(img: np.ndarray) -> Tuple[float, Heatmap]:
    """Words on one printed line share a baseline. Pasted text often does not."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    heat = np.zeros((h, w), dtype=np.float32)
    rows = group_rows(detect_words(img))
    residuals = []

    for row in rows:
        if len(row) < 3:
            continue
        xs = np.array([(b[0] + b[2]) / 2.0 for b in row])
        ys = np.array([float(b[3]) for b in row])          # baseline proxy
        hs = np.array([float(b[3] - b[1]) for b in row])
        try:
            coef = np.polyfit(xs, ys, 1)
        except (np.linalg.LinAlgError, ValueError):
            continue
        pred = np.polyval(coef, xs)
        res = np.abs(ys - pred) / (np.median(hs) + 1e-3)
        residuals.extend(res.tolist())
        for b, r in zip(row, res):
            if r > 0.12:
                heat[b[1]:b[3], b[0]:b[2]] = float(np.clip(r / 0.5, 0, 1))

    if not residuals or heat.max() <= 0:
        return 0.0, heat
    heat = cv2.GaussianBlur(heat, (9, 9), 0)
    return concentration(heat), heat


@dataclass
class ForensicResult:
    scores: Dict[str, float]
    heatmaps: Dict[str, Heatmap]

    def vector(self, names: List[str]) -> np.ndarray:
        return np.array([self.scores.get(n, 0.0) for n in names],
                        dtype=np.float32)


DETECTORS: Dict[str, Callable[[np.ndarray], Tuple[float, Heatmap]]] = {
    "ela": ela,
    "noise": noise_inconsistency,
    "resampling": resampling,
    "copy_move": copy_move,
    "glyph": glyph_consistency,
    "geometry": baseline_geometry,
}

DETECTOR_NAMES = list(DETECTORS)


def run_all(img: np.ndarray) -> ForensicResult:
    """Run every detector and derive two scalars per detector.

    ``<name>``      concentration of suspicion across the heatmap support
    ``<name>_fld``  how far the most suspicious text field sits from the rest

    Both are kept because they answer different questions and the fusion layer
    is allowed to decide which matters for which attack.
    """
    boxes = detect_words(img)
    scores, heats = {}, {}
    for name, fn in DETECTORS.items():
        try:
            s, hm = fn(img)
        except Exception:  # a detector failing must not kill the pipeline
            s, hm = 0.0, np.zeros(img.shape[:2], dtype=np.float32)
        scores[name] = float(s)
        scores[f"{name}_fld"] = field_outlier(hm, boxes)
        heats[name] = hm
    scores["n_words"] = float(len(boxes)) / 100.0
    return ForensicResult(scores, heats)


FEATURE_NAMES: List[str] = (
    DETECTOR_NAMES
    + [f"{n}_fld" for n in DETECTOR_NAMES]
    + ["n_words"]
)
