"""Text region detection from pixels.

The forensic detectors deliberately do NOT use the generator's ground-truth
layout dictionary. If they did, they would be blind to an inserted line (which
has no entry in that dictionary) and we would be quietly grading ourselves on
an easier problem than the real one.

These boxes are recovered from the image the way a real pipeline would have to.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 12)


def detect_words(img: np.ndarray, min_area: int = 40) -> List[Box]:
    """Word-level boxes via morphological grouping of ink."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bw = binarize(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    joined = cv2.dilate(bw, kernel, iterations=1)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Box] = []
    h, w = gray.shape
    for c in contours:
        x, y, bw_, bh_ = cv2.boundingRect(c)
        if bw_ * bh_ < min_area or bh_ < 6 or bh_ > h * 0.15 or bw_ > w * 0.95:
            continue
        boxes.append((x, y, x + bw_, y + bh_))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def group_rows(boxes: List[Box], tol: float = 0.6) -> List[List[Box]]:
    """Group word boxes into text lines by vertical overlap."""
    rows: List[List[Box]] = []
    for b in sorted(boxes, key=lambda b: b[1]):
        placed = False
        for row in rows:
            ry0 = np.mean([r[1] for r in row])
            ry1 = np.mean([r[3] for r in row])
            height = max(1.0, ry1 - ry0)
            centre = (b[1] + b[3]) / 2.0
            if ry0 - height * tol <= centre <= ry1 + height * tol:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    return [sorted(r, key=lambda b: b[0]) for r in rows if r]
