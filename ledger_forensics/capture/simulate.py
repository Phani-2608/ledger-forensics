"""Capture and degradation simulation.

A submitted document is never a clean render. It is a phone photo: skewed,
unevenly lit, resized by the upload path and JPEG compressed more than once.

This stage runs AFTER tampering, so the degradation applies to the tampered
image the way it would in reality. Every parameter is recorded so the
degradation-curve experiment can sweep one axis at a time.
"""

from __future__ import annotations

import io
import random
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class CaptureParams:
    jpeg_quality: int = 90
    blur_sigma: float = 0.0
    scale: float = 1.0            # 1.0 = no downsample
    perspective: float = 0.0      # 0..1, fraction of width corners move
    illumination: float = 0.0     # 0..1 strength of lighting gradient
    sensor_noise: float = 0.0     # gaussian sigma

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


CLEAN = CaptureParams(jpeg_quality=95)


def sample_params(rng: random.Random) -> CaptureParams:
    return CaptureParams(
        jpeg_quality=rng.choice([70, 78, 85, 90, 95]),
        blur_sigma=rng.choice([0.0, 0.0, 0.4, 0.8, 1.2]),
        scale=rng.choice([1.0, 1.0, 0.85, 0.7]),
        perspective=rng.choice([0.0, 0.01, 0.02, 0.035]),
        illumination=rng.uniform(0.0, 0.6),
        sensor_noise=rng.choice([0.0, 1.0, 2.0, 3.5]),
    )


def warp_boxes(layout: Dict[str, Tuple[int, int, int, int]], M: np.ndarray
               ) -> Dict[str, Tuple[int, int, int, int]]:
    """Carry field boxes through the same homography as the image."""
    out = {}
    for name, (x0, y0, x1, y1) in layout.items():
        pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
        w = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        out[name] = (int(w[:, 0].min()), int(w[:, 1].min()),
                     int(w[:, 0].max()), int(w[:, 1].max()))
    return out


def photograph(img: np.ndarray, layout: Dict[str, Tuple[int, int, int, int]],
               params: CaptureParams, rng: Optional[random.Random] = None
               ) -> Tuple[np.ndarray, Dict[str, Tuple[int, int, int, int]]]:
    """Produce the authentic photograph of a document.

    ORDER MATTERS. This runs BEFORE tampering, not after.

    A fraudster does not edit a pristine render. They receive an already
    photographed, already JPEG-compressed file and edit that. If we tampered
    first and compressed afterwards, the edited region and the untouched
    background would share an identical compression history and Error Level
    Analysis would be structurally incapable of seeing anything. Getting this
    order wrong quietly turns the whole forensic channel into noise.
    """
    rng = rng or random.Random(0)
    out = img.copy()
    M = None

    if params.perspective > 0:
        h, w = out.shape[:2]
        d = params.perspective * w
        j = lambda: rng.uniform(-d, d)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[j(), j()], [w + j(), j()],
                          [w + j(), h + j()], [j(), h + j()]])
        M = cv2.getPerspectiveTransform(src, dst)
        border = int(np.percentile(out.reshape(-1, 3), 90, axis=0).mean())
        out = cv2.warpPerspective(out, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(border, border, border))

    out = _photometric(out, params, rng)
    out = resave(out, int(params.jpeg_quality))
    new_layout = warp_boxes(layout, M) if M is not None else dict(layout)
    return out, new_layout


def resave(img: np.ndarray, quality: int) -> np.ndarray:
    """One more JPEG generation, as any save or upload would add."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def _photometric(out: np.ndarray, params: CaptureParams,
                 rng: random.Random) -> np.ndarray:
    if params.illumination > 0:
        h, w = out.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        r = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
        gain = 1.0 - params.illumination * 0.45 * (r / (r.max() + 1e-8))
        out = np.clip(out.astype(np.float32) * gain[:, :, None], 0, 255)
        out = out.astype(np.uint8)

    if params.blur_sigma > 0:
        k = int(max(3, round(params.blur_sigma * 4) | 1))
        out = cv2.GaussianBlur(out, (k, k), params.blur_sigma)

    if params.scale != 1.0:
        h, w = out.shape[:2]
        nh, nw = max(64, int(h * params.scale)), max(64, int(w * params.scale))
        out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_AREA)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

    if params.sensor_noise > 0:
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, params.sensor_noise, size=out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out
