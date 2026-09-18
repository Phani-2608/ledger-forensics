"""External validity harness.

Everything else in this repo is measured on documents this repo generated. That
is internal validity only, and no amount of internal rigour turns it into
evidence that the detectors work on real receipts.

This module is the honest bridge. It takes a small set of REAL photographed
receipts, plus hand-edited copies of them, and runs the same detectors against
them. A few dozen documents will not give tight confidence intervals. That is
fine. The point is not a precise number, it is whether the signal survives
contact with images we did not render.

USAGE

    external_data/
      authentic/     real receipt photos, any filename
      tampered/      hand-edited copies, same filenames as their originals
      masks/         optional, white on black PNG marking the edited region

    python -m ledger_forensics.experiments.external external_data

If ``masks/`` is absent, localisation is skipped and only the detection
comparison is reported.

If you have no real receipts, this reports that and exits without pretending.
The README states the external-validity result as unmeasured in that case,
which is the truthful thing to say.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from ..forensics import detectors, localize

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _load(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _peak(result: detectors.ForensicResult) -> float:
    return max(result.scores.get(f"{n}_fld", 0.0)
               for n in detectors.DETECTOR_NAMES)


def run(root: str | Path) -> Dict[str, object]:
    root = Path(root)
    auth_dir, tamp_dir = root / "authentic", root / "tampered"
    mask_dir = root / "masks"

    if not auth_dir.is_dir() or not tamp_dir.is_dir():
        return {"status": "not_available",
                "reason": f"expected {auth_dir} and {tamp_dir}",
                "note": "external validity is UNMEASURED; results in this "
                        "repository are internal validity only"}

    auth_files = sorted(p for p in auth_dir.iterdir()
                        if p.suffix.lower() in IMAGE_SUFFIXES)
    tamp_files = sorted(p for p in tamp_dir.iterdir()
                        if p.suffix.lower() in IMAGE_SUFFIXES)
    if not auth_files or not tamp_files:
        return {"status": "not_available", "reason": "no images found",
                "note": "external validity is UNMEASURED"}

    auth_scores, tamp_scores, ious = [], [], []
    per_file: List[Dict[str, object]] = []

    for p in auth_files:
        r = detectors.run_all(_load(p))
        s = _peak(r)
        auth_scores.append(s)
        per_file.append({"file": p.name, "label": "authentic", "peak": s})

    for p in tamp_files:
        img = _load(p)
        r = detectors.run_all(img)
        s = _peak(r)
        tamp_scores.append(s)
        row: Dict[str, object] = {"file": p.name, "label": "tampered",
                                  "peak": s}

        mask_path = mask_dir / p.name if mask_dir.is_dir() else None
        if mask_path is not None and mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
            if mask.shape == img.shape[:2]:
                m = localize.localisation_metrics(r, (mask > 127).astype(np.uint8))
                row["iou"] = m["iou"]
                ious.append(m["iou"])
        per_file.append(row)

    threshold = float(np.percentile(auth_scores, 90))
    detection = float(np.mean([s > threshold for s in tamp_scores]))

    return {
        "status": "measured",
        "n_authentic": len(auth_scores),
        "n_tampered": len(tamp_scores),
        "authentic_p90_threshold": threshold,
        "authentic_mean_peak": float(np.mean(auth_scores)),
        "tampered_mean_peak": float(np.mean(tamp_scores)),
        "detection_rate_at_10pct_false_alarm": detection,
        "mean_iou": float(np.mean(ious)) if ious else None,
        "per_file": per_file,
        "caveat": "small sample; treat as a smoke test of transfer, not as a "
                  "performance estimate",
    }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "external_data"
    print(json.dumps(run(target), indent=2))
