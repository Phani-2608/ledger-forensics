"""Dataset caching for staged runs.

The full suite takes longer than a single session, so the built dataset is
cached to disk and each experiment stage reloads it. Images are NOT cached:
they are hundreds of megabytes and nothing downstream of the detectors needs
them. What is cached is exactly the evidence every experiment consumes.

Records duck-type ``Submission`` so the experiment functions are unchanged.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np


def _pack(sub) -> Dict[str, object]:
    return {
        "features": np.asarray(sub.features(), dtype=np.float32),
        "label": int(sub.label),
        "tamper_class": str(sub.tamper_class),
        "exposure": float(sub.exposure),
        "doc_id": sub.doc.doc_id,
        "generator_id": sub.doc.generator_id,
        "has_local_mask": bool(sub.doc.has_local_mask),
        "quality_score": float(sub.quality.score),
        "quality_rejected": bool(sub.quality.rejected),
        "quality_reasons": list(sub.quality.reasons),
        "forensic_scores": {k: float(v) for k, v in sub.forensics.scores.items()},
        "semantic_failures": {k: bool(v) for k, v in sub.semantics.failures.items()},
        "semantic_any": bool(sub.semantics.any_failure),
        "oracle_vector": (list(sub.semantics_oracle.vector())
                          if sub.semantics_oracle is not None else None),
        "oracle_any": (bool(sub.semantics_oracle.any_failure)
                       if sub.semantics_oracle is not None else False),
        "localisation": {k: float(v) for k, v in sub.localisation.items()},
        "ocr_char_errors": int(sub.ocr.char_errors),
    }


def _unpack(d: Dict[str, object]):
    feats = np.asarray(d["features"], dtype=np.float32)
    rec = SimpleNamespace(
        label=d["label"],
        tamper_class=d["tamper_class"],
        exposure=d["exposure"],
        localisation=d["localisation"],
        doc=SimpleNamespace(doc_id=d["doc_id"],
                            generator_id=d["generator_id"],
                            has_local_mask=d["has_local_mask"]),
        quality=SimpleNamespace(score=d["quality_score"],
                                rejected=d["quality_rejected"],
                                reasons=d["quality_reasons"]),
        forensics=SimpleNamespace(scores=d["forensic_scores"]),
        semantics=SimpleNamespace(failures=d["semantic_failures"],
                                  any_failure=d["semantic_any"]),
        semantics_oracle=(
            SimpleNamespace(vector=lambda v=d["oracle_vector"]: v,
                            any_failure=d["oracle_any"])
            if d["oracle_vector"] is not None else None),
        ocr=SimpleNamespace(char_errors=d["ocr_char_errors"]),
    )
    rec.features = lambda f=feats: f
    return rec


def save(subs: Sequence, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump([_pack(s) for s in subs], fh)


def load(path: Path) -> Optional[List]:
    if not Path(path).exists():
        return None
    with open(path, "rb") as fh:
        return [_unpack(d) for d in pickle.load(fh)]


def build_or_load(path: Path, n: int, **kwargs) -> List:
    cached = load(path)
    if cached is not None and len(cached) == n:
        return cached
    from ..pipeline import build_dataset
    subs = build_dataset(n, **kwargs)
    save(subs, path)
    return load(path)
