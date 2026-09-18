"""FastAPI service.

Two things it does NOT do, deliberately:

- It does not let the language model touch the risk score. The narrative
  endpoint reads a finished evidence packet.
- It does not return a bare probability with no context. Every scored response
  carries the operating point, the decision band, and the quality verdict, so a
  caller cannot mistake an abstention for a clearance.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..decision.policy import AbstentionPolicy
from ..experiments.suite import pick_model, split
from ..pipeline import build_dataset
from ..reporting.evidence import build_packet, narrate
from ..tampering.engine import ALL_CLASSES, TamperSpec

app = FastAPI(
    title="Ledger Forensics",
    description="Multichannel document fraud forensics with error attribution",
    version="1.0.0",
)

STATE: Dict[str, object] = {"model": None, "policy": None, "builder": None}
ARTIFACTS = Path("artifacts")


class ScoreResponse(BaseModel):
    doc_id: str
    risk_score: float
    decision: str
    is_tampered_truth: int
    tamper_class: str
    quality_score: float
    quality_rejected: bool
    quality_reasons: List[str]
    semantic_failures: List[str]
    forensic_scores: Dict[str, float]
    # Optional because localisation is UNDEFINED, not zero, for tamper classes
    # with no local edit region (reprint, full_forgery). Returning 0.0 there
    # would be a wrong answer rather than a missing one.
    localisation: Dict[str, Optional[float]]
    report: str
    report_source: str


def _ensure_model(n: int = 160) -> None:
    if STATE["model"] is not None:
        return
    subs = build_dataset(n, seed=7, tamper_rate=0.5)
    train, _, _ = split(subs, frac=0.7, seed=7)
    X = np.array([s.features() for s in train], dtype=np.float32)
    y = np.array([s.label for s in train], dtype=int)
    STATE["model"] = pick_model(train).fit(X, y)

    clean = X[y == 0]
    p_clean = STATE["model"].predict_proba(clean) if len(clean) else np.array([0.5])
    STATE["policy"] = AbstentionPolicy(
        review_threshold=float(np.percentile(p_clean, 80)),
        reject_threshold=float(np.percentile(p_clean, 97)))

    from ..pipeline import SubmissionBuilder
    STATE["builder"] = SubmissionBuilder(seed=99)


@app.get("/health")
def health() -> Dict[str, object]:
    return {"status": "ok", "model_loaded": STATE["model"] is not None}


@app.get("/classes")
def classes() -> Dict[str, List[str]]:
    return {"tamper_classes": ALL_CLASSES}


@app.get("/results")
def results() -> Dict[str, object]:
    path = ARTIFACTS / "results.json"
    if not path.exists():
        raise HTTPException(404, "no results yet; run `python run.py demo`")
    return json.loads(path.read_text())


@app.post("/score", response_model=ScoreResponse)
def score(tamper_class: Optional[str] = None) -> ScoreResponse:
    """Generate one submission, score it, and return the evidence report.

    ``tamper_class`` is a ground-truth control for demonstration. A production
    endpoint would accept an uploaded image instead; the scoring path below is
    identical either way.
    """
    _ensure_model()
    if tamper_class and tamper_class not in ALL_CLASSES:
        raise HTTPException(400, f"unknown class; try one of {ALL_CLASSES}")

    builder = STATE["builder"]
    spec = TamperSpec(tamper_class) if tamper_class else None
    sub = builder.build(tamper_class=tamper_class, spec=spec)

    X = sub.features().reshape(1, -1)
    p = float(STATE["model"].predict_proba(X)[0])
    decision = STATE["policy"].decide(p)

    packet = build_packet(sub, p, decision)
    report = narrate(packet)

    return ScoreResponse(
        doc_id=sub.doc.doc_id,
        risk_score=round(p, 4),
        decision=decision,
        is_tampered_truth=sub.label,
        tamper_class=sub.tamper_class,
        quality_score=round(sub.quality.score, 4),
        quality_rejected=sub.quality.rejected,
        quality_reasons=list(sub.quality.reasons),
        semantic_failures=[k for k, v in sub.semantics.failures.items() if v],
        forensic_scores={k: round(float(v), 4)
                         for k, v in sub.forensics.scores.items()},
        localisation={k: (None if not np.isfinite(v) else round(float(v), 4))
                      for k, v in sub.localisation.items()},
        report=report["text"],
        report_source=str(report["source"]),
    )


@app.get("/image/{tamper_class}")
def image(tamper_class: str) -> Dict[str, str]:
    """Return a base64 PNG of a generated submission, for the dashboard."""
    from PIL import Image

    _ensure_model()
    cls = None if tamper_class == "none" else tamper_class
    if cls and cls not in ALL_CLASSES:
        raise HTTPException(400, "unknown class")
    sub = STATE["builder"].build(tamper_class=cls,
                                 spec=TamperSpec(cls) if cls else None)
    buf = io.BytesIO()
    Image.fromarray(sub.doc.image).save(buf, format="PNG")
    return {"doc_id": sub.doc.doc_id,
            "png_base64": base64.b64encode(buf.getvalue()).decode()}
