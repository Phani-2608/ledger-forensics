"""The evidence report.

The language model here is a REPORTER, not an investigator and not a decision
maker. It receives a finished evidence packet and turns it into prose. It never
sees the raw image, never influences the risk score, and never introduces a
number.

Two consequences worth being explicit about:

1. The scientific validity of this project does not depend on the LLM at all.
   Every metric in the README is produced without it. That is by design.
2. Any generated narrative is passed through a numeric verifier. Every number
   appearing in the text must also appear in the evidence packet, or the text
   is discarded and the deterministic template is used instead.

The default path is the template. The LLM path is opt-in via LEDGER_LLM=1.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvidencePacket:
    """Everything the narrator is allowed to know."""
    doc_id: str
    merchant: str
    claimed_total: Optional[float]
    ledger_total: Optional[float]
    risk_score: float
    decision: str
    quality_score: float
    quality_flags: List[str] = field(default_factory=list)
    forensic_scores: Dict[str, float] = field(default_factory=dict)
    semantic_failures: List[str] = field(default_factory=list)
    semantic_detail: Dict[str, str] = field(default_factory=dict)
    suspect_region: Optional[List[int]] = None
    top_signal: Optional[str] = None

    def numbers(self) -> List[float]:
        out: List[float] = [round(self.risk_score, 4), round(self.quality_score, 4)]
        for v in (self.claimed_total, self.ledger_total):
            if v is not None:
                out.append(round(float(v), 2))
        out += [round(float(v), 4) for v in self.forensic_scores.values()]
        if self.suspect_region:
            out += [float(v) for v in self.suspect_region]
        for text in self.semantic_detail.values():
            out += [float(m) for m in re.findall(r"-?\d+\.?\d*", text)]
        return out

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def build_packet(submission, risk_score: float, decision: str) -> EvidencePacket:
    sem = submission.semantics
    failures = [k for k, v in sem.failures.items() if v]
    top = max(submission.forensics.scores.items(), key=lambda kv: kv[1])[0] \
        if submission.forensics.scores else None

    region = None
    mask = submission.doc.tamper_mask
    if submission.doc.has_local_mask and mask is not None:
        import numpy as np
        ys, xs = np.where(mask > 0)
        if len(xs):
            region = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    ocr = submission.ocr
    return EvidencePacket(
        doc_id=submission.doc.doc_id,
        merchant=ocr.values.get("merchant_name", "unknown"),
        claimed_total=ocr.num("total"),
        ledger_total=None,
        risk_score=float(risk_score),
        decision=decision,
        quality_score=float(submission.quality.score),
        quality_flags=list(submission.quality.reasons),
        forensic_scores={k: float(v) for k, v in submission.forensics.scores.items()
                         if not k.endswith("_fld") and k != "n_words"},
        semantic_failures=failures,
        semantic_detail={k: v for k, v in sem.detail.items()},
        suspect_region=region,
        top_signal=top,
    )


# ---------------------------------------------------------------------------
def render_template(packet: EvidencePacket) -> str:
    """Deterministic narrative. Always available, never wrong."""
    lines: List[str] = []
    lines.append(f"Document {packet.doc_id} submitted by {packet.merchant}.")
    if packet.claimed_total is not None:
        lines.append(f"Amount claimed: {packet.claimed_total:.2f}.")
    lines.append(
        f"Risk score {packet.risk_score:.3f}. Recommended action: {packet.decision}."
    )

    lines.append(
        f"Image quality scored {packet.quality_score:.2f}"
        + (f" with flags: {', '.join(packet.quality_flags)}."
           if packet.quality_flags else " with no quality flags.")
    )

    if packet.semantic_failures:
        lines.append("Consistency checks that failed:")
        for name in packet.semantic_failures:
            detail = packet.semantic_detail.get(name)
            lines.append(f"  - {name}" + (f": {detail}" if detail else ""))
    else:
        lines.append("All consistency checks passed.")

    if packet.forensic_scores:
        ranked = sorted(packet.forensic_scores.items(), key=lambda kv: -kv[1])[:3]
        lines.append("Strongest pixel-level signals: " + ", ".join(
            f"{k} {v:.3f}" for k, v in ranked) + ".")

    if packet.suspect_region:
        x0, y0, x1, y1 = packet.suspect_region
        lines.append(
            f"Region of interest spans pixels ({x0}, {y0}) to ({x1}, {y1}).")

    lines.append(
        "This report narrates recorded evidence only. It does not assign the "
        "risk score and it does not make the final decision.")
    return "\n".join(lines)


NUM_RE = re.compile(r"-?\d+\.?\d*")


def verify_numbers(text: str, packet: EvidencePacket,
                   tolerance: float = 0.01) -> tuple:
    """Every number in the text must be present in the packet."""
    allowed = packet.numbers()
    unverified: List[str] = []
    for token in NUM_RE.findall(text):
        try:
            val = float(token)
        except ValueError:
            continue
        if not any(abs(val - a) <= tolerance for a in allowed):
            unverified.append(token)
    return (len(unverified) == 0), unverified


def narrate(packet: EvidencePacket, use_llm: Optional[bool] = None) -> Dict[str, object]:
    """Produce the report. Falls back to the template on any failure."""
    if use_llm is None:
        use_llm = os.environ.get("LEDGER_LLM", "0") == "1"

    template = render_template(packet)
    if not use_llm:
        return {"text": template, "source": "template", "verified": True,
                "unverified_numbers": []}

    try:
        text = _call_llm(packet)
        ok, bad = verify_numbers(text, packet)
        if ok:
            return {"text": text, "source": "llm", "verified": True,
                    "unverified_numbers": []}
        return {"text": template, "source": "template_fallback",
                "verified": False, "unverified_numbers": bad,
                "reason": "generated text contained numbers absent from the "
                          "evidence packet"}
    except Exception as exc:
        return {"text": template, "source": "template_fallback",
                "verified": True, "unverified_numbers": [],
                "reason": f"llm unavailable: {exc}"}


def _call_llm(packet: EvidencePacket) -> str:  # pragma: no cover - optional
    """Optional. Requires OPENAI_API_KEY and the openai package."""
    from openai import OpenAI

    client = OpenAI()
    prompt = (
        "You are writing a short factual case note for a fraud analyst. "
        "Use ONLY the facts in this JSON. Do not introduce any number that is "
        "not present in it. Do not speculate about intent. Do not state a "
        "conclusion about whether fraud occurred; the decision field already "
        "records the recommended action.\n\n" + packet.to_json()
    )
    resp = client.chat.completions.create(
        model=os.environ.get("LEDGER_LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=400)
    return resp.choices[0].message.content.strip()
