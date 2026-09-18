"""Semantic and arithmetic consistency checks.

Every fact these checks rely on is defined by our own registry and generator.
We are not asserting anything about real merchants or real tax law, and the
README says so. Inside the closed world the reference data is ground truth, and
that is what makes this channel defensible rather than fabricated.

The checks read OCR OUTPUT, not the document's true values. That is deliberate:
an OCR error can mask a genuine inconsistency or invent a false one, and the
attribution stage needs to be able to catch it doing exactly that.

Several tamper classes are invisible here BY CONSTRUCTION (a copied block
changes no value). That is not a weakness to hide, it is the reason the channel
confusion matrix is worth reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..extraction.ocr import OCRResult
from ..generation.document import Document
from ..world.registry import MERCHANTS, Ledger, tax_rate_for

CHECK_NAMES = [
    "line_arithmetic",
    "subtotal_arithmetic",
    "tax_arithmetic",
    "total_arithmetic",
    "ledger_amount_match",
    "ledger_date_match",
    "ledger_merchant_match",
    "near_duplicate",
    "template_consistency",
]

TOL = 0.02  # dollars; OCR noise and rounding live below this


@dataclass
class SemanticResult:
    failures: Dict[str, bool] = field(default_factory=dict)
    magnitudes: Dict[str, float] = field(default_factory=dict)
    detail: Dict[str, str] = field(default_factory=dict)

    def vector(self) -> List[float]:
        out: List[float] = []
        for name in CHECK_NAMES:
            out.append(1.0 if self.failures.get(name) else 0.0)
            out.append(float(np.clip(self.magnitudes.get(name, 0.0), 0, 1)))
        return out

    @property
    def any_failure(self) -> bool:
        return any(self.failures.values())

    @property
    def n_failures(self) -> int:
        return sum(1 for v in self.failures.values() if v)


def _rel(a: float, b: float) -> float:
    return float(min(1.0, abs(a - b) / max(1.0, abs(b))))


def run_checks(doc: Document, ocr: OCRResult, ledger: Ledger,
               history: Optional[List[Dict[str, str]]] = None) -> SemanticResult:
    r = SemanticResult()
    for name in CHECK_NAMES:
        r.failures[name] = False
        r.magnitudes[name] = 0.0

    n_items = int(ocr.values.get("n_items", "0") or 0)

    # ---- 1. quantity x unit price = line total -------------------------
    worst = 0.0
    for i in range(n_items):
        q = ocr.num(f"item_{i}_qty")
        p = ocr.num(f"item_{i}_price")
        t = ocr.num(f"item_{i}_total")
        if q is None or p is None or t is None:
            continue
        expected = round(q * p, 2)
        if abs(expected - t) > TOL:
            r.failures["line_arithmetic"] = True
            worst = max(worst, _rel(t, expected))
            r.detail["line_arithmetic"] = f"line {i}: {q}x{p} != {t}"
    r.magnitudes["line_arithmetic"] = worst

    # ---- 2. sum of lines = subtotal ------------------------------------
    line_sum = 0.0
    seen = 0
    for i in range(n_items):
        t = ocr.num(f"item_{i}_total")
        if t is not None:
            line_sum += t
            seen += 1
    subtotal = ocr.num("subtotal")
    if subtotal is not None and seen == n_items and n_items > 0:
        if abs(round(line_sum, 2) - subtotal) > TOL:
            r.failures["subtotal_arithmetic"] = True
            r.magnitudes["subtotal_arithmetic"] = _rel(line_sum, subtotal)
            r.detail["subtotal_arithmetic"] = (
                f"lines sum to {line_sum:.2f}, subtotal reads {subtotal:.2f}")

    # ---- 3. subtotal x registry tax rate = tax -------------------------
    rate = tax_rate_for(doc.merchant_id)   # authoritative in this world
    tax = ocr.num("tax")
    if subtotal is not None and tax is not None:
        expected = round(subtotal * rate, 2)
        if abs(expected - tax) > max(TOL, subtotal * 0.001):
            r.failures["tax_arithmetic"] = True
            r.magnitudes["tax_arithmetic"] = _rel(tax, expected)
            r.detail["tax_arithmetic"] = (
                f"tax reads {tax:.2f}, registry rate implies {expected:.2f}")

    # ---- 4. subtotal + tax = total -------------------------------------
    total = ocr.num("total")
    if subtotal is not None and tax is not None and total is not None:
        expected = round(subtotal + tax, 2)
        if abs(expected - total) > TOL:
            r.failures["total_arithmetic"] = True
            r.magnitudes["total_arithmetic"] = _rel(total, expected)
            r.detail["total_arithmetic"] = (
                f"total reads {total:.2f}, components sum to {expected:.2f}")

    # ---- 5-7. the payment ledger ---------------------------------------
    txn = ledger.get(doc.transaction_id)
    if txn is not None:
        if total is not None and abs(total - txn.amount) > TOL:
            r.failures["ledger_amount_match"] = True
            r.magnitudes["ledger_amount_match"] = _rel(total, txn.amount)
            r.detail["ledger_amount_match"] = (
                f"document claims {total:.2f}, ledger recorded {txn.amount:.2f}")

        doc_date = ocr.values.get("date", "")
        if txn.date not in doc_date:
            r.failures["ledger_date_match"] = True
            r.magnitudes["ledger_date_match"] = 1.0
            r.detail["ledger_date_match"] = (
                f"document dated {doc_date}, ledger dated {txn.date}")

        expected_name = MERCHANTS[txn.merchant_id].name
        read_name = ocr.values.get("merchant_name", "")
        if _name_mismatch(read_name, expected_name):
            r.failures["ledger_merchant_match"] = True
            r.magnitudes["ledger_merchant_match"] = 1.0
            r.detail["ledger_merchant_match"] = (
                f"document names {read_name}, ledger names {expected_name}")

    # ---- 8. near-duplicate submissions ---------------------------------
    if history:
        key_now = (ocr.values.get("merchant_name", ""),
                   ocr.values.get("date", ""),
                   ocr.values.get("invoice_id", ""))
        for prev in history[-400:]:
            key_prev = (prev.get("merchant", ""), prev.get("date", ""),
                        prev.get("invoice_id", ""))
            same = sum(1 for a, b in zip(key_now, key_prev) if a == b and a)
            if same >= 2 and prev.get("total") != ocr.values.get("total"):
                r.failures["near_duplicate"] = True
                r.magnitudes["near_duplicate"] = 1.0
                r.detail["near_duplicate"] = (
                    f"matches earlier submission {prev.get('doc_id')} "
                    "with a different total")
                break

    # ---- 9. merchant identity is internally consistent -----------------
    merchant = MERCHANTS[doc.merchant_id]
    read_name = ocr.values.get("merchant_name", "")
    read_addr = ocr.values.get("merchant_address", "")
    if _name_mismatch(read_name, merchant.name) and read_addr == merchant.address:
        # The header names one business while the address and phone below it
        # still belong to another.
        r.failures["template_consistency"] = True
        r.magnitudes["template_consistency"] = 1.0
        r.detail["template_consistency"] = (
            f"header reads {read_name} but address block is {merchant.name}")

    return r


def _name_mismatch(read: str, expected: str) -> bool:
    """Tolerant comparison so a stray OCR character is not a fraud signal."""
    if not read or not expected:
        return False
    a = "".join(ch for ch in read.lower() if ch.isalnum())
    b = "".join(ch for ch in expected.lower() if ch.isalnum())
    if not a or not b:
        return False
    if a == b:
        return False
    # Levenshtein-lite: allow up to 15 percent of characters to differ.
    if abs(len(a) - len(b)) > max(2, int(0.15 * len(b))):
        return True
    diffs = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    return diffs > max(2, int(0.15 * len(b)))
