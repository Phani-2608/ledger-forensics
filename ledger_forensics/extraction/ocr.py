"""OCR and layout extraction.

HONEST LIMITATION, STATED UP FRONT
----------------------------------
The default engine is ``SimulatedOCR``. It reads the values that are actually
rendered on the current image (so it correctly reads a tampered total, not the
original one) and then injects character-level errors at a controlled rate that
scales with local image quality.

This is a deliberate choice, not a shortcut, and it cuts both ways:

  What it costs us: we are not measuring a real OCR engine's accuracy, and any
  claim about absolute OCR performance would be invalid.

  What it buys us: OCR error rate becomes a DIAL. The error-attribution study
  needs to answer "how much decision error is caused by the reading stage
  rather than the detecting stage", and being able to sweep OCR error from 0 to
  20 percent while holding everything else fixed answers that far more cleanly
  than a fixed real engine would.

``TesseractOCR`` is a drop-in adapter for anyone who wants to swap in a real
engine. Results in the README are reported with the simulated engine and
labelled as such.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..generation.document import Document

CONFUSIONS = {
    "0": "8", "1": "7", "3": "8", "5": "6", "6": "5", "8": "3", "9": "0",
    "2": "7", "4": "1", "7": "1",
    "O": "0", "l": "1", "S": "5", "B": "8", "-": "_",
}


@dataclass
class OCRResult:
    """What the reader believes the document says."""
    values: Dict[str, str] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    char_errors: int = 0
    fields_read: int = 0

    def num(self, key: str) -> Optional[float]:
        raw = self.values.get(key)
        if raw is None:
            return None
        cleaned = re.sub(r"[^0-9.\-]", "", raw)
        try:
            return float(cleaned)
        except ValueError:
            return None

    @property
    def error_rate(self) -> float:
        return self.char_errors / max(1, self.fields_read)


class OCREngine:
    def read(self, doc: Document, quality_score: float = 1.0) -> OCRResult:
        raise NotImplementedError


class SimulatedOCR(OCREngine):
    """Reads rendered values, then corrupts characters at a controlled rate."""

    def __init__(self, base_error_rate: float = 0.02, seed: int = 0):
        self.base_error_rate = base_error_rate
        self.rng = random.Random(seed)

    def _corrupt(self, s: str, rate: float) -> Tuple[str, int]:
        if rate <= 0:
            return s, 0
        out, errs = [], 0
        for ch in s:
            if self.rng.random() < rate and ch in CONFUSIONS:
                out.append(CONFUSIONS[ch])
                errs += 1
            else:
                out.append(ch)
        return "".join(out), errs

    def read(self, doc: Document, quality_score: float = 1.0) -> OCRResult:
        f = doc.rendered_fields
        # Worse images are read worse. Clean image -> base rate, poor image ->
        # up to 6x base rate.
        rate = self.base_error_rate * (1.0 + 5.0 * (1.0 - float(quality_score)))

        raw: Dict[str, str] = {
            "merchant_name": f.merchant_name,
            "merchant_address": f.merchant_address,
            "merchant_phone": f.merchant_phone,
            "invoice_id": f.invoice_id,
            "date": f.date,
            "subtotal": f"{f.subtotal:.2f}",
            "tax": f"{f.tax:.2f}",
            "tax_rate": f"{f.tax_rate:.4f}",
            "total": f"{f.total:.2f}",
            "card_last4": f.card_last4,
            "auth_code": f.auth_code,
        }
        for i, li in enumerate(f.line_items):
            raw[f"item_{i}_desc"] = li.description
            raw[f"item_{i}_qty"] = str(li.quantity)
            raw[f"item_{i}_price"] = f"{li.unit_price:.2f}"
            raw[f"item_{i}_total"] = f"{li.line_total:.2f}"
        raw["n_items"] = str(len(f.line_items))

        result = OCRResult()
        total_errs = 0
        for key, val in raw.items():
            # tax_rate and n_items are structural, read from layout not glyphs
            r = 0.0 if key in ("tax_rate", "n_items") else rate
            got, errs = self._corrupt(val, r)
            result.values[key] = got
            result.confidence[key] = float(np.clip(
                1.0 - r * 6.0 - (0.25 if errs else 0.0), 0.05, 0.99))
            total_errs += errs
        result.char_errors = total_errs
        result.fields_read = len(raw)
        return result


class TesseractOCR(OCREngine):
    """Adapter for a real engine. Requires pytesseract and the tesseract binary."""

    def __init__(self) -> None:
        try:
            import pytesseract  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError(
                "TesseractOCR requires `pip install pytesseract` and the "
                "tesseract binary. The default SimulatedOCR needs neither. "
                "See docs/SCOPE.md for why results are reported with the "
                "simulated engine."
            ) from exc

    def read(self, doc: Document, quality_score: float = 1.0
             ) -> OCRResult:  # pragma: no cover - optional path
        import pytesseract
        from PIL import Image

        result = OCRResult()
        for name, (x0, y0, x1, y1) in doc.layout.items():
            if name.startswith("_"):
                continue
            crop = doc.image[max(0, y0):y1, max(0, x0):x1]
            if crop.size == 0:
                continue
            txt = pytesseract.image_to_string(
                Image.fromarray(crop), config="--psm 7").strip()
            result.values[name] = txt
            result.confidence[name] = 0.8
            result.fields_read += 1
        return result
