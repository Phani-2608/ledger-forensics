"""Controlled tampering engine.

Design notes that matter for the experiments:

1. Strength is NOT a single ladder. "Wrong font" and "recompressed after edit"
   are independent properties of an edit, not increasing points on one scale.
   Each axis is varied independently and gets its own curve.

   Axes:
     font_match          0/1  - edit rendered in a mismatched or matching face
     colour_match        0/1  - pure black ink vs ink colour sampled locally
     compression_match   0/1  - whole image re-encoded after the edit
     rerender            0/1  - bitmap paste vs region re-rendered on paper

2. ``reprint`` and ``full_forgery`` are separate CLASSES, not high strength
   levels of a local edit. They have no localisable edit region, so
   localisation metrics are undefined for them and reported separately.

3. Some classes are semantically invisible by construction (they change no
   value the arithmetic or the ledger can see). That is deliberate: it is what
   produces the channel confusion matrix.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..generation.document import BBox, Document, LineItem
from ..generation.generator import GENERATORS, DocumentRenderer
from ..world.registry import MERCHANTS

LOCAL_CLASSES = [
    "digit_substitution",
    "date_modification",
    "merchant_substitution",
    "line_item_insertion",
    "copy_move",
    "splice",
]
GLOBAL_CLASSES = ["reprint", "full_forgery"]
ALL_CLASSES = LOCAL_CLASSES + GLOBAL_CLASSES

MISMATCH_FONT = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"


@dataclass
class TamperSpec:
    tamper_class: str
    font_match: int = 1
    colour_match: int = 1
    compression_match: int = 1
    rerender: int = 0
    # for digit_substitution: does the fraudster fix the dependent arithmetic?
    arithmetic_consistent: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "tamper_class": self.tamper_class,
            "font_match": self.font_match,
            "colour_match": self.colour_match,
            "compression_match": self.compression_match,
            "rerender": self.rerender,
            "arithmetic_consistent": self.arithmetic_consistent,
        }


def _sample_paper(img: np.ndarray, bbox: BBox, rng: random.Random) -> np.ndarray:
    """Estimate the paper background around a region."""
    x0, y0, x1, y1 = bbox
    h, w = img.shape[:2]
    pad = 6
    ys = slice(max(0, y0 - pad), min(h, y1 + pad))
    xs = slice(max(0, x0 - pad), min(w, x1 + pad))
    patch = img[ys, xs].reshape(-1, 3)
    # paper is the bright mode; take a high percentile per channel
    return np.percentile(patch, 88, axis=0)


def _sample_ink(img: np.ndarray, bbox: BBox) -> Tuple[int, int, int]:
    x0, y0, x1, y1 = bbox
    patch = img[max(0, y0):y1, max(0, x0):x1].reshape(-1, 3)
    if patch.size == 0:
        return (20, 20, 20)
    v = np.percentile(patch, 6, axis=0)
    return tuple(int(c) for c in v)


def _erase(img: np.ndarray, bbox: BBox, rng: random.Random,
           grain: float = 2.5) -> None:
    """Paint over a region with locally-matched paper."""
    x0, y0, x1, y1 = bbox
    h, w = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    base = _sample_paper(img, bbox, rng)
    region = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32) + base
    region += np.random.default_rng(rng.randrange(1 << 30)).normal(
        0, grain, size=(y1 - y0, x1 - x0, 1))
    img[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)


def _draw_text_patch(img: np.ndarray, bbox: BBox, text: str,
                     font_path: str, size: int, colour: Tuple[int, int, int],
                     right_align: bool, rng: random.Random,
                     rerender: bool) -> np.ndarray:
    """Render replacement text into a region. Returns the touched mask."""
    x0, y0, x1, y1 = bbox
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    _erase(img, bbox, rng, grain=1.2 if rerender else 3.0)

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(font_path, size)
    tw = draw.textlength(text, font=font)
    tx = (x1 - tw - 1) if right_align else (x0 + 1)
    ty = y0 + 1
    draw.text((tx, ty), text, font=font, fill=colour)
    img[:, :, :] = np.array(pil)

    mx0, my0 = max(0, min(x0, int(tx)) - 2), max(0, y0 - 2)
    mx1, my1 = min(w, max(x1, int(tx + tw)) + 2), min(h, y1 + 2)
    mask[my0:my1, mx0:mx1] = 255
    return mask


def _jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


class TamperEngine:
    """Applies a labelled edit to a document and records exact ground truth."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    def apply(self, doc: Document, spec: TamperSpec,
              donor: Optional[Document] = None) -> Document:
        handler = getattr(self, f"_t_{spec.tamper_class}")
        img = doc.image.copy()
        mask = handler(doc, img, spec, donor)

        if spec.compression_match:
            img = _jpeg_roundtrip(img, quality=self.rng.choice([88, 90, 92]))

        doc.image = img
        doc.tamper_mask = mask
        doc.is_tampered = True
        doc.tamper_class = spec.tamper_class
        doc.tamper_params = spec.as_dict()
        return doc

    # ---- helpers ------------------------------------------------------
    def _font_for(self, doc: Document, spec: TamperSpec, bold: bool = False
                  ) -> Tuple[str, int]:
        cfg = GENERATORS[doc.generator_id]
        if spec.font_match:
            return (cfg.bold if bold else cfg.regular), cfg.base_size
        return MISMATCH_FONT, cfg.base_size + 1

    def _colour_for(self, doc: Document, spec: TamperSpec, bbox: BBox
                    ) -> Tuple[int, int, int]:
        if spec.colour_match:
            return _sample_ink(doc.clean_image, bbox)
        return (0, 0, 0)

    # ---- local edits --------------------------------------------------
    def _t_digit_substitution(self, doc, img, spec, donor) -> np.ndarray:
        f = doc.rendered_fields
        bbox = doc.layout["total"]
        old = f.total
        factor = self.rng.choice([2.0, 3.0, 5.0, 10.0])
        new_total = round(old * factor, 2)
        if new_total >= 100000:
            new_total = round(old + self.rng.uniform(200, 900), 2)

        font, size = self._font_for(doc, spec, bold=True)
        colour = self._colour_for(doc, spec, bbox)
        mask = _draw_text_patch(img, bbox, f"{new_total:.2f}", font, size,
                                colour, True, self.rng, bool(spec.rerender))
        f.total = new_total
        touched = ["total"]

        if spec.arithmetic_consistent:
            # A careful fraudster keeps subtotal + tax = total. The arithmetic
            # channel goes quiet; only the ledger disagrees.
            new_sub = round(new_total / (1 + f.tax_rate), 2)
            new_tax = round(new_total - new_sub, 2)
            for name, val in (("subtotal", new_sub), ("tax", new_tax)):
                bb = doc.layout[name]
                m2 = _draw_text_patch(img, bb, f"{val:.2f}", font, size,
                                      self._colour_for(doc, spec, bb), True,
                                      self.rng, bool(spec.rerender))
                mask = np.maximum(mask, m2)
            f.subtotal, f.tax = new_sub, new_tax
            touched += ["subtotal", "tax"]

        doc.tampered_fields = touched
        return mask

    def _t_date_modification(self, doc, img, spec, donor) -> np.ndarray:
        f = doc.rendered_fields
        bbox = doc.layout["date"]
        d = date.fromisoformat(f.date)
        nd = d + timedelta(days=self.rng.choice([-120, -45, 30, 90, 200]))
        font, size = self._font_for(doc, spec)
        colour = self._colour_for(doc, spec, bbox)
        prefix = "" if doc.template_id == "T_WIDE" else "Date  "
        mask = _draw_text_patch(img, bbox, f"{prefix}{nd.isoformat()}", font,
                                size, colour, doc.template_id == "T_WIDE",
                                self.rng, bool(spec.rerender))
        f.date = nd.isoformat()
        doc.tampered_fields = ["date"]
        return mask

    def _t_merchant_substitution(self, doc, img, spec, donor) -> np.ndarray:
        f = doc.rendered_fields
        bbox = doc.layout["merchant_name"]
        others = [m for m in MERCHANTS.values() if m.merchant_id != doc.merchant_id]
        new = self.rng.choice(others)
        cfg = GENERATORS[doc.generator_id]
        font, _ = self._font_for(doc, spec, bold=True)
        colour = self._colour_for(doc, spec, bbox)
        mask = _draw_text_patch(img, bbox, new.name, font, cfg.base_size + 5,
                                colour, False, self.rng, bool(spec.rerender))
        f.merchant_name = new.name
        doc.tampered_fields = ["merchant_name"]
        return mask

    def _t_line_item_insertion(self, doc, img, spec, donor) -> np.ndarray:
        """Insert a phantom line. Displayed items no longer sum to subtotal."""
        f = doc.rendered_fields
        n = len(f.line_items)
        anchor = doc.layout.get(f"item_{n-1}_desc")
        if anchor is None:
            return np.zeros(img.shape[:2], dtype=np.uint8)
        cfg = GENERATORS[doc.generator_id]
        x0, y0, x1, y1 = anchor
        y_new = y1 + 2
        h, w = img.shape[:2]
        if y_new + cfg.line_spacing >= h:
            y_new = max(0, h - cfg.line_spacing - 4)

        right = w - cfg.margin
        font, size = self._font_for(doc, spec)
        colour = self._colour_for(doc, spec, anchor)
        qty = self.rng.randint(1, 3)
        unit = round(self.rng.uniform(15, 120), 2)
        li = LineItem("Service charge", qty, unit, round(qty * unit, 2))

        mask = np.zeros((h, w), dtype=np.uint8)
        row_h = int(cfg.line_spacing)
        cells = [
            ((cfg.margin, y_new, cfg.margin + 200, y_new + row_h), li.description, False),
            ((right - 230, y_new, right - 190, y_new + row_h), str(li.quantity), True),
            ((right - 165, y_new, right - 105, y_new + row_h), f"{li.unit_price:.2f}", True),
            ((right - 70, y_new, right, y_new + row_h), f"{li.line_total:.2f}", True),
        ]
        for bb, txt, ra in cells:
            m = _draw_text_patch(img, bb, txt, font, size, colour, ra,
                                 self.rng, bool(spec.rerender))
            mask = np.maximum(mask, m)

        # A fraudster inserting a row would not leave the totals rule broken.
        # Repairing it keeps the class from being trivially detectable by eye,
        # which would otherwise inflate our detection rate.
        rule = doc.layout.get("_rule_totals")
        if rule is not None:
            rx0, ry0, rx1, ry1 = rule
            if ry0 < y_new + row_h and ry1 > y_new:
                pil = Image.fromarray(img)
                ImageDraw.Draw(pil).line(
                    [(rx0, (ry0 + ry1) // 2), (rx1, (ry0 + ry1) // 2)],
                    fill=colour, width=1)
                img[:, :, :] = np.array(pil)

        f.line_items.append(li)
        doc.tampered_fields = ["line_items"]
        return mask

    def _t_copy_move(self, doc, img, spec, donor) -> np.ndarray:
        """Duplicate a text block elsewhere on the same document.

        Changes no field value, so the semantic channel is blind by
        construction. This is a pure forensics case.
        """
        h, w = img.shape[:2]
        src = doc.layout.get("auth_code") or doc.layout["merchant_phone"]
        x0, y0, x1, y1 = src
        bw, bh = x1 - x0, y1 - y0
        if bw < 8 or bh < 6:
            return np.zeros((h, w), dtype=np.uint8)
        patch = img[y0:y1, x0:x1].copy()

        for _ in range(30):
            dx = self.rng.randint(10, max(11, w - bw - 10))
            dy = self.rng.randint(10, max(11, h - bh - 10))
            if abs(dy - y0) > bh * 2:
                break
        dx = min(dx, w - bw)
        dy = min(dy, h - bh)
        img[dy:dy + bh, dx:dx + bw] = patch

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[dy:dy + bh, dx:dx + bw] = 255
        doc.tampered_fields = []
        return mask

    def _t_splice(self, doc, img, spec, donor) -> np.ndarray:
        """Paste the merchant header from a different document."""
        h, w = img.shape[:2]
        if donor is None or donor.clean_image is None:
            return self._t_merchant_substitution(doc, img, spec, donor)
        dbox = donor.layout["merchant_name"]
        tbox = doc.layout["merchant_name"]
        dp = donor.clean_image[dbox[1]:dbox[3], dbox[0]:dbox[2]]
        th, tw = tbox[3] - tbox[1], tbox[2] - tbox[0]
        if dp.size == 0 or th <= 2 or tw <= 2:
            return self._t_merchant_substitution(doc, img, spec, donor)

        patch = np.array(Image.fromarray(dp).resize((tw, th), Image.LANCZOS))
        _erase(img, tbox, self.rng)
        img[tbox[1]:tbox[1] + th, tbox[0]:tbox[0] + tw] = patch

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[tbox[1]:tbox[1] + th, tbox[0]:tbox[0] + tw] = 255
        doc.rendered_fields.merchant_name = donor.rendered_fields.merchant_name
        doc.tampered_fields = ["merchant_name"]
        return mask

    # ---- global classes (no localisable edit) -------------------------
    def _t_reprint(self, doc, img, spec, donor) -> np.ndarray:
        """Print the document and photograph it again.

        Every value is honest. There is no edited region. This class exists to
        measure the floor: how often the system fires on a document that is
        merely second-generation rather than fraudulent.
        """
        out = _jpeg_roundtrip(img, quality=self.rng.choice([62, 70, 78]))
        out = _jpeg_roundtrip(out, quality=self.rng.choice([80, 86]))
        noise = np.random.default_rng(self.rng.randrange(1 << 30)).normal(
            0, 2.2, size=out.shape)
        img[:, :, :] = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        doc.tampered_fields = []
        return np.zeros(img.shape[:2], dtype=np.uint8)

    def _t_full_forgery(self, doc, img, spec, donor) -> np.ndarray:
        """Regenerate the whole document with fabricated amounts.

        No pixel is inherited from an authentic document, so pixel forensics
        has nothing local to find. The ledger is the only witness.
        """
        cfg = GENERATORS[doc.generator_id]
        f = doc.rendered_fields
        scale = self.rng.uniform(1.8, 4.5)
        for li in f.line_items:
            li.unit_price = round(li.unit_price * scale, 2)
            li.line_total = round(li.quantity * li.unit_price, 2)
        f.subtotal = round(sum(li.line_total for li in f.line_items), 2)
        f.tax = round(f.subtotal * f.tax_rate, 2)
        f.total = round(f.subtotal + f.tax, 2)

        renderer = DocumentRenderer(cfg, self.rng)
        new_img, new_layout = renderer.render(f, doc.template_id)
        h = min(new_img.shape[0], img.shape[0])
        w = min(new_img.shape[1], img.shape[1])
        img[:, :, :] = np.array(Image.fromarray(new_img).resize(
            (img.shape[1], img.shape[0]), Image.LANCZOS))
        doc.layout = new_layout
        doc.tampered_fields = ["line_items", "subtotal", "tax", "total"]
        return np.zeros(img.shape[:2], dtype=np.uint8)


def random_spec(rng: random.Random, tamper_class: str) -> TamperSpec:
    """Sample a tamper spec with independently varied axes."""
    return TamperSpec(
        tamper_class=tamper_class,
        font_match=rng.choice([0, 1]),
        colour_match=rng.choice([0, 1]),
        compression_match=rng.choice([0, 1]),
        rerender=rng.choice([0, 1]),
        arithmetic_consistent=rng.random() < 0.4,
    )
