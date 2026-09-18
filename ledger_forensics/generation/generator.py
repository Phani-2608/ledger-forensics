"""Synthetic document generator.

Four generator configurations (A, B, C, D) differ in font family, metrics,
layout template, paper texture and ink characteristics. They are not cosmetic
variants: generator D is held out entirely in the unseen-generator experiment
to test whether the detectors learned tampering evidence or learned our own
rendering pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..world.registry import (CATALOG, MERCHANTS, Ledger, TransactionRecord,
                              auth_code, tax_rate_for)
from .document import BBox, Document, DocumentFields, LineItem

FONT_DIR_LIB = "/usr/share/fonts/truetype/liberation"
FONT_DIR_DEJA = "/usr/share/fonts/truetype/dejavu"


@dataclass(frozen=True)
class GeneratorConfig:
    generator_id: str
    regular: str
    bold: str
    base_size: int
    line_spacing: int
    margin: int
    width: int
    height: int
    paper_tint: Tuple[int, int, int]
    paper_grain: float
    ink_level: int          # 0 = pure black, higher = greyer ink
    ink_jitter: float       # per-glyph vertical jitter, in pixels
    letter_spacing: int


GENERATORS: Dict[str, GeneratorConfig] = {
    "A": GeneratorConfig("A", f"{FONT_DIR_LIB}/LiberationMono-Regular.ttf",
                         f"{FONT_DIR_LIB}/LiberationMono-Bold.ttf",
                         17, 26, 34, 620, 900, (252, 251, 246), 3.2, 18, 0.35, 0),
    "B": GeneratorConfig("B", f"{FONT_DIR_LIB}/LiberationSans-Regular.ttf",
                         f"{FONT_DIR_LIB}/LiberationSans-Bold.ttf",
                         18, 28, 42, 660, 940, (255, 255, 255), 1.8, 8, 0.15, 0),
    "C": GeneratorConfig("C", f"{FONT_DIR_DEJA}/DejaVuSerif.ttf",
                         f"{FONT_DIR_DEJA}/DejaVuSerif-Bold.ttf",
                         16, 27, 38, 640, 920, (250, 248, 240), 4.5, 26, 0.5, 0),
    # D is the held-out generator: different family, tighter metrics,
    # condensed face, warmer paper, heavier grain.
    "D": GeneratorConfig("D", f"{FONT_DIR_DEJA}/DejaVuSansCondensed.ttf",
                         f"{FONT_DIR_DEJA}/DejaVuSansCondensed-Bold.ttf",
                         19, 24, 30, 600, 880, (248, 245, 235), 5.5, 34, 0.6, 1),
}


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _paper(cfg: GeneratorConfig, rng: random.Random) -> Image.Image:
    """Paper background with grain and a soft illumination gradient."""
    w, h = cfg.width, cfg.height
    base = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        base[:, :, c] = cfg.paper_tint[c]

    grain = np.random.default_rng(rng.randrange(1 << 30)).normal(
        0.0, cfg.paper_grain, size=(h, w, 1))
    base += grain

    # gentle two-axis illumination falloff, different every document
    yy, xx = np.mgrid[0:h, 0:w]
    cx = rng.uniform(0.25, 0.75) * w
    cy = rng.uniform(0.25, 0.75) * h
    r = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    base -= (r * rng.uniform(3.0, 9.0))[:, :, None]

    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


class DocumentRenderer:
    """Renders a DocumentFields onto paper and records exact field bboxes."""

    def __init__(self, cfg: GeneratorConfig, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.regular = _font(cfg.regular, cfg.base_size)
        self.bold = _font(cfg.bold, cfg.base_size)
        self.title = _font(cfg.bold, cfg.base_size + 5)
        self.small = _font(cfg.regular, cfg.base_size - 3)

    def _ink(self) -> Tuple[int, int, int]:
        v = self.cfg.ink_level + int(self.rng.uniform(-4, 4))
        v = max(0, min(90, v))
        return (v, v, v)

    def _text(self, draw: ImageDraw.ImageDraw, xy, s: str,
              font: ImageFont.FreeTypeFont, anchor: Optional[str] = None) -> BBox:
        x, y = xy
        y = y + self.rng.uniform(-self.cfg.ink_jitter, self.cfg.ink_jitter)
        if anchor == "ra":
            w = draw.textlength(s, font=font)
            x = x - w
        draw.text((x, y), s, font=font, fill=self._ink())
        w = draw.textlength(s, font=font)
        asc, desc = font.getmetrics()
        return (int(x) - 1, int(y) - 1, int(x + w) + 1, int(y + asc + desc * 0.4) + 1)

    def render(self, fields: DocumentFields, template_id: str
               ) -> Tuple[np.ndarray, Dict[str, BBox]]:
        cfg = self.cfg
        img = _paper(cfg, self.rng)
        draw = ImageDraw.Draw(img)
        layout: Dict[str, BBox] = {}

        m = cfg.margin
        right = cfg.width - m
        y = m
        wide = template_id == "T_WIDE"

        layout["merchant_name"] = self._text(draw, (m, y), fields.merchant_name,
                                             self.title)
        y += cfg.line_spacing + 8
        layout["merchant_address"] = self._text(draw, (m, y),
                                                fields.merchant_address, self.small)
        y += cfg.line_spacing - 4
        layout["merchant_phone"] = self._text(draw, (m, y),
                                              f"Tel {fields.merchant_phone}",
                                              self.small)
        y += cfg.line_spacing + 6

        draw.line([(m, y), (right, y)], fill=self._ink(), width=1)
        y += 14

        if wide:
            layout["invoice_id"] = self._text(draw, (m, y),
                                              f"Invoice  {fields.invoice_id}",
                                              self.regular)
            layout["date"] = self._text(draw, (right, y), fields.date,
                                        self.regular, anchor="ra")
            y += cfg.line_spacing
        else:
            layout["invoice_id"] = self._text(draw, (m, y),
                                              f"Invoice  {fields.invoice_id}",
                                              self.regular)
            y += cfg.line_spacing - 2
            layout["date"] = self._text(draw, (m, y), f"Date  {fields.date}",
                                        self.regular)
            y += cfg.line_spacing

        y += 8
        col_qty = right - 190
        col_price = right - 105
        self._text(draw, (m, y), "ITEM", self.bold)
        self._text(draw, (col_qty, y), "QTY", self.bold, anchor="ra")
        self._text(draw, (col_price, y), "PRICE", self.bold, anchor="ra")
        self._text(draw, (right, y), "AMOUNT", self.bold, anchor="ra")
        y += cfg.line_spacing + 4
        draw.line([(m, y - 6), (right, y - 6)], fill=self._ink(), width=1)

        for i, li in enumerate(fields.line_items):
            desc = li.description[:24]
            layout[f"item_{i}_desc"] = self._text(draw, (m, y), desc, self.regular)
            layout[f"item_{i}_qty"] = self._text(draw, (col_qty, y),
                                                 str(li.quantity), self.regular,
                                                 anchor="ra")
            layout[f"item_{i}_price"] = self._text(draw, (col_price, y),
                                                   f"{li.unit_price:.2f}",
                                                   self.regular, anchor="ra")
            layout[f"item_{i}_total"] = self._text(draw, (right, y),
                                                   f"{li.line_total:.2f}",
                                                   self.regular, anchor="ra")
            y += cfg.line_spacing

        y += 10
        draw.line([(right - 250, y), (right, y)], fill=self._ink(), width=1)
        layout["_rule_totals"] = (right - 250, int(y) - 1, right, int(y) + 1)
        y += 12

        self._text(draw, (right - 250, y), "Subtotal", self.regular)
        layout["subtotal"] = self._text(draw, (right, y), f"{fields.subtotal:.2f}",
                                        self.regular, anchor="ra")
        y += cfg.line_spacing

        self._text(draw, (right - 250, y), f"Tax {fields.tax_rate * 100:.2f}%",
                   self.regular)
        layout["tax"] = self._text(draw, (right, y), f"{fields.tax:.2f}",
                                   self.regular, anchor="ra")
        y += cfg.line_spacing + 4

        self._text(draw, (right - 250, y), "TOTAL", self.bold)
        layout["total"] = self._text(draw, (right, y), f"{fields.total:.2f}",
                                     self.bold, anchor="ra")
        y += cfg.line_spacing + 18

        draw.line([(m, y), (right, y)], fill=self._ink(), width=1)
        y += 14
        layout["card_last4"] = self._text(draw, (m, y),
                                          f"CARD ****{fields.card_last4}",
                                          self.small)
        y += cfg.line_spacing - 6
        layout["auth_code"] = self._text(draw, (m, y), f"AUTH {fields.auth_code}",
                                         self.small)
        y += cfg.line_spacing + 4
        self._text(draw, (m, y), "Thank you for your business", self.small)

        # Crop to content so receipt length varies with item count, the way a
        # real till roll does.
        content_bottom = int(y) + cfg.line_spacing + cfg.margin
        content_bottom = min(cfg.height, max(240, content_bottom))
        arr = np.array(img)[:content_bottom, :, :]
        return arr, layout


class DocumentGenerator:
    """Generates honest documents plus their matching ledger transactions."""

    def __init__(self, seed: int = 0, ledger: Optional[Ledger] = None):
        self.rng = random.Random(seed)
        self.ledger = ledger if ledger is not None else Ledger()
        self._n = 0

    def _fields(self, merchant_id: str, doc_id: str) -> DocumentFields:
        merchant = MERCHANTS[merchant_id]
        catalog = CATALOG[merchant.category]
        n_items = self.rng.randint(2, 6)
        items: List[LineItem] = []
        for desc, price in self.rng.sample(catalog, min(n_items, len(catalog))):
            qty = self.rng.randint(1, 4)
            unit = round(price * self.rng.uniform(0.95, 1.05), 2)
            items.append(LineItem(desc, qty, unit, round(qty * unit, 2)))

        subtotal = round(sum(li.line_total for li in items), 2)
        rate = tax_rate_for(merchant_id)
        tax = round(subtotal * rate, 2)
        total = round(subtotal + tax, 2)

        d = date(2025, 1, 1) + timedelta(days=self.rng.randint(0, 540))
        return DocumentFields(
            merchant_name=merchant.name,
            merchant_address=merchant.address,
            merchant_phone=merchant.phone,
            invoice_id=f"{merchant_id}-{self.rng.randint(10000, 99999)}",
            date=d.isoformat(),
            line_items=items,
            subtotal=subtotal,
            tax_rate=rate,
            tax=tax,
            total=total,
            card_last4=f"{self.rng.randint(0, 9999):04d}",
            auth_code=auth_code(doc_id),
        )

    def generate(self, generator_id: Optional[str] = None,
                 merchant_id: Optional[str] = None) -> Document:
        self._n += 1
        doc_id = f"DOC{self._n:06d}"
        gid = generator_id or self.rng.choice(["A", "B", "C"])
        mid = merchant_id or self.rng.choice(list(MERCHANTS))
        cfg = GENERATORS[gid]
        merchant = MERCHANTS[mid]

        fields = self._fields(mid, doc_id)
        renderer = DocumentRenderer(cfg, self.rng)
        image, layout = renderer.render(fields, merchant.template_id)

        txn_id = f"TXN{self._n:06d}"
        self.ledger.add(TransactionRecord(
            transaction_id=txn_id, merchant_id=mid, amount=fields.total,
            date=fields.date, card_last4=fields.card_last4,
            auth_code=fields.auth_code))

        return Document(
            doc_id=doc_id, merchant_id=mid, transaction_id=txn_id,
            generator_id=gid, template_id=merchant.template_id,
            true_fields=fields, rendered_fields=fields.copy(),
            layout=layout, image=image, clean_image=image.copy())
