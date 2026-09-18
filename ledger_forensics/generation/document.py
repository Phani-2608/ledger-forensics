"""Document data model.

The important distinction in this file is between *true* values and *rendered*
values.

- ``true_*``   : what the honest document said, and what the ledger recorded.
- ``rendered_*``: what is currently printed on the image.

For a clean document these agree. The tampering engine changes the rendered
values and leaves the true values alone, which is what gives us exact ground
truth for every downstream stage without any labelling effort.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]  # x0, y0, x1, y1


@dataclass
class LineItem:
    description: str
    quantity: int
    unit_price: float
    line_total: float

    def recompute(self) -> float:
        return round(self.quantity * self.unit_price, 2)


@dataclass
class DocumentFields:
    """The values printed on the document."""
    merchant_name: str
    merchant_address: str
    merchant_phone: str
    invoice_id: str
    date: str
    line_items: List[LineItem]
    subtotal: float
    tax_rate: float
    tax: float
    total: float
    card_last4: str
    auth_code: str

    def copy(self) -> "DocumentFields":
        return DocumentFields(
            merchant_name=self.merchant_name,
            merchant_address=self.merchant_address,
            merchant_phone=self.merchant_phone,
            invoice_id=self.invoice_id,
            date=self.date,
            line_items=[LineItem(li.description, li.quantity, li.unit_price,
                                 li.line_total) for li in self.line_items],
            subtotal=self.subtotal,
            tax_rate=self.tax_rate,
            tax=self.tax,
            total=self.total,
            card_last4=self.card_last4,
            auth_code=self.auth_code,
        )


@dataclass
class Document:
    doc_id: str
    merchant_id: str
    transaction_id: str
    generator_id: str                 # A / B / C / D
    template_id: str
    true_fields: DocumentFields       # honest values, matches the ledger
    rendered_fields: DocumentFields   # what is printed right now
    layout: Dict[str, BBox] = field(default_factory=dict)
    image: Optional[np.ndarray] = None        # uint8 HxWx3, current state
    clean_image: Optional[np.ndarray] = None  # pre-tamper, kept for analysis

    # ---- tamper ground truth -------------------------------------------
    is_tampered: bool = False
    tamper_class: str = "none"
    tamper_mask: Optional[np.ndarray] = None  # uint8 HxW, 255 = edited pixels
    tamper_params: Dict[str, object] = field(default_factory=dict)
    tampered_fields: List[str] = field(default_factory=list)

    @property
    def has_local_mask(self) -> bool:
        """Reprints and full forgeries have no localisable edit region.

        Localisation metrics are undefined for those classes and we report them
        separately rather than scoring them as a miss.
        """
        return self.tamper_mask is not None and self.tamper_mask.any()
