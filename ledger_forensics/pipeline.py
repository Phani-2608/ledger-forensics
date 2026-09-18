"""End-to-end assembly and scoring.

Stage order, and why it is this order:

    generate  ->  photograph  ->  tamper  ->  resave  ->  quality gate
                                                              |
                                          +-------------------+-------------------+
                                          |                                       |
                                  visual forensics                      OCR + semantic checks
                                          |                                       |
                                          +-------------------+-------------------+
                                                              |
                                                    evidence fusion
                                                              |
                                              calibration + abstention
                                                              |
                                                     error attribution
                                                              |
                                                 verified evidence report

The photograph stage runs before tampering because a fraudster edits a file
that has already been captured and compressed. Every clean and tampered
submission goes through the identical final resave so that compression history
alone cannot leak the label.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .capture.simulate import CaptureParams, photograph, resave, sample_params
from .extraction.ocr import OCREngine, OCRResult, SimulatedOCR
from .forensics import detectors, localize
from .generation.document import Document
from .generation.generator import DocumentGenerator
from .quality.gate import QualityReport, assess
from .semantics.checks import SemanticResult, run_checks
from .tampering.engine import ALL_CLASSES, TamperEngine, TamperSpec
from .world.registry import Ledger


@dataclass
class Submission:
    """One document as it arrives for review, plus all ground truth."""
    doc: Document
    capture: CaptureParams
    quality: QualityReport
    ocr: OCRResult
    forensics: detectors.ForensicResult
    semantics: SemanticResult
    # What the semantic channel WOULD have said with a perfect reader. Stored
    # at build time so the attribution stage can run its counterfactual repair
    # without re-deriving the whole document.
    semantics_oracle: SemanticResult = None
    localisation: Dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> int:
        return int(self.doc.is_tampered)

    @property
    def tamper_class(self) -> str:
        return self.doc.tamper_class

    @property
    def exposure(self) -> float:
        """Dollars at risk. Used by the expected-loss policy."""
        return float(self.doc.rendered_fields.total)

    def features(self) -> np.ndarray:
        f = list(self.forensics.vector(detectors.FEATURE_NAMES))
        s = list(self.semantics.vector())
        q = [self.quality.score, float(len(self.quality.reasons)) / 4.0]
        return np.array(f + s + q, dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        # Single source of truth. Two independent definitions of this list were
        # a silent drift hazard: the vector and its names could disagree and
        # nothing would fail loudly.
        from .fusion.models import feature_names as _names
        return _names()


class SubmissionBuilder:
    """Builds labelled submissions with exact ground truth at every stage."""

    def __init__(self, seed: int = 0, ocr: Optional[OCREngine] = None,
                 ocr_error_rate: float = 0.02,
                 generators: Optional[List[str]] = None):
        self.rng = random.Random(seed)
        self.ledger = Ledger()
        self.generator = DocumentGenerator(seed=seed, ledger=self.ledger)
        self.tamperer = TamperEngine(seed=seed + 1)
        self.ocr = ocr or SimulatedOCR(base_error_rate=ocr_error_rate,
                                       seed=seed + 2)
        self.perfect_ocr = SimulatedOCR(base_error_rate=0.0, seed=seed + 3)
        self.generators = generators or ["A", "B", "C"]
        self.history: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    def build(self, tamper_class: Optional[str] = None,
              spec: Optional[TamperSpec] = None,
              capture: Optional[CaptureParams] = None,
              generator_id: Optional[str] = None) -> Submission:
        gid = generator_id or self.rng.choice(self.generators)
        doc = self.generator.generate(generator_id=gid)

        cap = capture or sample_params(self.rng)
        self._last_capture = cap
        photo, warped_layout = photograph(doc.image, doc.layout, cap, self.rng)
        doc.image = photo
        doc.layout = warped_layout
        doc.clean_image = photo.copy()

        if tamper_class is not None and tamper_class != "none":
            donor = None
            if tamper_class == "splice":
                donor = self.generator.generate(
                    generator_id=self.rng.choice(self.generators))
                dphoto, dlayout = photograph(donor.image, donor.layout,
                                             cap, self.rng)
                donor.image, donor.clean_image, donor.layout = (
                    dphoto, dphoto.copy(), dlayout)
            spec = spec or TamperSpec(tamper_class)
            self.tamperer.apply(doc, spec, donor=donor)

        # Identical final save for clean and tampered submissions alike, so
        # compression history cannot leak the label.
        doc.image = resave(doc.image, quality=self.rng.choice([88, 90, 92]))

        return self._score(doc)

    # ------------------------------------------------------------------
    def _score(self, doc: Document) -> Submission:
        cap = getattr(self, "_last_capture", None) or CaptureParams()
        quality = assess(doc.image)
        ocr = self.ocr.read(doc, quality_score=quality.score)
        forensic = detectors.run_all(doc.image)
        semantic = run_checks(doc, ocr, self.ledger, self.history)
        oracle_ocr = self.perfect_ocr.read(doc, quality_score=1.0)
        semantic_oracle = run_checks(doc, oracle_ocr, self.ledger, self.history)
        loc = localize.localisation_metrics(forensic, doc.tamper_mask)

        self.history.append({
            "doc_id": doc.doc_id,
            "merchant": ocr.values.get("merchant_name", ""),
            "date": ocr.values.get("date", ""),
            "total": ocr.values.get("total", ""),
            "invoice_id": ocr.values.get("invoice_id", ""),
        })

        return Submission(doc=doc, capture=cap, quality=quality,
                          ocr=ocr, forensics=forensic, semantics=semantic,
                          semantics_oracle=semantic_oracle, localisation=loc)


def build_dataset(n: int, seed: int = 0, tamper_rate: float = 0.5,
                  generators: Optional[List[str]] = None,
                  classes: Optional[List[str]] = None,
                  ocr_error_rate: float = 0.02,
                  progress: bool = False) -> List[Submission]:
    """Build a labelled dataset with a balanced spread of tamper classes."""
    from .tampering.engine import random_spec

    rng = random.Random(seed)
    builder = SubmissionBuilder(seed=seed, generators=generators,
                                ocr_error_rate=ocr_error_rate)
    classes = classes or ALL_CLASSES
    out: List[Submission] = []
    for i in range(n):
        if rng.random() < tamper_rate:
            cls = classes[i % len(classes)]
            spec = random_spec(rng, cls)
            out.append(builder.build(tamper_class=cls, spec=spec))
        else:
            out.append(builder.build(tamper_class=None))
        if progress and (i + 1) % 25 == 0:
            print(f"  built {i + 1}/{n}", flush=True)
    return out
