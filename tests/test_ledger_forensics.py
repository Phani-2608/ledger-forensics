"""Tests.

The ones that matter here are the INVARIANT tests. A synthetic-data project
lives or dies on whether its ground truth is actually correct, so the ground
truth is what gets checked hardest: that clean documents are arithmetically
consistent, that tamper masks cover the pixels that really changed, that the
label cannot leak through compression history, and that localisation is only
scored where it is defined.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from ledger_forensics.capture.simulate import CLEAN, CaptureParams, photograph
from ledger_forensics.decision.policy import (AbstentionPolicy, CostModel,
                                              Calibrator,
                                              expected_calibration_error,
                                              expected_loss, optimal_threshold)
from ledger_forensics.forensics import detectors, localize
from ledger_forensics.generation.generator import GENERATORS, DocumentGenerator
from ledger_forensics.pipeline import SubmissionBuilder, build_dataset
from ledger_forensics.reporting.evidence import (build_packet, narrate,
                                                 render_template,
                                                 verify_numbers)
from ledger_forensics.semantics.checks import CHECK_NAMES
from ledger_forensics.tampering.engine import (ALL_CLASSES, GLOBAL_CLASSES,
                                               LOCAL_CLASSES, TamperEngine,
                                               TamperSpec)
from ledger_forensics.world.registry import MERCHANTS, tax_rate_for


# --------------------------------------------------------------- world
def test_every_merchant_has_a_valid_jurisdiction():
    for m in MERCHANTS.values():
        assert 0.0 <= tax_rate_for(m.merchant_id) < 0.2


# ---------------------------------------------------- generation invariants
@pytest.mark.parametrize("gid", ["A", "B", "C", "D"])
def test_generated_documents_are_arithmetically_consistent(gid):
    gen = DocumentGenerator(seed=42)
    for _ in range(4):
        doc = gen.generate(generator_id=gid)
        f = doc.true_fields
        for li in f.line_items:
            assert abs(li.recompute() - li.line_total) < 0.011
        assert abs(sum(li.line_total for li in f.line_items) - f.subtotal) < 0.011
        assert abs(f.subtotal * f.tax_rate - f.tax) < 0.011
        assert abs(f.subtotal + f.tax - f.total) < 0.011


def test_ledger_matches_the_honest_document():
    gen = DocumentGenerator(seed=1)
    doc = gen.generate()
    txn = gen.ledger.get(doc.transaction_id)
    assert txn is not None
    assert abs(txn.amount - doc.true_fields.total) < 0.011
    assert txn.date == doc.true_fields.date


def test_generators_are_actually_different():
    """If the four configs render identically, the unseen-generator experiment
    is measuring nothing."""
    imgs = {}
    for gid in ["A", "B", "C", "D"]:
        doc = DocumentGenerator(seed=3).generate(generator_id=gid)
        imgs[gid] = doc.image
    shapes = {g: im.shape for g, im in imgs.items()}
    assert len(set(shapes.values())) >= 3


# ---------------------------------------------------- tampering invariants
@pytest.mark.parametrize("cls", LOCAL_CLASSES)
def test_local_tampers_produce_a_mask_over_changed_pixels(cls):
    gen = DocumentGenerator(seed=11)
    doc = gen.generate(generator_id="A")
    donor = gen.generate(generator_id="B")
    doc.image, doc.layout = photograph(doc.image, doc.layout, CLEAN)
    doc.clean_image = doc.image.copy()
    donor.image, donor.layout = photograph(donor.image, donor.layout, CLEAN)
    donor.clean_image = donor.image.copy()
    before = doc.clean_image.copy()

    TamperEngine(seed=11).apply(
        doc, TamperSpec(cls, compression_match=0), donor=donor)

    assert doc.tamper_mask is not None
    assert doc.tamper_mask.any(), f"{cls} produced an empty mask"

    changed = (np.abs(before.astype(int) - doc.image.astype(int)).sum(axis=2) > 12)
    assert changed.any(), f"{cls} changed no pixels"
    covered = changed & (doc.tamper_mask > 0)
    # The mask must account for the large majority of changed pixels, or the
    # localisation metrics are being graded against a wrong answer.
    assert covered.sum() / changed.sum() > 0.75, (
        f"{cls}: mask covers only {covered.sum() / changed.sum():.1%} "
        "of changed pixels")


@pytest.mark.parametrize("cls", GLOBAL_CLASSES)
def test_global_classes_have_no_localisable_region(cls):
    gen = DocumentGenerator(seed=12)
    doc = gen.generate(generator_id="A")
    TamperEngine(seed=12).apply(doc, TamperSpec(cls))
    assert not doc.has_local_mask
    metrics = localize.localisation_metrics(
        detectors.run_all(doc.image), doc.tamper_mask)
    assert metrics["defined"] == 0.0
    assert np.isnan(metrics["iou"])


def test_strength_axes_are_independent():
    """Each axis must be settable without touching the others."""
    spec = TamperSpec("digit_substitution", font_match=0, colour_match=1,
                      compression_match=0, rerender=1)
    d = spec.as_dict()
    assert d["font_match"] == 0 and d["colour_match"] == 1
    assert d["compression_match"] == 0 and d["rerender"] == 1


# ---------------------------------------------------------- label leakage
def test_compression_history_does_not_leak_the_label():
    """Clean and tampered submissions must get an identical final save.

    If tampered documents were saved differently, a model could reach a high
    AUC by reading compression history and every result would be worthless.
    """
    builder = SubmissionBuilder(seed=5)
    clean_sizes, tampered_sizes = [], []
    import io
    from PIL import Image
    for i in range(6):
        c = builder.build(None)
        t = builder.build(tamper_class="copy_move",
                          spec=TamperSpec("copy_move"))
        for sub, bucket in ((c, clean_sizes), (t, tampered_sizes)):
            buf = io.BytesIO()
            Image.fromarray(sub.doc.image).save(buf, format="PNG")
            bucket.append(buf.tell() / sub.doc.image.size)
    # Distributions should overlap; a hard separation would be a leak.
    assert min(clean_sizes) < max(tampered_sizes)
    assert min(tampered_sizes) < max(clean_sizes)


# --------------------------------------------------------------- forensics
def test_detectors_return_bounded_scores_and_matching_heatmaps():
    doc = DocumentGenerator(seed=8).generate(generator_id="B")
    result = detectors.run_all(doc.image)
    for name in detectors.DETECTOR_NAMES:
        assert 0.0 <= result.scores[name] <= 1.0
        assert result.heatmaps[name].shape == doc.image.shape[:2]
    assert len(result.vector(detectors.FEATURE_NAMES)) == len(
        detectors.FEATURE_NAMES)


def test_copy_move_fires_on_a_copy_and_not_on_a_clean_document():
    gen = DocumentGenerator(seed=21)
    clean = gen.generate(generator_id="A")
    clean.image, clean.layout = photograph(clean.image, clean.layout, CLEAN)
    clean.clean_image = clean.image.copy()
    clean_score, _ = detectors.copy_move(clean.image)

    doc = DocumentGenerator(seed=21).generate(generator_id="A")
    doc.image, doc.layout = photograph(doc.image, doc.layout, CLEAN)
    doc.clean_image = doc.image.copy()
    TamperEngine(seed=21).apply(doc, TamperSpec("copy_move",
                                                compression_match=0))
    tampered_score, _ = detectors.copy_move(doc.image)
    assert tampered_score > clean_score


def test_concentration_is_zero_on_a_flat_map():
    assert detectors.concentration(np.ones((64, 64), dtype=np.float32)) == 0.0


def test_iou_is_exact_on_a_known_overlap():
    pred = np.zeros((10, 10), dtype=np.uint8)
    truth = np.zeros((10, 10), dtype=np.uint8)
    pred[0:4, 0:4] = 1
    truth[2:6, 0:4] = 255
    assert abs(localize.iou(pred, truth) - (8 / 24)) < 1e-6


# --------------------------------------------------------------- semantics
def test_clean_documents_pass_every_check_with_a_perfect_reader():
    builder = SubmissionBuilder(seed=31, ocr_error_rate=0.0)
    for _ in range(6):
        sub = builder.build(None)
        failed = [k for k, v in sub.semantics.failures.items() if v]
        assert not failed, f"clean document failed {failed}"


def test_digit_substitution_is_visible_to_the_ledger():
    builder = SubmissionBuilder(seed=32, ocr_error_rate=0.0)
    sub = builder.build(tamper_class="digit_substitution",
                        spec=TamperSpec("digit_substitution",
                                        arithmetic_consistent=True))
    # An arithmetically consistent forgery goes quiet on the arithmetic checks
    # but the payment ledger still disagrees.
    assert sub.semantics.failures["ledger_amount_match"]


def test_copy_move_is_semantically_invisible_by_construction():
    builder = SubmissionBuilder(seed=33, ocr_error_rate=0.0)
    sub = builder.build(tamper_class="copy_move",
                        spec=TamperSpec("copy_move"))
    assert not sub.semantics.any_failure


def test_semantic_vector_width_is_stable():
    builder = SubmissionBuilder(seed=34, ocr_error_rate=0.0)
    sub = builder.build(None)
    assert len(sub.semantics.vector()) == len(CHECK_NAMES) * 2


# --------------------------------------------------------------- decision
def test_calibration_improves_a_deliberately_miscalibrated_score():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, size=400)
    p = np.clip(0.5 + 0.25 * y + rng.normal(0, 0.1, size=400), 0, 1)
    squashed = p ** 3  # badly miscalibrated on purpose
    before, _ = expected_calibration_error(squashed, y)
    cal = Calibrator("isotonic").fit(squashed[:200], y[:200])
    after, _ = expected_calibration_error(cal.transform(squashed[200:]), y[200:])
    assert after < before


def test_expected_loss_prefers_catching_expensive_fraud():
    p = np.array([0.9, 0.1])
    y = np.array([1, 0])
    exposure = np.array([5000.0, 50.0])
    strict = expected_loss(p, y, exposure, 0.5, CostModel())
    lax = expected_loss(p, y, exposure, 0.99, CostModel())
    assert strict["net_saving"] > lax["net_saving"]


def test_abstention_policy_orders_its_bands():
    policy = AbstentionPolicy(review_threshold=0.3, reject_threshold=0.7)
    assert policy.decide(0.1) == "pay"
    assert policy.decide(0.5) == "review"
    assert policy.decide(0.9) == "reject"


# --------------------------------------------------------------- reporting
def test_report_never_invents_a_number():
    builder = SubmissionBuilder(seed=41)
    sub = builder.build(tamper_class="digit_substitution",
                        spec=TamperSpec("digit_substitution"))
    packet = build_packet(sub, 0.62, "review")
    text = render_template(packet)
    ok, unverified = verify_numbers(text, packet)
    assert ok, f"template emitted unverified numbers: {unverified}"


def test_narrator_falls_back_to_the_template_without_an_llm():
    builder = SubmissionBuilder(seed=42)
    sub = builder.build(None)
    out = narrate(build_packet(sub, 0.1, "pay"), use_llm=False)
    assert out["source"] == "template" and out["verified"]


def test_verifier_rejects_a_hallucinated_number():
    builder = SubmissionBuilder(seed=43)
    sub = builder.build(None)
    packet = build_packet(sub, 0.1, "pay")
    ok, bad = verify_numbers("The loss was 918273.55 dollars.", packet)
    assert not ok and "918273.55" in bad


# -------------------------------------------------------------- attribution
def test_attribution_assigns_every_error_a_cause():
    from ledger_forensics.attribution.attribute import (CAUSE_ORDER,
                                                        OracleProfile,
                                                        attribute, summarise)
    from ledger_forensics.experiments.suite import pick_model

    subs = build_dataset(30, seed=51, tamper_rate=0.5)
    X = np.array([s.features() for s in subs], dtype=np.float32)
    y = np.array([s.label for s in subs], dtype=int)
    model = pick_model(subs).fit(X, y)
    oracle = OracleProfile.fit(X)
    atts = attribute(subs, X, y, model.predict_proba, 0.5, oracle)
    for a in atts:
        assert a.primary_cause in CAUSE_ORDER
        assert a.sufficient_causes
    summary = summarise(atts, len(subs))
    assert summary["n_total"] == 30


# ---------------------------------------------------------------- pipeline
def test_pipeline_produces_stable_feature_widths():
    from ledger_forensics.fusion.models import ALL_FEATURES
    subs = build_dataset(6, seed=61, tamper_rate=0.5)
    widths = {len(s.features()) for s in subs}
    assert widths == {len(ALL_FEATURES)}


def test_dataset_is_reproducible_from_a_seed():
    a = build_dataset(6, seed=71, tamper_rate=0.5)
    b = build_dataset(6, seed=71, tamper_rate=0.5)
    assert [s.label for s in a] == [s.label for s in b]
    assert np.allclose(np.array([s.features() for s in a]),
                       np.array([s.features() for s in b]))
