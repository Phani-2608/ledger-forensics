"""The experiment suite.

Each function answers one question and returns plain dictionaries so the
results can be serialised, tested, and rendered without a plotting library in
the loop.

  E1 baseline_ladder        does two channels beat the obvious alternatives
  E2 channel_matrix         what does each channel catch that the other cannot
  E3 per_class              which tamper classes are catchable at all
  E4 leave_one_class_out    does the learned layer generalise to an unseen attack
  E5 unseen_generator       did we learn tampering or our own renderer
  E6 strength_axes          one curve per axis, never a single ladder
  E7 degradation            how capture quality erodes detection
  E8 ocr_sweep              how much decision error the reading stage causes
  E9 calibration_and_cost   is the score meaningful, and what is it worth
  E10 attribution           which stage is responsible for the errors
  E11 localisation          can we say WHERE, not just whether
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..attribution.attribute import OracleProfile, attribute, summarise
from ..capture.simulate import CaptureParams
from ..decision.policy import (Calibrator, CostModel, expected_calibration_error,
                               optimal_threshold, ranking_metrics,
                               sweep_thresholds)
from ..fusion.models import build_ladder
from ..pipeline import SubmissionBuilder, build_dataset
from ..tampering.engine import ALL_CLASSES, LOCAL_CLASSES, TamperSpec


def _xy(subs: Sequence):
    X = np.array([s.features() for s in subs], dtype=np.float32)
    y = np.array([s.label for s in subs], dtype=int)
    e = np.array([s.exposure for s in subs], dtype=float)
    return X, y, e


def pick_model(train):
    """Gradient boosting needs data. Below a few hundred rows it degenerates to
    predicting the base rate, so fall back to the linear model rather than
    silently reporting an AUC of 0.5 as a finding."""
    return build_ladder()["fused_gbm" if len(train) >= 150 else "fused_logreg"]


def resample_prevalence(subs: Sequence, prevalence: float, seed: int = 0):
    """Reweight a balanced test set to a realistic fraud rate.

    Datasets are built balanced so the models have something to learn from, but
    the ECONOMICS of a review queue are dominated by base rate: at 50 percent
    fraud, reviewing every case is trivially optimal and the threshold analysis
    says nothing. Real return-fraud prevalence is low single digits, so the cost
    experiment resamples to that before drawing any conclusion about operating
    points."""
    rng = random.Random(seed)
    clean = [s for s in subs if s.label == 0]
    tampered = [s for s in subs if s.label == 1]
    if not clean or not tampered:
        return list(subs)
    n_tamp = max(1, int(len(clean) * prevalence / max(1e-6, 1 - prevalence)))
    n_tamp = min(n_tamp, len(tampered))
    out = clean + rng.sample(tampered, n_tamp)
    rng.shuffle(out)
    return out


def split(subs: Sequence, frac: float = 0.6, seed: int = 0):
    idx = list(range(len(subs)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * frac)
    cut2 = int(len(idx) * (frac + (1 - frac) / 2))
    return ([subs[i] for i in idx[:cut]],
            [subs[i] for i in idx[cut:cut2]],
            [subs[i] for i in idx[cut2:]])


# ---------------------------------------------------------------- E1
def baseline_ladder(train, test) -> Dict[str, Dict[str, float]]:
    Xtr, ytr, _ = _xy(train)
    Xte, yte, _ = _xy(test)
    out: Dict[str, Dict[str, float]] = {}
    for name, model in build_ladder().items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)
        out[name] = ranking_metrics(p, yte)
    return out


# ---------------------------------------------------------------- E2
def channel_matrix(subs: Sequence) -> Dict[str, object]:
    """Which channel sees what.

    Forensics is counted as 'seeing' a document when its strongest field-level
    signal exceeds the 90th percentile of the clean distribution. Semantics is
    counted as seeing it when any check fails. Both operating points are fixed
    on clean data so the comparison is fair.
    """
    from ..forensics.detectors import DETECTOR_NAMES

    def forensic_peak(s) -> float:
        return max(s.forensics.scores.get(f"{n}_fld", 0.0)
                   for n in DETECTOR_NAMES)

    clean = [s for s in subs if s.label == 0]
    if not clean:
        return {}
    thr = float(np.percentile([forensic_peak(s) for s in clean], 90))

    cells = {"both": 0, "forensics_only": 0, "semantics_only": 0, "neither": 0}
    per_class: Dict[str, Dict[str, int]] = {}
    for s in subs:
        if s.label == 0:
            continue
        f = forensic_peak(s) > thr
        m = s.semantics.any_failure
        key = ("both" if f and m else "forensics_only" if f
               else "semantics_only" if m else "neither")
        cells[key] += 1
        per_class.setdefault(s.tamper_class, dict.fromkeys(cells, 0))[key] += 1

    total = max(1, sum(cells.values()))
    return {
        "forensic_threshold": thr,
        "counts": cells,
        "share": {k: v / total for k, v in cells.items()},
        "by_class": per_class,
    }


# ---------------------------------------------------------------- E3
def per_class_detection(train, test, threshold: Optional[float] = None
                        ) -> Dict[str, object]:
    Xtr, ytr, _ = _xy(train)
    Xte, yte, _ = _xy(test)
    model = pick_model(train).fit(Xtr, ytr)
    p = model.predict_proba(Xte)

    if threshold is None:
        clean_p = p[yte == 0]
        threshold = float(np.percentile(clean_p, 90)) if len(clean_p) else 0.5

    rows: Dict[str, Dict[str, float]] = {}
    for cls in ALL_CLASSES + ["none"]:
        sel = [i for i, s in enumerate(test) if s.tamper_class == cls]
        if not sel:
            continue
        pp = p[sel]
        if cls == "none":
            rows[cls] = {"n": len(sel),
                         "false_alarm_rate": float((pp >= threshold).mean()),
                         "mean_score": float(pp.mean())}
        else:
            rows[cls] = {"n": len(sel),
                         "detection_rate": float((pp >= threshold).mean()),
                         "mean_score": float(pp.mean())}
    return {"threshold": threshold, "by_class": rows}


# ---------------------------------------------------------------- E4
def leave_one_class_out(subs: Sequence, classes: Optional[List[str]] = None
                        ) -> Dict[str, object]:
    """Train without one attack class, test exclusively on it.

    IMPORTANT CAVEAT, and it changes how this table should be read.

    Only the FUSION layer is learned. The six forensic detectors and the nine
    semantic checks are deterministic algorithms with no fitted parameters, so
    holding an attack out does not hide it from them. What this experiment
    actually tests is whether the learned combiner generalises to an evidence
    pattern it never saw during fitting. That is a real and useful question,
    but it is a narrower one than "can the system detect an unseen attack", and
    reporting it as the wider claim would be dishonest.
    """
    classes = classes or LOCAL_CLASSES
    out: Dict[str, Dict[str, float]] = {}
    for held in classes:
        train = [s for s in subs if s.tamper_class not in (held,)]
        test = [s for s in subs if s.tamper_class in (held, "none")]
        if len(test) < 10 or len({s.label for s in train}) < 2:
            continue
        Xtr, ytr, _ = _xy(train)
        Xte, yte, _ = _xy(test)
        model = pick_model(train).fit(Xtr, ytr)
        out[held] = ranking_metrics(model.predict_proba(Xte), yte)
    return {"note": "only the fusion layer is learned; detectors are "
                    "deterministic and see no training data",
            "held_out": out}


# ---------------------------------------------------------------- E5
def unseen_generator(n_train: int = 240, n_test: int = 120, seed: int = 0
                     ) -> Dict[str, object]:
    """Train on generators A, B, C. Test on D, which was never seen.

    If performance collapses, the system learned our renderer rather than
    tampering, and every other number in the repo should be discounted.
    """
    train = build_dataset(n_train, seed=seed, generators=["A", "B", "C"])
    test_abc = build_dataset(n_test, seed=seed + 500, generators=["A", "B", "C"])
    test_d = build_dataset(n_test, seed=seed + 900, generators=["D"])

    Xtr, ytr, _ = _xy(train)
    model = pick_model(train).fit(Xtr, ytr)

    res = {}
    for name, ds in (("seen_ABC", test_abc), ("unseen_D", test_d)):
        X, y, _ = _xy(ds)
        res[name] = ranking_metrics(model.predict_proba(X), y)
    res["auc_drop"] = res["seen_ABC"]["roc_auc"] - res["unseen_D"]["roc_auc"]
    return res


# ---------------------------------------------------------------- E6
def strength_axes(n_per_cell: int = 12, seed: int = 0,
                  tamper_class: str = "digit_substitution") -> Dict[str, object]:
    """One curve per axis. Never one ladder.

    'Wrong font' and 'recompressed after the edit' are independent properties
    of an edit, not increasing points on a single scale. Plotting them as one
    curve would imply an ordering that does not exist.
    """
    from ..forensics.detectors import DETECTOR_NAMES

    def peak(s) -> float:
        return max(s.forensics.scores.get(f"{n}_fld", 0.0)
                   for n in DETECTOR_NAMES)

    axes = ["font_match", "colour_match", "compression_match", "rerender"]
    results: Dict[str, Dict[str, float]] = {}

    builder = SubmissionBuilder(seed=seed)
    baseline = [peak(builder.build(None)) for _ in range(n_per_cell)]
    ref = float(np.percentile(baseline, 90))

    for axis in axes:
        curve = {}
        for value in (0, 1):
            kwargs = {a: 1 for a in axes}
            kwargs[axis] = value
            b = SubmissionBuilder(seed=seed + value + 1)
            scores = []
            for _ in range(n_per_cell):
                spec = TamperSpec(tamper_class, **kwargs)
                scores.append(peak(b.build(tamper_class=tamper_class, spec=spec)))
            curve[f"{axis}={value}"] = {
                "mean_signal": float(np.mean(scores)),
                "detection_rate": float(np.mean([s > ref for s in scores])),
            }
        results[axis] = curve

    return {"clean_reference_p90": ref, "tamper_class": tamper_class,
            "axes": results}


# ---------------------------------------------------------------- E7
def degradation_curves(n_per_cell: int = 10, seed: int = 0) -> Dict[str, object]:
    """Sweep capture quality one variable at a time."""
    from ..forensics.detectors import DETECTOR_NAMES

    def peak(s) -> float:
        return max(s.forensics.scores.get(f"{n}_fld", 0.0)
                   for n in DETECTOR_NAMES)

    sweeps = {
        "jpeg_quality": [(("jpeg_quality", q),) for q in (95, 85, 75, 65)],
        "blur_sigma": [(("blur_sigma", b),) for b in (0.0, 0.5, 1.0, 1.6)],
        "scale": [(("scale", s),) for s in (1.0, 0.85, 0.7, 0.55)],
    }

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for axis, settings in sweeps.items():
        rows = {}
        for setting in settings:
            kw = dict(setting)
            cap = CaptureParams(**{**{"jpeg_quality": 92}, **kw})
            b_clean = SubmissionBuilder(seed=seed)
            b_tamp = SubmissionBuilder(seed=seed + 1)
            clean = [peak(b_clean.build(None, capture=cap))
                     for _ in range(n_per_cell)]
            tamp = [peak(b_tamp.build(tamper_class="digit_substitution",
                                      spec=TamperSpec("digit_substitution",
                                                      compression_match=0),
                                      capture=cap))
                    for _ in range(n_per_cell)]
            thr = float(np.percentile(clean, 90))
            rows[str(list(kw.values())[0])] = {
                "clean_p90": thr,
                "tampered_mean": float(np.mean(tamp)),
                "detection_rate": float(np.mean([t > thr for t in tamp])),
            }
        out[axis] = rows
    return out


# ---------------------------------------------------------------- E8
def ocr_sweep(rates: Sequence[float] = (0.0, 0.002, 0.005, 0.01, 0.02),
              n: int = 40, seed: int = 0) -> Dict[str, object]:
    """How much decision error does the reading stage cause on its own?"""
    out: Dict[str, Dict[str, float]] = {}
    for rate in rates:
        subs = build_dataset(n, seed=seed, ocr_error_rate=rate, tamper_rate=0.5)
        clean = [s for s in subs if s.label == 0]
        tampered = [s for s in subs if s.label == 1]
        out[f"{rate:.4f}"] = {
            "semantic_false_alarm_rate":
                float(np.mean([s.semantics.any_failure for s in clean]))
                if clean else float("nan"),
            "semantic_detection_rate":
                float(np.mean([s.semantics.any_failure for s in tampered]))
                if tampered else float("nan"),
            "mean_char_errors":
                float(np.mean([s.ocr.char_errors for s in subs])),
        }
    return out


# ---------------------------------------------------------------- E9
def calibration_and_cost(train, calib, test, cost: Optional[CostModel] = None,
                         prevalence: float = 0.05, seed: int = 0
                         ) -> Dict[str, object]:
    cost = cost or CostModel()
    test = resample_prevalence(test, prevalence, seed=seed)
    Xtr, ytr, _ = _xy(train)
    Xc, yc, _ = _xy(calib)
    Xte, yte, ete = _xy(test)

    model = pick_model(train).fit(Xtr, ytr)
    raw_te = model.predict_proba(Xte)
    cal = Calibrator("isotonic").fit(model.predict_proba(Xc), yc)
    cal_te = cal.transform(raw_te)

    ece_raw, _ = expected_calibration_error(raw_te, yte)
    ece_cal, bins = expected_calibration_error(cal_te, yte)

    best = optimal_threshold(cal_te, yte, ete, cost)
    curve = sweep_thresholds(cal_te, yte, ete, cost, n=40)

    return {
        "metrics_raw": ranking_metrics(raw_te, yte),
        "metrics_calibrated": ranking_metrics(cal_te, yte),
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "reliability_bins": bins,
        "optimal_operating_point": best,
        "coverage_curve": [
            {"threshold": r["threshold"], "coverage": r["coverage"],
             "recall": r["recall"], "precision": r["precision"],
             "net_saving": r["net_saving"]} for r in curve],
        "cost_model": cost.__dict__,
        "prevalence_used": prevalence,
        "n_test_after_resampling": int(len(test)),
        "note": "the test set is resampled to a realistic fraud prevalence "
                "before any cost conclusion; at the balanced 50 percent rate "
                "reviewing every case is trivially optimal and the threshold "
                "analysis would be meaningless",
    }


# ---------------------------------------------------------------- E10
def attribution_study(train, test, threshold: Optional[float] = None
                      ) -> Dict[str, object]:
    Xtr, ytr, _ = _xy(train)
    Xte, yte, _ = _xy(test)
    model = pick_model(train).fit(Xtr, ytr)
    oracle = OracleProfile.fit(Xtr)

    p = model.predict_proba(Xte)
    if threshold is None:
        clean_p = p[yte == 0]
        threshold = float(np.percentile(clean_p, 90)) if len(clean_p) else 0.5

    atts = attribute(test, Xte, yte, model.predict_proba, threshold, oracle)
    summary = summarise(atts, len(test))
    summary["threshold"] = threshold
    summary["examples"] = [
        {"doc_id": a.doc_id, "class": a.tamper_class, "label": a.label,
         "primary_cause": a.primary_cause, "all_causes": a.sufficient_causes}
        for a in atts[:12]]
    return summary


# ---------------------------------------------------------------- E11
def localisation_study(subs: Sequence) -> Dict[str, object]:
    rows: Dict[str, Dict[str, float]] = {}
    for cls in LOCAL_CLASSES:
        sel = [s for s in subs if s.tamper_class == cls
               and s.localisation.get("defined", 0.0) == 1.0]
        if not sel:
            continue
        rows[cls] = {
            "n": len(sel),
            "mean_iou": float(np.mean([s.localisation["iou"] for s in sel])),
            "pointing_accuracy":
                float(np.mean([s.localisation["pointing"] for s in sel])),
        }
    undefined = sorted({s.tamper_class for s in subs
                        if s.label == 1
                        and s.localisation.get("defined", 0.0) != 1.0})
    return {"by_class": rows,
            "undefined_for": undefined,
            "note": "reprint and full_forgery have no localisable edit region; "
                    "they are excluded rather than scored as misses"}
