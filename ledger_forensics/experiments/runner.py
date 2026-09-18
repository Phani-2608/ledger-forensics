"""Run the experiment suite and write artifacts.

    python run.py demo    small, a couple of minutes, proves the whole thing works
    python run.py full    the numbers reported in the README
    python run.py smoke   tiny, used by CI
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore")

from ..pipeline import build_dataset
from . import cache, suite

PROFILES = {
    "smoke": {"n_main": 40, "n_gen_train": 40, "n_gen_test": 20,
              "n_cell": 4, "n_ocr": 16},
    "demo": {"n_main": 220, "n_gen_train": 140, "n_gen_test": 70,
             "n_cell": 10, "n_ocr": 40},
    "full": {"n_main": 600, "n_gen_train": 400, "n_gen_test": 200,
             "n_cell": 20, "n_ocr": 80},
}


def _j(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


STAGES = ["data", "core", "holdout", "curves", "decide"]


def _load_results(out: Path) -> Dict[str, object]:
    path = out / "results.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def run(profile: str = "demo", outdir: str = "artifacts", seed: int = 0,
        stages: Optional[List[str]] = None) -> Dict[str, object]:
    """Run the suite, optionally one stage at a time.

    The full sweep takes longer than a single session on a small machine, so
    stages are resumable: the built dataset is cached and results.json is
    merged rather than overwritten.
    """
    cfg = PROFILES[profile]
    stages = stages or STAGES
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    results = _load_results(out)
    results.update({"profile": profile, "config": cfg, "seed": seed})

    cache_path = out / f"dataset_{profile}_{seed}.pkl"
    subs = cache.build_or_load(cache_path, cfg["n_main"], seed=seed,
                               tamper_rate=0.5, progress=True)
    train, calib, test = suite.split(subs, frac=0.6, seed=seed)
    print(f"dataset: {len(subs)} submissions "
          f"(train {len(train)} / calib {len(calib)} / test {len(test)})",
          flush=True)

    if "core" in stages:
        print("E1 baseline ladder", flush=True)
        results["E1_baseline_ladder"] = suite.baseline_ladder(train, test)
        print("E2 channel confusion matrix", flush=True)
        results["E2_channel_matrix"] = suite.channel_matrix(subs)
        print("E3 per-class detection / E11 localisation", flush=True)
        results["E3_per_class"] = suite.per_class_detection(train, test)
        results["E11_localisation"] = suite.localisation_study(subs)
        _save(results, out)

    if "holdout" in stages:
        print("E4 leave-one-class-out", flush=True)
        results["E4_leave_one_class_out"] = suite.leave_one_class_out(subs)
        _save(results, out)
        print("E5 unseen generator D", flush=True)
        results["E5_unseen_generator"] = suite.unseen_generator(
            n_train=cfg["n_gen_train"], n_test=cfg["n_gen_test"], seed=seed)
        _save(results, out)

    if "curves" in stages:
        print("E6 strength axes", flush=True)
        results["E6_strength_axes"] = suite.strength_axes(
            n_per_cell=cfg["n_cell"], seed=seed)
        _save(results, out)
        print("E7 degradation curves", flush=True)
        results["E7_degradation"] = suite.degradation_curves(
            n_per_cell=max(4, cfg["n_cell"] // 2), seed=seed)
        _save(results, out)
        print("E8 OCR sweep", flush=True)
        results["E8_ocr_sweep"] = suite.ocr_sweep(n=cfg["n_ocr"], seed=seed)
        _save(results, out)

    if "decide" in stages:
        print("E9 calibration and cost", flush=True)
        results["E9_calibration_and_cost"] = suite.calibration_and_cost(
            train, calib, test)
        print("E10 error attribution", flush=True)
        results["E10_attribution"] = suite.attribution_study(train, test)
        _save(results, out)

    results["runtime_seconds"] = round(
        results.get("runtime_seconds", 0.0) + time.time() - t0, 1)
    _save(results, out)
    _write_summary(results, out / "SUMMARY.md")

    try:
        from .plots import make_plots
        make_plots(results, out)
        print(f"wrote plots to {out}")
    except Exception as exc:
        print(f"(plots skipped: {exc})")

    print(f"wrote {out / 'results.json'}")
    return results


def _save(results: Dict[str, object], out: Path) -> None:
    (out / "results.json").write_text(json.dumps(results, indent=2, default=_j))


def _write_summary(r: Dict[str, object], path: Path) -> None:
    L = ["# Results", "",
         f"Profile `{r['profile']}`, seed {r['seed']}, "
         f"{r['runtime_seconds']}s.", ""]

    L += ["## E1 Model ladder", "",
          "| model | ROC AUC | PR AUC | Brier |", "|---|---|---|---|"]
    for name, m in r["E1_baseline_ladder"].items():
        L.append(f"| {name} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} "
                 f"| {m['brier']:.3f} |")

    cm = r.get("E2_channel_matrix", {})
    if cm.get("share"):
        L += ["", "## E2 Which channel catches what", "",
              "| outcome | share of tampered documents |", "|---|---|"]
        for k, v in cm["share"].items():
            L.append(f"| {k} | {v:.1%} |")

    pc = r.get("E3_per_class", {}).get("by_class", {})
    if pc:
        L += ["", "## E3 Detection by tamper class", "",
              "| class | n | detection rate |", "|---|---|---|"]
        for k, v in pc.items():
            rate = v.get("detection_rate", v.get("false_alarm_rate"))
            tag = "" if "detection_rate" in v else " (false alarm)"
            L.append(f"| {k} | {v['n']} | {rate:.1%}{tag} |")

    ug = r.get("E5_unseen_generator", {})
    if ug:
        L += ["", "## E5 Unseen generator", "",
              f"Seen A/B/C ROC AUC {ug['seen_ABC']['roc_auc']:.3f}, "
              f"unseen D {ug['unseen_D']['roc_auc']:.3f}, "
              f"drop {ug['auc_drop']:.3f}."]

    cal = r.get("E9_calibration_and_cost", {})
    if cal:
        op = cal["optimal_operating_point"]
        L += ["", "## E9 Calibration and cost", "",
              f"ECE before calibration {cal['ece_raw']:.3f}, "
              f"after {cal['ece_calibrated']:.3f}.",
              f"Best operating point: threshold {op['threshold']:.2f}, "
              f"review {op['coverage']:.1%} of cases, recall {op['recall']:.1%}, "
              f"net saving {op['net_saving']:.0f}."]

    at = r.get("E10_attribution", {})
    if at.get("primary_cause_share"):
        L += ["", "## E10 Which stage caused the errors", "",
              f"{at['n_errors']} errors out of {at['n_total']} "
              f"({at['error_rate']:.1%}).", "",
              "| primary cause | share of errors |", "|---|---|"]
        for k, v in sorted(at["primary_cause_share"].items(),
                           key=lambda kv: -kv[1]):
            L.append(f"| {k} | {v:.1%} |")

    path.write_text("\n".join(L) + "\n")
