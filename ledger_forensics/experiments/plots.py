"""Figures for the README. Matplotlib only, no seaborn, no styling tricks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_plots(r: Dict[str, object], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _ladder(r, out)
    _per_class(r, out)
    _reliability(r, out)
    _coverage(r, out)
    _attribution(r, out)
    _ocr(r, out)


def _ladder(r, out):
    d = r.get("E1_baseline_ladder", {})
    if not d:
        return
    names = list(d)
    auc = [d[n]["roc_auc"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(names, auc, color="#3b6ea5")
    ax.axvline(0.5, color="#999", ls="--", lw=1, label="chance")
    ax.set_xlim(0, 1)
    ax.set_xlabel("ROC AUC")
    ax.set_title("Model ladder: does fusion earn its complexity?")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_ladder.png", dpi=140)
    plt.close(fig)


def _per_class(r, out):
    d = r.get("E3_per_class", {}).get("by_class", {})
    if not d:
        return
    items = [(k, v) for k, v in d.items() if "detection_rate" in v]
    if not items:
        return
    items.sort(key=lambda kv: kv[1]["detection_rate"])
    names = [k for k, _ in items]
    vals = [v["detection_rate"] for _, v in items]
    colors = ["#b5433a" if v < 0.3 else "#c9873a" if v < 0.7 else "#3f7d4f"
              for v in vals]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(names, vals, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("detection rate at a fixed 10% clean false-alarm budget")
    ax.set_title("What is catchable, and what is not")
    fig.tight_layout()
    fig.savefig(out / "fig_per_class.png", dpi=140)
    plt.close(fig)


def _reliability(r, out):
    cal = r.get("E9_calibration_and_cost", {})
    bins = cal.get("reliability_bins") or []
    pts = [(b["confidence"], b["accuracy"]) for b in bins
           if b["n"] > 0 and np.isfinite(b["confidence"])]
    if not pts:
        return
    xs, ys = zip(*pts)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#999", label="perfect")
    ax.plot(xs, ys, "o-", color="#3b6ea5", label="calibrated")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed fraud rate")
    ax.set_title(f"Reliability (ECE {cal.get('ece_calibrated', 0):.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_reliability.png", dpi=140)
    plt.close(fig)


def _coverage(r, out):
    curve = r.get("E9_calibration_and_cost", {}).get("coverage_curve") or []
    if not curve:
        return
    cov = [c["coverage"] for c in curve]
    net = [c["net_saving"] for c in curve]
    rec = [c["recall"] for c in curve]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(cov, net, color="#3f7d4f", label="net saving")
    ax.set_xlabel("share of cases sent to review")
    ax.set_ylabel("net saving")
    ax.axhline(0, color="#999", lw=1)
    ax2 = ax.twinx()
    ax2.plot(cov, rec, color="#b5433a", ls="--", label="recall")
    ax2.set_ylabel("recall")
    ax.set_title("What each review budget buys")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_coverage.png", dpi=140)
    plt.close(fig)


def _attribution(r, out):
    share = r.get("E10_attribution", {}).get("primary_cause_share") or {}
    if not share:
        return
    items = sorted(share.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(names, vals, color="#6a5a8c")
    ax.set_xlabel("share of wrong decisions")
    ax.set_title("Which stage is actually responsible")
    fig.tight_layout()
    fig.savefig(out / "fig_attribution.png", dpi=140)
    plt.close(fig)


def _ocr(r, out):
    d = r.get("E8_ocr_sweep", {})
    if not d:
        return
    rates = sorted(float(k) for k in d)
    fa = [d[f"{x:.4f}"]["semantic_false_alarm_rate"] for x in rates]
    dr = [d[f"{x:.4f}"]["semantic_detection_rate"] for x in rates]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(rates, fa, "o-", color="#b5433a", label="false alarms on clean")
    ax.plot(rates, dr, "s-", color="#3f7d4f", label="detection on tampered")
    ax.set_xlabel("OCR character error rate")
    ax.set_ylabel("semantic channel fire rate")
    ax.set_title("The reading stage sets the semantic error floor")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_ocr.png", dpi=140)
    plt.close(fig)
