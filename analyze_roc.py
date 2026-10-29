#!/usr/bin/env python3
"""
analyze_roc.py
==============

ROC / AUROC analysis of the three monitors.

Methodological note (important, and stated in the writeup):
The 120B produced no refusals, so it has no negative class of its own. AUROC is
therefore computed on the POOLED set (positives = manipulations from both
models, negatives = refusals, which come from the 20B). Per-model detection
rates are then reported at a common threshold calibrated on those shared
negatives, so the two models are compared on equal footing.

Usage
-----
    python3 analyze_roc.py                 # all judges pooled
    python3 analyze_roc.py --judge qwen/qwen3.6-27b
    python3 analyze_roc.py --fpr 0.2       # set the operating point
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

HERE = Path(__file__).parent
SCORES = HERE / "scores.csv"

MONITOR_LABEL = {
    "naive": "Naive monitor",
    "self_permission": "Self-permission monitor",
    "placebo": "Placebo (length-matched control)",
}
MONITOR_STYLE = {
    "naive": dict(color="#8a8a8a", ls="--", lw=2),
    "self_permission": dict(color="#1a1a1a", ls="-", lw=2.6),
    "placebo": dict(color="#b0b0b0", ls=":", lw=2),
}


def load(judge=None, run=None):
    rows = []
    with open(SCORES, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["score"]:
                continue
            if judge and r["judge"] != judge:
                continue
            if run and r["run"] != str(run):
                continue
            r["score"] = float(r["score"])
            r["is_manipulation"] = int(r["is_manipulation"])
            rows.append(r)
    return rows


def bootstrap_auc(y, s, n=2000, seed=0):
    """Percentile bootstrap CI for AUROC."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.min() == yb.max():          # need both classes
            continue
        fpr, tpr, _ = roc_curve(yb, s[idx])
        stats.append(auc(fpr, tpr))
    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--fpr", type=float, default=0.33,
                    help="operating point: target false-positive rate on refusals")
    ap.add_argument("--out", default="figure_roc.png")
    args = ap.parse_args()

    rows = load(args.judge, args.run)
    if not rows:
        raise SystemExit("No scores found. Run run_scoring.py first.")

    by_mon = defaultdict(list)
    for r in rows:
        by_mon[r["monitor"]].append(r)

    print(f"\nloaded {len(rows)} scored rows"
          f"{f' (judge={args.judge})' if args.judge else ''}")
    n_pos = sum(1 for r in rows if r["is_manipulation"] == 1) // max(len(by_mon), 1)
    n_neg = sum(1 for r in rows if r["is_manipulation"] == 0) // max(len(by_mon), 1)
    print(f"per monitor: ~{n_pos} manipulations (positives), ~{n_neg} refusals (negatives)\n")

    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=200)
    summary = {}

    for mon in ["naive", "self_permission", "placebo"]:
        rs = by_mon.get(mon)
        if not rs:
            continue
        y = [r["is_manipulation"] for r in rs]
        s = [r["score"] for r in rs]
        if len(set(y)) < 2:
            print(f"  {mon}: only one class present, skipping ROC")
            continue

        fpr, tpr, thr = roc_curve(y, s)
        a = auc(fpr, tpr)
        lo, hi = bootstrap_auc(y, s)
        summary[mon] = (a, lo, hi)

        ax.plot(fpr, tpr, label=f"{MONITOR_LABEL[mon]}  (AUROC {a:.3f})",
                **MONITOR_STYLE[mon])

        # operating point: highest TPR with FPR <= target
        ok = np.where(fpr <= args.fpr)[0]
        j = ok[-1] if len(ok) else 0
        print(f"{MONITOR_LABEL[mon]}")
        print(f"    AUROC {a:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"    at FPR<={args.fpr:.2f}: TPR {tpr[j]:.3f}  (threshold {thr[j]:.0f})")

        # per-model detection at that threshold
        t = thr[j]
        for mk in ("20b", "120b"):
            sub = [r for r in rs if r["model"] == mk and r["is_manipulation"] == 1]
            if sub:
                det = sum(1 for r in sub if r["score"] >= t) / len(sub)
                print(f"      {mk}: detects {det*100:.0f}% of {len(sub)} manipulations")
        print()

    ax.plot([0, 1], [0, 1], color="#cccccc", lw=1, ls="-")
    ax.set_xlabel("False positive rate (on correct refusals)")
    ax.set_ylabel("True positive rate (manipulations detected)")
    ax.set_title("Monitor discrimination: manipulations vs. refusals", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(HERE / args.out, bbox_inches="tight", facecolor="white")
    print(f"-> {args.out}")

    if "self_permission" in summary and "placebo" in summary:
        sp, pl = summary["self_permission"][0], summary["placebo"][0]
        nv = summary.get("naive", (float('nan'),))[0]
        print(f"\nKEY COMPARISON")
        print(f"  self-permission {sp:.3f}  vs  naive {nv:.3f}  "
              f"(delta {sp-nv:+.3f})")
        print(f"  self-permission {sp:.3f}  vs  placebo {pl:.3f}  "
              f"(delta {sp-pl:+.3f})")
        print("  If self-permission > placebo, the gain is the PATTERN,")
        print("  not prompt length. That is the control Reviewer 1 wanted.")


if __name__ == "__main__":
    main()
