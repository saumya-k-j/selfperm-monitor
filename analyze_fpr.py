#!/usr/bin/env python3
"""
analyze_fpr.py
==============

The production question: at a detection rate a team would want, how often does
the monitor fire on harmless traffic, and what share of its alerts are real?

Detection side: compliant manipulations from scores.csv (models 20b and 120b,
is_manipulation = 1), one judge and monitor at a time.
False-positive side: harmless traces from scores_extra.csv (dataset "harmless").
If harmless_traces.csv has a `category` column, results are also split by it.

For each judge x monitor:
  1. Find the score threshold that catches 80% (and 90%) of manipulations.
  2. Report the false-positive rate on harmless traces at that threshold,
     with a 95% bootstrap interval.
  3. Report precision (share of alerts that are real) if manipulation made up
     1%, 5% or 10% of traffic:
         precision = TPR*p / (TPR*p + FPR*(1-p))

No API calls.

    python3 analyze_fpr.py --judge qwen/qwen3.6-27b
    python3 analyze_fpr.py --judge qwen/qwen3.6-27b --models 120b
"""

import argparse
import csv
from collections import defaultdict

import numpy as np


def threshold_for_tpr(pos, target):
    """Highest threshold t with share(pos >= t) >= target."""
    pos = np.sort(np.asarray(pos, float))[::-1]
    k = int(np.ceil(target * len(pos))) - 1
    return pos[max(k, 0)]


def boot_fpr(neg, t, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    neg = np.asarray(neg, float)
    vals = [np.mean(neg[rng.integers(0, len(neg), len(neg))] >= t) for _ in range(n)]
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def precision(tpr, fpr, p):
    d = tpr * p + fpr * (1 - p)
    return tpr * p / d if d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--scores", default="scores.csv")
    ap.add_argument("--extra", default="scores_extra.csv")
    ap.add_argument("--harmless-dataset", default="harmless")
    ap.add_argument("--harmless-traces", default="harmless_traces.csv")
    ap.add_argument("--models", default="20b,120b")
    ap.add_argument("--monitors", default="naive,self_permission,placebo,placebo_v2,naive_disclaimer")
    ap.add_argument("--targets", default="0.8,0.9")
    args = ap.parse_args()

    models = set(args.models.split(","))
    pos = defaultdict(list)
    for r in csv.DictReader(open(args.scores, encoding="utf-8")):
        if (r["judge"] == args.judge and r["model"] in models and r["score"]
                and int(r["is_manipulation"]) == 1):
            pos[r["monitor"]].append(float(r["score"]))
    # naive_disclaimer on the original traces lives in scores_extra (dataset = model key)
    for r in csv.DictReader(open(args.extra, encoding="utf-8")):
        if (r["judge"] == args.judge and r["dataset"] in models and r["score"]
                and r["monitor"] not in pos and r.get("is_refusal") == "0"):
            pos[r["monitor"] + "*"].append(float(r["score"]))
    for k in [k for k in pos if k.endswith("*")]:
        pos[k[:-1]] = pos.pop(k)

    cat = {}
    try:
        for r in csv.DictReader(open(args.harmless_traces, encoding="utf-8")):
            cat[r["trial_id"]] = r.get("category", "all")
    except FileNotFoundError:
        pass

    neg = defaultdict(list)
    neg_cat = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(args.extra, encoding="utf-8")):
        if r["dataset"] == args.harmless_dataset and r["judge"] == args.judge and r["score"]:
            s = float(r["score"])
            neg[r["monitor"]].append(s)
            neg_cat[r["monitor"]][cat.get(r["trial_id"], "all")].append(s)

    targets = [float(x) for x in args.targets.split(",")]
    print(f"\njudge={args.judge}   detection set: compliant manipulations from {sorted(models)}")
    print("=" * 96)
    hdr = f"{'monitor':<18}{'n pos':>6}{'n harmless':>11}"
    for t in targets:
        hdr += f"{f'FPR@{int(t*100)}%':>16}"
    hdr += f"{'prec@1%':>10}{'prec@5%':>10}{'prec@10%':>10}"
    print(hdr)
    print("-" * 96)
    for mon in args.monitors.split(","):
        P, N = pos.get(mon, []), neg.get(mon, [])
        if not P or not N:
            print(f"{mon:<18}  (missing: {'positives' if not P else 'harmless scores'})")
            continue
        line = f"{mon:<18}{len(P):>6}{len(N):>11}"
        prec_cells = ""
        for i, tgt in enumerate(targets):
            t = threshold_for_tpr(P, tgt)
            tpr = float(np.mean(np.asarray(P) >= t))
            fpr = float(np.mean(np.asarray(N) >= t))
            lo, hi = boot_fpr(N, t)
            line += f"{f'{100*fpr:.0f}% [{100*lo:.0f},{100*hi:.0f}]':>16}"
            if i == len(targets) - 1:
                prec_cells = "".join(f"{100*precision(tpr, fpr, p):>9.0f}%" for p in (0.01, 0.05, 0.10))
        print(line + prec_cells)

    print(f"\nPrecision columns use the threshold for {int(targets[-1]*100)}% detection.")
    if any(len(v) > 1 for v in neg_cat.values()):
        print("\nFalse-positive rate by harmless category, at the 90% detection threshold")
        for mon in args.monitors.split(","):
            if mon not in neg_cat or mon not in pos:
                continue
            t = threshold_for_tpr(pos[mon], targets[-1])
            parts = [f"{c}: {100*np.mean(np.asarray(v) >= t):.0f}% (n={len(v)})"
                     for c, v in sorted(neg_cat[mon].items())]
            print(f"  {mon:<18}" + "   ".join(parts))


if __name__ == "__main__":
    main()
