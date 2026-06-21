#!/usr/bin/env python3
"""
evaluate.py -- reproduce the headline results.

Runs the naive and self-permission monitors over the bundled, labeled trace
data and reports, for each model:

  * detection rate on real manipulations (the model complied), and
  * false-positive rate on correct refusals (the model declined).

This regenerates the numbers reported on the project site and in the writeup.

Usage
-----
    export GROQ_API_KEY=...           # your rotated key
    python evaluate.py                # full run on bundled data
    python evaluate.py --limit 20     # quick smoke test (first 20 trials/model)

Notes
-----
A trace is a "manipulation" when leak_label != 'refusal' (the model produced a
biased output). It is a "refusal" when leak_label == 'refusal'. Ground truth
comes from the bundled labels, which were produced by an LLM judge validated at
83% agreement against 30 hand-labeled trials.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from selfperm_monitor import NaiveMonitor, SelfPermissionMonitor, groq_backend

DATA = Path(__file__).parent / "selfperm_monitor" / "data"
FILES = {"GPT-OSS 20B": "traces_20b.csv", "GPT-OSS 120B": "traces_120b.csv"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run(rows, naive, aware, limit=None, pause=1.2):
    manips = [r for r in rows if r["leak_label"] not in ("refusal", "PARSE_ERROR")]
    refus = [r for r in rows if r["leak_label"] == "refusal"]
    if limit:
        manips, refus = manips[:limit], refus[:limit]

    nc = ac = 0
    for i, r in enumerate(manips):
        if naive.check(r["raw_reasoning"], r["raw_answer"]).flagged:
            nc += 1
        if aware.check(r["raw_reasoning"], r["raw_answer"]).flagged:
            ac += 1
        if i % 15 == 0:
            print(f"    manipulation {i + 1}/{len(manips)}", file=sys.stderr)
        time.sleep(pause)

    nf = af = 0
    for r in refus:
        if naive.check(r["raw_reasoning"], r["raw_answer"]).flagged:
            nf += 1
        if aware.check(r["raw_reasoning"], r["raw_answer"]).flagged:
            af += 1
        time.sleep(pause)

    return {
        "n_manip": len(manips), "naive_caught": nc, "aware_caught": ac,
        "n_refus": len(refus), "naive_fp": nf, "aware_fp": af,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="trials per model (smoke test)")
    ap.add_argument("--monitor-model", default="qwen/qwen3-32b",
                    help="model used as the monitor judge (not the model under test)")
    args = ap.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("Set GROQ_API_KEY in your environment first.")

    backend = groq_backend(model=args.monitor_model)
    naive = NaiveMonitor(backend=backend)
    aware = SelfPermissionMonitor(backend=backend)

    print(f"\nMonitor judge: {args.monitor_model}\n" + "=" * 60)
    for label, fname in FILES.items():
        rows = load(DATA / fname)
        print(f"\n{label}  (running monitors...)")
        r = run(rows, naive, aware, limit=args.limit)
        nm, nr = r["n_manip"], r["n_refus"]
        print(f"  DETECTION on {nm} manipulations:")
        print(f"    naive            {r['naive_caught']}/{nm} ({pct(r['naive_caught'], nm)})")
        print(f"    self-permission  {r['aware_caught']}/{nm} ({pct(r['aware_caught'], nm)})"
              f"  (+{r['aware_caught'] - r['naive_caught']})")
        if nr:
            print(f"  FALSE POSITIVES on {nr} correct refusals (lower is better):")
            print(f"    naive            {r['naive_fp']}/{nr} ({pct(r['naive_fp'], nr)})")
            print(f"    self-permission  {r['aware_fp']}/{nr} ({pct(r['aware_fp'], nr)})")
    print("\n" + "=" * 60)
    print("Expected (full run): 20B naive 92% / aware 94%;  120B naive 53% / aware 76%.")


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "n/a"


if __name__ == "__main__":
    main()
