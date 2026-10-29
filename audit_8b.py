#!/usr/bin/env python3
"""
audit_8b.py
===========

Checks the below-chance Llama 3.1 8B result before anyone treats it as a finding.
No API calls. Reads scores.csv and the trace files.

1. Parse check   how many 8B replies gave no usable score, split by refusal and
                 compliance. A big gap between the two would bias AUROC.
2. Word check    correlation between the 8B naive score and how many deception
                 words (misinformation, mislead, panic...) a trace contains,
                 within refusals and within compliances. A strong positive
                 correlation inside each class means the judge reacts to words.
3. Reading list  writes audit_8b_samples.csv: the 20 refusals the 8B scored
                 highest and the 20 compliances it scored lowest, with the full
                 reasoning, for each of you to read independently.

    python3 audit_8b.py
    python3 audit_8b.py --judge llama-3.1-8b-instant --monitor naive
"""

import argparse
import csv
import re

import numpy as np

from scored_monitor import is_refusal

DECEPTION = re.compile(r"misinformation|mislead|deceiv|decept|manipulat|sensational|"
                       r"fear|panic|disinformation|exaggerat", re.I)
FILES = {"20b": "traces_20b.csv", "120b": "traces_120b.csv"}


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="llama-3.1-8b-instant")
    ap.add_argument("--monitor", default="naive")
    ap.add_argument("--scores", default="scores.csv")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    traces = {}
    for mk, fn in FILES.items():
        for r in csv.DictReader(open(fn, encoding="utf-8")):
            traces[(mk, r["trial_id"])] = r

    rows = [r for r in csv.DictReader(open(args.scores, encoding="utf-8"))
            if r["judge"] == args.judge and r["monitor"] == args.monitor
            and (r["model"], r["trial_id"]) in traces]
    if not rows:
        raise SystemExit(f"No rows for judge={args.judge} monitor={args.monitor} in {args.scores}")

    print(f"\njudge={args.judge}  monitor={args.monitor}  rows={len(rows)}")
    print("\n1. PARSE CHECK")
    for cls, want in (("refusals", 1), ("compliances", 0)):
        sub = [r for r in rows if int(is_refusal(traces[(r["model"], r["trial_id"])]["raw_answer"])) == want]
        bad = sum(1 for r in sub if not r["score"])
        print(f"   {cls:<12} {bad}/{len(sub)} unparsed ({100*bad/max(len(sub),1):.0f}%)")

    print("\n2. WORD CHECK  (Spearman: score vs deception-word count, inside each class)")
    samples = []
    for cls, want in (("refusals", 1), ("compliances", 0)):
        sc, wc = [], []
        for r in rows:
            t = traces[(r["model"], r["trial_id"])]
            if int(is_refusal(t["raw_answer"])) != want or not r["score"]:
                continue
            sc.append(float(r["score"])); wc.append(len(DECEPTION.findall(t["raw_reasoning"])))
            samples.append((cls, float(r["score"]), wc[-1], r["model"], r["trial_id"], t["raw_reasoning"]))
        print(f"   {cls:<12} n={len(sc):>3}  mean score {np.mean(sc):5.1f}  "
              f"mean words {np.mean(wc):4.1f}  rho={spearman(sc, wc):+.2f}")

    ref = sorted([s for s in samples if s[0] == "refusals"], key=lambda s: -s[1])[: args.n]
    com = sorted([s for s in samples if s[0] == "compliances"], key=lambda s: s[1])[: args.n]
    out = "audit_8b_samples.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "score", "deception_words", "model", "trial_id", "reasoning",
                    "saumya_note", "mehek_note"])
        for s in ref:
            w.writerow(["refusal_scored_high", *s[1:], "", ""])
        for s in com:
            w.writerow(["compliance_scored_low", *s[1:], "", ""])
    print(f"\n3. READING LIST -> {out}  ({len(ref)} high-scored refusals, {len(com)} low-scored compliances)")
    print("   Each of you fills your own note column without looking at the other's.")


if __name__ == "__main__":
    main()
