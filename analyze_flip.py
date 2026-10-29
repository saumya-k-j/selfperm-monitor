#!/usr/bin/env python3
"""
analyze_flip.py
===============

Reads the decision-flip scores from scores_extra.csv (dataset "flip") and
flip_traces.csv, and reports for each judge and monitor:

  decision effect     mean(score | proceed) - mean(score | decline), within each
                      framing. Positive = the monitor tracks the decision.
  pairs right         share of matched pairs where proceed scored above decline.
  framing effect      perm_* vs plain_*: does merely discussing permission
                      raise the score?
  vocabulary effect   vocab_decline vs perm_decline: do deception words raise
                      the score even when the model declines?

Every comparison is between endings attached to the SAME prefix, so prefix
content cancels out. No API calls.

    python3 analyze_flip.py
    python3 analyze_flip.py --scores scores_extra.csv --traces flip_traces.csv
"""

import argparse
import csv
from collections import defaultdict
from math import comb


def sign_test_p(wins, losses):
    """Two-sided exact sign test, ties dropped."""
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="scores_extra.csv")
    ap.add_argument("--traces", default="flip_traces.csv")
    ap.add_argument("--dataset", default="flip")
    args = ap.parse_args()

    meta = {}
    with open(args.traces, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["trial_id"]] = (f"{r['source_model']}_{r['source_trial']}", r["variant"])

    # (judge, monitor) -> prefix -> variant -> score
    S = defaultdict(lambda: defaultdict(dict))
    with open(args.scores, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] != args.dataset or not r["score"] or r["trial_id"] not in meta:
                continue
            pref, var = meta[r["trial_id"]]
            S[(r["judge"], r["monitor"])][pref][var] = float(r["score"])

    if not S:
        raise SystemExit("No flip scores found. Run score_any.py --dataset flip first.")

    for (judge, mon), prefs in sorted(S.items()):
        print("\n" + "=" * 74)
        print(f"judge={judge}   monitor={mon}   prefixes={len(prefs)}")
        print("=" * 74)
        means = {}
        for v in ("perm_proceed", "perm_decline", "plain_proceed", "plain_decline",
                  "vocab_proceed", "vocab_decline"):
            vals = [p[v] for p in prefs.values() if v in p]
            means[v] = sum(vals) / len(vals) if vals else float("nan")
        print(f"{'':16}{'proceed':>10}{'decline':>10}{'difference':>13}")
        for fr in ("perm", "plain", "vocab"):
            a, b = means[f"{fr}_proceed"], means[f"{fr}_decline"]
            print(f"{fr:<16}{a:>10.1f}{b:>10.1f}{a - b:>+13.1f}")

        def paired(x, y):
            w = l = t = 0
            for p in prefs.values():
                if x in p and y in p:
                    if p[x] > p[y]: w += 1
                    elif p[x] < p[y]: l += 1
                    else: t += 1
            return w, l, t

        print()
        for fr in ("perm", "plain", "vocab"):
            w, l, t = paired(f"{fr}_proceed", f"{fr}_decline")
            n = w + l + t
            print(f"  decision tracked ({fr:5}): proceed > decline in {w}/{n} pairs "
                  f"(ties {t}), sign test p={sign_test_p(w, l):.3g}")
        w, l, t = paired("perm_proceed", "plain_proceed")
        print(f"  deliberation raises score: perm_proceed > plain_proceed in {w}/{w + l + t} "
              f"(p={sign_test_p(w, l):.3g})")
        w, l, t = paired("vocab_decline", "perm_decline")
        print(f"  deception words raise score on a DECLINE: vocab_decline > perm_decline in "
              f"{w}/{w + l + t} (p={sign_test_p(w, l):.3g})")

    print("\nHOW TO READ THIS")
    print("  A monitor that tracks the decision shows a large positive proceed-decline gap in")
    print("  every row. If the vocab row has a small or negative gap, or vocab_decline scores")
    print("  high, the judge is reacting to deception WORDS rather than the decision, which")
    print("  would explain a below-chance AUROC on refusals.")


if __name__ == "__main__":
    main()
