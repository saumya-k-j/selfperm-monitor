#!/usr/bin/env python3
"""
make_label_sheets.py
====================

Builds blind labelling sheets for the agent violations, following
progress/labelling_protocol.md.

Reads tau2_traces.csv (from extract_tau2.py) and writes, for each labeller, a CSV
with only what the protocol allows them to see: the conversation context is not
included here (open the tau2 run with `tau2 view` if needed), but the reasoning
and the tool call are. No monitor scores, no model name, no task purpose.

Items are shuffled with a different seed per labeller so order cannot be compared.
A separate key file maps item ids back to trace ids, for joining later.

    python3 make_label_sheets.py --traces tau2_traces.csv --labellers saumya,mehek
"""

import argparse
import csv
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="tau2_traces.csv")
    ap.add_argument("--labellers", default="saumya,mehek")
    ap.add_argument("--all-writes", action="store_true",
                    help="include correct write actions too (default: violations only)")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.traces, encoding="utf-8"))
            if args.all_writes or r["unexpected"] == "1"]
    for i, r in enumerate(rows):
        r["item_id"] = f"item{i + 1:03d}"

    with open("label_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "trial_id"])
        for r in rows:
            w.writerow([r["item_id"], r["trial_id"]])

    for k, name in enumerate(args.labellers.split(",")):
        items = rows[:]
        random.Random(1000 + k).shuffle(items)
        out = f"label_sheet_{name}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "tool_call", "tool_args", "agent_reasoning",
                        "label (SP/MI/NR/OT)", "uncertain (y/n)", "note"])
            for r in items:
                w.writerow([r["item_id"], r["tool"], r["tool_args"], r["raw_reasoning"], "", "", ""])
        print(f"{len(items)} items -> {out}")
    print("key -> label_key.csv (do not open until both labellers are done)")


if __name__ == "__main__":
    main()
