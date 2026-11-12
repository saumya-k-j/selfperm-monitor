#!/usr/bin/env python3
"""
make_label_sheets.py
====================

Builds blind labelling sheets for the agent violations, following
progress/labelling_protocol.md.

Reads tau2_traces.csv (from extract_tau2.py) and writes, for each labeller, a CSV
with only what the protocol allows them to see:
  - the conversation up to that turn (customer, agent and tool results), rebuilt
    from the tau2-bench results files when --results is given
  - the agent's reasoning in that turn
  - the tool call
No monitor scores, no model name, no task purpose, no answer key.
The airline policy (the same for every item) is written once to label_policy.md.

Items are shuffled with a different seed per labeller so order cannot be compared.
A separate key file maps item ids back to trace ids, for joining later.

    python3 make_label_sheets.py --traces tau2_gptoss.csv \
        --results "$HOME/tau2-bench/data/simulations/gptoss_*/results.json" \
        --labellers saumya,mehek
"""

import argparse
import csv
import glob
import json
import random

MAX_CONTEXT = 15000   # characters of conversation kept per item (most recent part)
MAX_TOOL = 800        # characters kept from each tool result


def load_conversations(pattern):
    convs, policy = {}, ""
    for path in sorted(glob.glob(pattern)):
        d = json.load(open(path, encoding="utf-8"))
        run = path.split("/")[-2]
        policy = policy or ((d.get("info") or {}).get("environment_info") or {}).get("policy", "")
        for s in d.get("simulations", []):
            convs[(run, str(s["task_id"]), str(s.get("trial", 0)))] = s.get("messages", [])
    return convs, policy


def render(messages, upto_turn):
    lines = []
    for m in messages:
        if m.get("turn_idx") is not None and m["turn_idx"] >= upto_turn:
            break
        role = m.get("role")
        if role == "user":
            lines.append(f"CUSTOMER: {m.get('content') or ''}")
        elif role == "assistant":
            if m.get("content"):
                lines.append(f"AGENT: {m['content']}")
            for c in m.get("tool_calls") or []:
                name = c.get("name") or (c.get("function") or {}).get("name")
                a = c.get("arguments") or (c.get("function") or {}).get("arguments")
                lines.append(f"AGENT CALLS TOOL: {name} {a if isinstance(a, str) else json.dumps(a)}")
        elif role == "tool":
            txt = m.get("content") or ""
            if len(txt) > MAX_TOOL:
                txt = txt[:MAX_TOOL] + " ...[cut]"
            lines.append(f"TOOL RESULT: {txt}")
    text = "\n\n".join(lines)
    return text if len(text) <= MAX_CONTEXT else "...[earlier conversation cut]\n\n" + text[-MAX_CONTEXT:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="tau2_traces.csv")
    ap.add_argument("--results", default=None, help="glob of tau2 results.json files, for the conversation")
    ap.add_argument("--labellers", default="saumya,mehek")
    ap.add_argument("--all-writes", action="store_true",
                    help="include correct write actions too (default: violations only)")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.traces, encoding="utf-8"))
            if args.all_writes or r["unexpected"] == "1"]
    for i, r in enumerate(rows):
        r["item_id"] = f"item{i + 1:03d}"

    convs, policy = load_conversations(args.results) if args.results else ({}, "")
    missing = 0
    for r in rows:
        run, task, trial, turn = r["trial_id"].split("|")[:4]
        msgs = convs.get((run, task.replace("task", ""), trial.replace("trial", "")))
        if msgs is None:
            missing += 1
            r["context"] = ""
        else:
            r["context"] = render(msgs, int(turn.replace("turn", "")))
    if args.results and missing:
        print(f"WARNING: no conversation found for {missing} items")
    if policy:
        open("label_policy.md", "w", encoding="utf-8").write(policy)
        print("policy -> label_policy.md (read this once before labelling)")

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
            w.writerow(["item_id", "conversation_so_far", "agent_reasoning", "tool_call", "tool_args",
                        "label (SP/MI/NR/OT)", "uncertain (y/n)", "note"])
            for r in items:
                w.writerow([r["item_id"], r.get("context", ""), r["raw_reasoning"], r["tool"],
                            r["tool_args"], "", "", ""])
        print(f"{len(items)} items -> {out}")
    print("key -> label_key.csv (do not open until both labellers are done)")


if __name__ == "__main__":
    main()
