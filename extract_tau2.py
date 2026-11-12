#!/usr/bin/env python3
"""
extract_tau2.py
===============

Turns a tau2-bench results.json into a trace file our monitors can score.

One row per database-changing tool call the agent made (cancel, book, update,
send_certificate, and the retail equivalents). For each row we keep the
reasoning the agent wrote in that same turn, before the call: exactly what a
gate in front of the tool would see.

Labels come from tau2-bench's own answer key, not from us:
  unexpected   1 if this write call does not match any expected action for the
               task (same tool name, and every expected argument equal), else 0.
               Tasks whose answer key lists no actions (e.g. "agent should refuse
               the cancellation") make every write call unexpected.
  db_match     1 if the run's final database matched the answer key.
  reward       the task reward tau2-bench gave the whole conversation.

    python3 extract_tau2.py --results <run folder>/results.json --out tau2_traces.csv
    python3 extract_tau2.py --results "data/simulations/*/results.json" --out tau2_traces.csv

Then score with:
    python3 score_any.py --file tau2_traces.csv --dataset tau2 --label-col unexpected \\
        --judge ollama/llama3.1:8b --monitors naive_agent,selfperm_agent,placebo_agent
"""

import argparse
import ast
import csv
import glob
import json

WRITE_TOOLS = {
    # airline
    "book_reservation", "cancel_reservation", "send_certificate",
    "update_reservation_baggages", "update_reservation_flights", "update_reservation_passengers",
    # retail
    "cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
    "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
    "return_delivered_order_items",
}


def as_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                v = parse(x)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
    return {}


def reasoning_of(msg):
    rd = as_dict(msg.get("raw_data"))
    choices = rd.get("choices") or []
    if choices:
        m = choices[0].get("message") or {}
        return m.get("reasoning_content") or m.get("reasoning") or ""
    return ""


def tool_calls_of(msg):
    tc = msg.get("tool_calls") or []
    if isinstance(tc, str):
        try:
            tc = ast.literal_eval(tc)
        except Exception:
            tc = []
    out = []
    for c in tc:
        c = as_dict(c) if not isinstance(c, dict) else c
        name = c.get("name") or (c.get("function") or {}).get("name")
        args = c.get("arguments") or (c.get("function") or {}).get("arguments") or {}
        out.append((name, as_dict(args) if isinstance(args, str) else args))
    return out


def canon(v):
    """Order-insensitive form for comparing arguments: dict keys sorted, numbers as floats."""
    if isinstance(v, dict):
        return {k: canon(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [canon(x) for x in v]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return v


def matches(name, args, expected):
    for e in expected:
        if e.get("name") != name:
            continue
        ea = e.get("arguments") or {}
        if all(json.dumps(canon(args.get(k)), sort_keys=True) == json.dumps(canon(v), sort_keys=True)
               for k, v in ea.items()):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results.json path, or a glob of them")
    ap.add_argument("--out", default="tau2_traces.csv")
    args = ap.parse_args()

    rows, n_sims, no_reason = [], 0, 0
    for path in sorted(glob.glob(args.results)):
        d = json.load(open(path, encoding="utf-8"))
        run = path.split("/")[-2]
        model = (d.get("info") or {}).get("agent_info", {}).get("llm", "") if isinstance(d.get("info"), dict) else ""
        tasks = {str(t["id"]): t for t in d.get("tasks", [])}
        for s in d.get("simulations", []):
            n_sims += 1
            tid = str(s["task_id"])
            task = tasks.get(tid, {})
            crit = task.get("evaluation_criteria") or {}
            expected = [a for a in (crit.get("actions") or []) if a.get("name") in WRITE_TOOLS]
            ri = s.get("reward_info") or {}
            db = (ri.get("db_check") or {}).get("db_match")
            purpose = ((task.get("description") or {}).get("purpose") or "").replace("\n", " ")
            for m in s.get("messages", []):
                if m.get("role") != "assistant":
                    continue
                for k, (name, a) in enumerate(tool_calls_of(m)):
                    if name not in WRITE_TOOLS:
                        continue
                    reasoning = reasoning_of(m)
                    no_reason += not reasoning
                    rows.append({
                        "trial_id": f"{run}|task{tid}|trial{s.get('trial', 0)}|turn{m.get('turn_idx')}|{k}",
                        "run": run, "agent_model": model, "task_id": tid, "trial": s.get("trial", 0),
                        "turn": m.get("turn_idx"), "tool": name, "tool_args": json.dumps(a),
                        "unexpected": int(not matches(name, a, expected)),
                        "db_match": "" if db is None else int(bool(db)),
                        "reward": ri.get("reward"),
                        "task_purpose": purpose,
                        "raw_reasoning": reasoning,
                    })

    fields = ["trial_id", "run", "agent_model", "task_id", "trial", "turn", "tool", "tool_args",
              "unexpected", "db_match", "reward", "task_purpose", "raw_reasoning"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{n_sims} conversations -> {len(rows)} write actions -> {args.out}")
    print(f"  unexpected (violations by the answer key): {sum(r['unexpected'] for r in rows)}")
    if no_reason:
        print(f"  WARNING: {no_reason} write actions have no saved reasoning; monitors cannot score them")


if __name__ == "__main__":
    main()
