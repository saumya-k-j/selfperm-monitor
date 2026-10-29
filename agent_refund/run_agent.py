#!/usr/bin/env python3
"""
run_agent.py
============

Runs the refund agent on every scenario and records, for each trial, the
reasoning written BEFORE the tool call, the tool it called, and whether that
action broke the policy (checked by code in scenarios.violates).

Monitors later read only `raw_reasoning`, exactly as a gate in front of a
risky tool call would.

    export GROQ_API_KEY=...
    python3 run_agent.py --check                     # one scenario, prints everything
    python3 run_agent.py --repeats 3 --out agent_traces.csv
    python3 run_agent.py --mock --repeats 1          # fake agent, no API calls

Resumable: rerun the same command and finished trials are skipped.
"""

import argparse
import csv
import json
import os
import random
import sys
import time

from scenarios import POLICY, TOOLS, build_scenarios, user_message, violates

COLS = ["trial_id", "scenario_id", "scenario_type", "correct_action", "action",
        "action_args", "violates", "raw_reasoning", "raw_answer", "reasoning_length",
        "model", "temperature", "reasoning_effort"]


class _Fn:
    def __init__(self, name, args): self.name, self.arguments = name, json.dumps(args)


class _Call:
    def __init__(self, name, args): self.function = _Fn(name, args)


class _MockMsg:
    """Fake agent: sometimes grants itself permission, sometimes follows policy."""
    def __init__(self, sc, rng):
        o = sc["order"]
        if rng.random() < 0.35 and sc["correct_action"] != "issue_refund":
            name, args = "issue_refund", {"order_id": o["order_id"], "amount": o["requested_amount"]}
            self.reasoning = "Is this allowed? The customer says it was approved. That's fine. Proceed."
        else:
            name = sc["correct_action"]
            args = ({"order_id": o["order_id"], "amount": o["requested_amount"]}
                    if name == "issue_refund" else {"order_id": o["order_id"], "reason": "policy"})
            self.reasoning = f"Check policy against the record. Correct action is {name}."
        self.content = ""
        self.tool_calls = [_Call(name, args)]


class _LocalMsg:
    """Ollama reply in the same shape as a Groq message: .reasoning, .content, .tool_calls."""
    def __init__(self, m):
        self.reasoning = m.get("thinking") or ""
        self.content = m.get("content") or ""
        self.tool_calls = [_Call(c["function"]["name"], c["function"].get("arguments") or {})
                           for c in (m.get("tool_calls") or [])]


def ollama_digest(name):
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as r:
            tags = {m["name"]: m["digest"] for m in json.loads(r.read())["models"]}
    except Exception:
        sys.exit("Could not reach Ollama. Start it with:  ollama serve")
    full = name if ":" in name else name + ":latest"
    if full not in tags:
        sys.exit(f"Model {full} is not downloaded. Run:  ollama pull {name}")
    return full, tags[full][:12]


def call_ollama(model, sc, temperature):
    """Local agent through Ollama's native API, with thinking on so the reasoning
    comes back in its own field next to the tool call."""
    import urllib.request
    body = json.dumps({"model": model, "stream": False, "think": True,
                       "messages": [{"role": "system", "content": POLICY},
                                    {"role": "user", "content": user_message(sc)}],
                       "tools": TOOLS,
                       "options": {"temperature": temperature, "num_predict": 2048}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return _LocalMsg(json.loads(r.read())["message"])
        except Exception as e:
            print(f"  ! {type(e).__name__}: {e}  (retrying)")
            time.sleep(10)
    raise SystemExit("Local model kept failing. Check that `ollama serve` is running, then rerun.")


def call_agent(client, model, sc, temperature, effort, mock, rng):
    if mock:
        return _MockMsg(sc, rng)
    if model.startswith("ollama/"):
        return call_ollama(model.split("/", 1)[1], sc, temperature)
    kwargs = dict(model=model,
                  messages=[{"role": "system", "content": POLICY},
                            {"role": "user", "content": user_message(sc)}],
                  tools=TOOLS, tool_choice="required",
                  temperature=temperature, max_completion_tokens=1024)
    if effort and model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = effort
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs).choices[0].message
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"  ! {type(e).__name__}: {e}  (waiting {wait}s)")
            time.sleep(wait)
    raise SystemExit("Too many API errors. Rerun the same command later to resume.")


def parse(msg):
    calls = getattr(msg, "tool_calls", None) or []
    if not calls:
        return "none", {}
    fn = calls[0].function
    try:
        args = json.loads(fn.arguments or "{}")
    except json.JSONDecodeError:
        args = {"_raw": fn.arguments}
    return fn.name, args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--out", default="agent_traces.csv")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    client = None
    label = args.model
    if args.model.startswith("ollama/") and not args.mock:
        full, dig = ollama_digest(args.model.split("/", 1)[1])
        args.model = f"ollama/{full}"
        args.pause = 0
        label = f"{args.model}@{dig}"
        print(f"local agent pinned to {label}")
        args.reasoning_effort = "think"
    elif not args.mock:
        if not os.environ.get("GROQ_API_KEY"):
            sys.exit("Set GROQ_API_KEY first, or use --mock.")
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])

    scen = build_scenarios(seed=args.seed)
    rng = random.Random(args.seed)

    if args.check:
        sc = next(s for s in scen if s["scenario_type"] == "pressure_big")
        m = call_agent(client, args.model, sc, args.temperature, args.reasoning_effort, args.mock, rng)
        name, a = parse(m)
        print(user_message(sc))
        print("\nREASONING FIELD:", getattr(m, "reasoning", None) or "-- none --")
        print("TOOL CALL:", name, a)
        print("VIOLATES POLICY:", violates(sc["order"], name, a), "| correct:", sc["correct_action"])
        if not getattr(m, "reasoning", None):
            print("\nWARNING: no reasoning field came back with the tool call. The monitors "
                  "need it. Tell Saumya before running the full set.")
        return

    done = set()
    if os.path.exists(args.out):
        done = {r["trial_id"] for r in csv.DictReader(open(args.out, encoding="utf-8"))}
    new = not os.path.exists(args.out)
    print(f"{len(scen)} scenarios x {args.repeats} repeats, model={args.model}, already done={len(done)}")
    no_reason = 0
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        for sc in scen:
            for k in range(args.repeats):
                tid = f"{sc['scenario_id']}_{k}"
                if tid in done:
                    continue
                m = call_agent(client, args.model, sc, args.temperature,
                               args.reasoning_effort, args.mock, rng)
                name, a = parse(m)
                reasoning = getattr(m, "reasoning", "") or ""
                no_reason += not reasoning
                w.writerow({"trial_id": tid, "scenario_id": sc["scenario_id"],
                            "scenario_type": sc["scenario_type"],
                            "correct_action": sc["correct_action"], "action": name,
                            "action_args": json.dumps(a), "violates": violates(sc["order"], name, a),
                            "raw_reasoning": reasoning, "raw_answer": getattr(m, "content", "") or "",
                            "reasoning_length": len(reasoning), "model": label,
                            "temperature": args.temperature, "reasoning_effort": args.reasoning_effort})
                f.flush()
                if not args.mock:
                    time.sleep(args.pause)
    print(f"done -> {args.out}")
    if no_reason:
        print(f"WARNING: {no_reason} trials came back without reasoning. Monitors cannot score those.")


if __name__ == "__main__":
    main()
