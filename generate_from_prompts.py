#!/usr/bin/env python3
"""
generate_from_prompts.py
========================

Generates subject-model traces from a prompt CSV (category, prompt_id, prompt,
repeats). Used for the harmless set. Unlike the original harness, it sets the
temperature and reasoning effort explicitly and writes them into every row, so
runs can be compared and reproduced.

Resumable: rerun the same command and finished trials are skipped.

    export GROQ_API_KEY=...
    python3 generate_from_prompts.py --check            # one call, shows the fields
    python3 generate_from_prompts.py --prompts harmless_prompts.csv \\
        --model openai/gpt-oss-120b --out harmless_traces.csv
    python3 generate_from_prompts.py ... --mock          # fake replies, no API calls

Output columns:
  trial_id, model, category, prompt_id, prompt, raw_reasoning, raw_answer,
  reasoning_length, temperature, reasoning_effort
"""

import argparse
import csv
import os
import sys
import time

COLS = ["trial_id", "model", "category", "prompt_id", "prompt", "raw_reasoning",
        "raw_answer", "reasoning_length", "temperature", "reasoning_effort"]


class _MockMsg:
    def __init__(self, p):
        self.reasoning = f"The user asks: {p[:60]}... This is fine. Let's answer."
        self.content = "Mock answer."


def call(client, model, prompt, temperature, effort, max_tokens, mock):
    if mock:
        return _MockMsg(prompt)
    kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}],
                  max_completion_tokens=max_tokens, temperature=temperature)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="harmless_prompts.csv")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--out", default="harmless_traces.csv")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--repeats", type=int, default=None, help="override the CSV's repeats column")
    ap.add_argument("--check", action="store_true", help="one call, print fields, stop")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    client = None
    if not args.mock:
        if not os.environ.get("GROQ_API_KEY"):
            sys.exit("Set GROQ_API_KEY first, or use --mock.")
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])

    if args.check:
        m = call(client, args.model, "Summarize in one sentence: The library extended its hours.",
                 args.temperature, args.reasoning_effort, 300, args.mock)
        print("REASONING FIELD:", getattr(m, "reasoning", None) or "-- none --")
        print("CONTENT:", m.content)
        return

    prompts = list(csv.DictReader(open(args.prompts, encoding="utf-8")))
    done = set()
    if os.path.exists(args.out):
        done = {r["trial_id"] for r in csv.DictReader(open(args.out, encoding="utf-8"))}
    new = not os.path.exists(args.out)
    total = sum(args.repeats or int(p["repeats"]) for p in prompts)
    print(f"{len(prompts)} prompts -> {total} trials, model={args.model}, "
          f"T={args.temperature}, effort={args.reasoning_effort}, already done={len(done)}")

    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        n = 0
        for p in prompts:
            for k in range(args.repeats or int(p["repeats"])):
                tid = f"{p['prompt_id']}_{k}"
                if tid in done:
                    continue
                m = call(client, args.model, p["prompt"], args.temperature,
                         args.reasoning_effort, args.max_tokens, args.mock)
                reasoning = getattr(m, "reasoning", "") or ""
                w.writerow({"trial_id": tid, "model": args.model, "category": p["category"],
                            "prompt_id": p["prompt_id"], "prompt": p["prompt"],
                            "raw_reasoning": reasoning, "raw_answer": m.content or "",
                            "reasoning_length": len(reasoning),
                            "temperature": args.temperature,
                            "reasoning_effort": args.reasoning_effort})
                f.flush()
                n += 1
                if n % 10 == 0:
                    print(f"  {n} new trials written")
                if not args.mock:
                    time.sleep(args.pause)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
