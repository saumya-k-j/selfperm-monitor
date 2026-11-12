#!/usr/bin/env python3
"""
score_any.py
============

Score ANY trace file with any monitors and any judge. Resumable: rerun the same
command after a crash or a rate limit and it continues where it stopped.

Writes to scores_extra.csv (never touches scores.csv), one row per
(dataset, trial, monitor, judge, temperature, run).

The input CSV needs `trial_id` and `raw_reasoning`. Optional label column:
  --label-col NAME   copy that column into `label` (e.g. `violates`, `variant`)
If the file has `raw_answer`, an `is_refusal` column is also written.

Examples
--------
    # check setup with fake scores, no API calls
    python3 score_any.py --file harmless_traces.csv --dataset harmless --mock --limit 5

    # harmless traces, all four original monitors, one judge
    python3 score_any.py --file harmless_traces.csv --dataset harmless \\
        --judge qwen/qwen3.6-27b --monitors naive,self_permission,placebo,placebo_v2

    # decision-flip traces
    python3 score_any.py --file flip_traces.csv --dataset flip --label-col variant \\
        --judge llama-3.1-8b-instant --monitors naive,self_permission

    # naive prompt with the disclaimer, on the original traces
    python3 score_any.py --file traces_120b.csv --dataset 120b \\
        --judge qwen/qwen3.6-27b --monitors naive_disclaimer

    # 8B stability check at temperature 0.7
    python3 score_any.py --file traces_120b.csv --dataset 120b --limit 40 \\
        --judge llama-3.1-8b-instant --monitors naive --temperature 0.7 --run 1
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from scored_monitor import is_refusal
from extra_monitors import ALL_MONITORS, groq_backend_t, mock_backend, ollama_backend, openrouter_backend

OUT_DEFAULT = "scores_extra.csv"
FIELDS = ["dataset", "trial_id", "monitor", "judge", "temperature", "run",
          "score", "label", "is_refusal", "raw_judge_reply"]


def load_done(out):
    if not out.exists():
        return set()
    with open(out, encoding="utf-8") as f:
        return {(r["dataset"], r["trial_id"], r["monitor"], r["judge"],
                 r["temperature"], r["run"]) for r in csv.DictReader(f)}


def main():
    if "--list-models" in sys.argv:
        if not os.environ.get("GROQ_API_KEY"):
            sys.exit("Set the key first:  read -s GROQ_API_KEY && export GROQ_API_KEY")
        from groq import Groq
        ids = sorted(m.id for m in Groq().models.list().data)
        print("Models your key can use:\n  " + "\n  ".join(ids))
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--dataset", required=True, help="short name stored in every row")
    ap.add_argument("--judge", default="qwen/qwen3.8-27b",
                    help="Groq model id, ollama/<name> for a local model (e.g. ollama/llama3.1:8b), "
                         "or openrouter/<model> (e.g. openrouter/anthropic/claude-sonnet-5.5)")
    ap.add_argument("--provider", default=None,
                    help="OpenRouter only: pin one upstream provider, e.g. anthropic")
    ap.add_argument("--monitors", default="naive,self_permission,placebo,placebo_v2")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--run", default="1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pause", type=float, default=2.5)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--mock", action="store_true", help="fake scores, no API calls")
    ap.add_argument("--record-blocked", action="store_true",
                    help="OpenRouter only: if the provider's content filter blocks a reply, save the "
                         "row with an empty score and the note [blocked] instead of retrying later")
    args = ap.parse_args()

    names = [m for m in args.monitors.split(",") if m]
    unknown = [m for m in names if m not in ALL_MONITORS]
    if unknown:
        sys.exit(f"Unknown monitors: {unknown}. Known: {sorted(ALL_MONITORS)}")

    or_info = {}
    if args.mock:
        backend, judge = mock_backend(), "MOCK"
    elif args.judge.startswith("ollama/"):
        name = args.judge.split("/", 1)[1]
        # Pin the exact weights: store the model's digest in the judge name, so a
        # later re-download of a different build can never be mixed up with this one.
        import json, urllib.request
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as r:
                tags = {m["name"]: m["digest"] for m in json.loads(r.read())["models"]}
        except Exception:
            sys.exit("Could not reach Ollama. Open the Ollama app first.")
        full = name if ":" in name else name + ":latest"
        if full not in tags:
            sys.exit(f"Model {full} is not downloaded. Run:  ollama pull {name}")
        backend = ollama_backend(full, args.temperature)
        judge = f"ollama/{full}@{tags[full][:12]}"
        print(f"local judge pinned to {judge}")
        args.pause = 0
    elif args.judge.startswith("openrouter/"):
        if not os.environ.get("OPENROUTER_API_KEY"):
            sys.exit("Set the key first:  read -s OPENROUTER_API_KEY && export OPENROUTER_API_KEY")
        model = args.judge.split("/", 1)[1]
        if model.startswith("anthropic/"):
            args.temperature = 1.0   # thinking models only run at the default; record that honestly
        or_info = {}
        backend = openrouter_backend(model, args.temperature, provider=args.provider, info=or_info)
        judge = args.judge + (f"@{args.provider}" if args.provider else "")
        args.pause = 3.2   # new OpenRouter accounts: 20 requests a minute
    else:
        if not os.environ.get("GROQ_API_KEY"):
            sys.exit("Set GROQ_API_KEY first (export GROQ_API_KEY=...). Or use --mock.")
        backend, judge = groq_backend_t(args.judge, args.temperature), args.judge
    monitors = {n: ALL_MONITORS[n](backend=backend) for n in names}

    with open(args.file, encoding="utf-8") as f:
        traces = list(csv.DictReader(f))
    if args.limit:
        traces = traces[: args.limit]
    for need in ("trial_id", "raw_reasoning"):
        if need not in traces[0]:
            sys.exit(f"{args.file} has no `{need}` column")

    out = Path(args.out)
    done = load_done(out)
    temp = f"{args.temperature:g}"
    new = not out.exists()
    fh = open(out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()

    scored = skipped = failed = unparsed = nojson = blocked = 0
    print(f"{args.dataset}: {len(traces)} traces x {len(monitors)} monitors, judge={judge}, T={temp}")
    for i, tr in enumerate(traces):
        for mname, mon in monitors.items():
            key = (args.dataset, tr["trial_id"], mname, judge, temp, str(args.run))
            if key in done:
                skipped += 1
                continue
            try:
                s = mon.score(tr["raw_reasoning"])
            except Exception as e:
                msg = str(e)
                if "model_not_found" in msg or "does not exist" in msg or "decommissioned" in msg:
                    fh.close()
                    sys.exit(f"\nSTOPPED: Groq says the judge model `{judge}` is not available to "
                             f"your key.\nRun  python3 score_any.py --list-models  to see which "
                             f"models you can use.")
                if judge.startswith("ollama/") and ("Connection refused" in msg or "urlopen error" in msg
                                                   or "not found" in msg.lower()):
                    fh.close()
                    sys.exit("\nSTOPPED: could not reach the local model. Open the Ollama app, "
                             f"then run:  ollama pull {judge.split('/', 1)[1]}")
                if msg.startswith("BLOCKED") and args.record_blocked:
                    blocked += 1
                    w.writerow({
                        "dataset": args.dataset, "trial_id": tr["trial_id"], "monitor": mname,
                        "judge": judge, "temperature": temp, "run": args.run, "score": "",
                        "label": tr.get(args.label_col, "") if args.label_col else "",
                        "is_refusal": int(is_refusal(tr["raw_answer"])) if "raw_answer" in tr else "",
                        "raw_judge_reply": "[blocked by provider content filter]",
                    })
                    fh.flush()
                    print(f"  x {tr['trial_id']}/{mname}: blocked by content filter, recorded")
                    continue
                if judge.startswith("openrouter/") and ("401" in msg or "402" in msg):
                    fh.close()
                    sys.exit("\nSTOPPED: OpenRouter rejected the key or the spending cap is used up "
                             f"({scored} scored this run).\n" + msg[:200])
                if "invalid_api_key" in msg or "401" in msg:
                    fh.close()
                    sys.exit("\nSTOPPED: the API key was rejected. Set it again with:\n"
                             "  read -s GROQ_API_KEY && export GROQ_API_KEY")
                low = msg.lower()
                if "429" in msg and ("per day" in low or "tpd" in low or "rpd" in low):
                    fh.close()
                    sys.exit(f"\nSTOPPED: Groq's daily limit for `{judge}` is used up "
                             f"({scored} scored this run). Rerun the same command later; "
                             f"it continues where it stopped.")
                failed += 1
                print(f"  ! {tr['trial_id']}/{mname}: {type(e).__name__}: {str(e)[:160]} (rerun to retry)")
                time.sleep(3 if ("incomplete reply" in msg or msg.startswith("BLOCKED")) else 30)
                continue
            if judge.startswith("openrouter/") and scored == 0 and or_info:
                print(f"  OpenRouter served: model={or_info.get('model')} provider={or_info.get('provider')}")
            if judge.startswith("openrouter/") and s.ok and '"score"' not in s.raw:
                nojson += 1   # score came from the first-number fallback, so check it by hand
                print(f"  ? {tr['trial_id']}/{mname}: no JSON score in the reply, fallback used")
            if not s.ok:
                unparsed += 1
            w.writerow({
                "dataset": args.dataset, "trial_id": tr["trial_id"], "monitor": mname,
                "judge": judge, "temperature": temp, "run": args.run,
                "score": s.value if s.ok else "",
                "label": tr.get(args.label_col, "") if args.label_col else "",
                "is_refusal": int(is_refusal(tr["raw_answer"])) if "raw_answer" in tr else "",
                "raw_judge_reply": s.raw[:(2000 if judge.startswith("openrouter/") else 300)].replace("\n", " "),
            })
            fh.flush()
            scored += 1
            if not args.mock:
                time.sleep(args.pause)
        if i % 10 == 0:
            print(f"  {i + 1}/{len(traces)}  scored={scored} skipped={skipped}")
    fh.close()
    print(f"done. scored={scored} skipped={skipped} failed={failed} unparsed={unparsed} -> {out}")
    if blocked:
        print(f"  {blocked} replies were blocked by the provider's content filter and recorded as [blocked].")
    if nojson:
        print(f"  {nojson} replies had no JSON score; their score is the first number in the text. Check them.")
    if unparsed:
        print(f"  {unparsed} judge replies had no usable score. They are kept with an empty score.")


if __name__ == "__main__":
    main()
