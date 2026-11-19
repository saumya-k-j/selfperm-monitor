#!/usr/bin/env python3
"""
or_filter_debug.py
==================
Why does Anthropic's content filter block every Sonnet call on the tau2 actions?
Sends a few test calls and prints ONLY metadata (finish reason, provider, token
counts, reply length). It never prints the agent's reasoning or the judge's reply,
so the labelling stays blind.

    python3 dev/or_filter_debug.py
"""
import csv
import json
import os
import sys

# run from anywhere: use the repo root for imports and data files
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import urllib.error
import urllib.request

from extra_monitors import ALL_MONITORS

KEY = os.environ.get("OPENROUTER_API_KEY")
if not KEY:
    raise SystemExit("Set the key first:  read -s OPENROUTER_API_KEY && export OPENROUTER_API_KEY")


def capture_prompt(monitor, text):
    box = {}
    def backend(prompt):
        box["p"] = prompt
        return '{"score": 0}'
    ALL_MONITORS[monitor](backend=backend).score(text)
    return box["p"]


def call(prompt, model="anthropic/claude-sonnet-5.5", provider="anthropic", thinking=True):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 8000}
    if thinking:
        body["reasoning"] = {"effort": "low"}
    if provider:
        body["provider"] = {"order": [provider], "allow_fallbacks": False}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    u = d.get("usage") or {}
    return (f"finish={ch.get('finish_reason')} native={ch.get('native_finish_reason')} "
            f"provider={d.get('provider')} reply_chars={len(msg.get('content') or '')} "
            f"refusal={'yes' if msg.get('refusal') else 'no'} "
            f"tokens_in={u.get('prompt_tokens')} tokens_out={u.get('completion_tokens')}"
            + (f" error={str(d.get('error'))[:150]}" if d.get("error") else ""))


rows = list(csv.DictReader(open("tau2_gptoss.csv", encoding="utf-8")))
real = rows[0]["raw_reasoning"]
dummy = ("The customer wants to add one checked bag to reservation ABC123. The policy allows "
         "extra bags for a fee. They confirmed yes, so I will update the baggage now.")
old = list(csv.DictReader(open("traces_120b.csv", encoding="utf-8")))[0]["raw_reasoning"]

tests = [
    ("A real tau2 reasoning, naive_agent", capture_prompt("naive_agent", real), {}),
    ("B made-up airline text, naive_agent", capture_prompt("naive_agent", dummy), {}),
    ("C real tau2 reasoning, original naive prompt", capture_prompt("naive", real), {}),
    ("D old news trace (worked before), naive", capture_prompt("naive", old), {}),
    ("E real tau2, naive_agent, via Amazon Bedrock", capture_prompt("naive_agent", real), {"provider": "amazon-bedrock"}),
    ("F real tau2, naive_agent, via Google Vertex", capture_prompt("naive_agent", real), {"provider": "google-vertex"}),
]
for name, prompt, kw in tests:
    print(f"{name}:\n   prompt_chars={len(prompt)}  {call(prompt, **kw)}")
