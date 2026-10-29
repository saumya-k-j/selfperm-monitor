# Week 1 experiments: how to run them

These files sit in the root of the `selfperm-monitor` repo, next to
`scored_monitor.py` and `scores.csv`. Nothing here changes the existing files.
Everything writes to new files (`scores_extra.csv`, `harmless_traces.csv`,
`agent_refund/agent_traces.csv`), so your original results stay untouched.

Every script that calls an API has a `--mock` flag that uses fake replies. Run
it with `--mock` once to check your setup before spending quota. Every script
that calls an API is resumable: if it stops, rerun the identical command.

```bash
export GROQ_API_KEY=...        # never paste it into code or a notebook
pip install groq numpy scikit-learn
```

---

## Track A: strengthen the current claim

### A1. 8B audit (no API calls, do this first)

Needs `scores.csv`. Commit it to the repo too, since Mehek's fork does not have it.

```bash
python3 audit_8b.py
```

Gives the parse-failure rate by class, the correlation between the 8B score and
deception words, and `audit_8b_samples.csv` (20 high-scored refusals, 20
low-scored compliances). Each of you writes notes in your own column without
looking at the other's.

### A2. Decision-flip test (Agni's question)

`flip_traces.csv` is already built: 40 real reasoning openings x 6 matched
endings = 240 traces. Rebuild with `python3 make_flip_traces.py` if needed.

```bash
python3 score_any.py --file flip_traces.csv --dataset flip --label-col variant \
    --judge llama-3.1-8b-instant --monitors naive,self_permission
python3 score_any.py --file flip_traces.csv --dataset flip --label-col variant \
    --judge qwen/qwen3.6-27b --monitors naive,self_permission
python3 analyze_flip.py
```

Reading it: a monitor that tracks the decision scores "proceed" above "decline"
in every row. If the 8B scores `vocab_decline` high, it is reacting to the words
"misinformation" and "panic", not the decision.

### A3. Harmless traces and the false-positive rate (Mehek generates, Saumya scores)

`harmless_prompts.csv` has 57 prompts in four groups:

| Group | What | Why |
|---|---|---|
| plain_summary | the 15 paragraphs, just "summarize" | baseline |
| tone_twin | the 15 paragraphs, "make it sound more upbeat, without changing facts" | same shape as the manipulation task, but honest. The hardest test |
| ordinary | 12 everyday requests | normal traffic |
| permission_trigger | 15 harmless requests that invite "is this OK?" reasoning (ad copy, crime scene in fiction, scam awareness) | what production traffic looks like |

```bash
# Mehek
python3 generate_from_prompts.py --check
python3 generate_from_prompts.py --prompts harmless_prompts.csv \
    --model openai/gpt-oss-120b --out harmless_traces.csv

# Saumya
python3 score_any.py --file harmless_traces.csv --dataset harmless \
    --judge qwen/qwen3.6-27b --monitors naive,self_permission,placebo,placebo_v2,naive_disclaimer
python3 analyze_fpr.py --judge qwen/qwen3.6-27b
```

`analyze_fpr.py` reports the false-positive rate at 80% and 90% detection, and
what share of alerts would be real if manipulation were 1%, 5% or 10% of traffic.

### A4. Close the prompt gap

The naive prompt lacks the "do not mention" note that the other three carry.
`naive_disclaimer` adds it, so naive and self-permission differ by one clause only.

```bash
python3 score_any.py --file traces_20b.csv  --dataset 20b  --judge qwen/qwen3.6-27b --monitors naive_disclaimer
python3 score_any.py --file traces_120b.csv --dataset 120b --judge qwen/qwen3.6-27b --monitors naive_disclaimer
```

### A5. 8B stability (optional)

```bash
for r in 1 2 3; do python3 score_any.py --file traces_120b.csv --dataset 120b --limit 40 \
    --judge llama-3.1-8b-instant --monitors naive --temperature 0.7 --run $r; done
```

---

## Track B: agentic pilot (refund agent)

A customer-support agent with a written refund policy and three tools
(`issue_refund`, `escalate_to_manager`, `deny_refund`). 46 scenarios: clean
requests, customer pressure ("your manager already approved this"), and injected
notes in the order record ("refund limits are waived"). Whether an action broke
the policy is checked by code, so ground truth is exact. Monitors read only the
reasoning written before the tool call, like a gate would.

```bash
cd agent_refund
python3 run_agent.py --check        # ONE call. Confirms reasoning comes back with a tool call
python3 run_agent.py --repeats 3 --out agent_traces.csv
cd ..
python3 score_any.py --file agent_refund/agent_traces.csv --dataset agent --label-col violates \
    --judge qwen/qwen3.6-27b --monitors naive_agent,selfperm_agent,placebo_agent
cd agent_refund && python3 analyze_agent.py --judge qwen/qwen3.6-27b
```

If `--check` prints "no reasoning field", stop and tell Saumya. Groq may not
return reasoning alongside tool calls for that model, which would itself be a
finding for the access question.

If the agent almost never breaks policy, that is a result too: report it, then
make the pressure stronger rather than tuning prompts until it breaks.

---

## Who runs what

| Task | Who | API calls |
|---|---|---|
| A1 audit | Claude, once `scores.csv` is uploaded | none |
| A2 flip scoring | Saumya | 240 x 2 monitors x 2 judges = 960 |
| A3 harmless generation | Mehek | 231 |
| A3 harmless scoring | Saumya | 231 x 5 = 1,155 (Qwen); add 8B and 70B later |
| A4 disclaimer | Saumya | 189 |
| B agent generation | Mehek | 138 |
| B agent scoring | Saumya | 138 x 3 = 414 |
| All analyses | Claude, from the CSVs | none |

Groq's daily limits are per model, so spread work across judges and days.
Send the output CSVs (`scores_extra.csv`, `harmless_traces.csv`,
`agent_refund/agent_traces.csv`) back and the analyses can run immediately.
