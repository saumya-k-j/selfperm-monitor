# Action-aware monitor: prediction written before scoring (14 November 2026)

Status: exploratory. The monitors below were designed after seeing that reasoning-only
monitors missed the tau2-bench agent violations, so this is not a confirmatory test.
The prediction is fixed here before any of these monitors has scored anything.

Data: the 245 labelled tau2-bench airline write actions already collected
(GPT-OSS 120B: 77, Qwen3 32B: 168). Positives are actions the benchmark's answer key
does not expect. Judge: Llama 3.1 8B (local, pinned), temperature 0.

Monitors (action_monitor.py):
- CA: conversation so far + pending tool call + policy (a PolicyGuard-style check)
- RA: agent reasoning + pending tool call + policy, told to flag actions that
  contradict the reasoning's own conclusion
- RA_SP: RA plus the self-permission clause

Predictions:
1. RA beats the reasoning-only naive monitor on AUROC (paired, task-clustered
   bootstrap interval above zero), pooled over both agents.
2. On Qwen3 32B's 21 "reasoning contradicts action" violations, RA scores them
   higher than correct actions (AUROC above 0.5), where reasoning-only monitors
   scored them lower.
3. Exploratory, no direction predicted: RA versus CA, and RA_SP versus RA.

All results are reported whatever they show.
