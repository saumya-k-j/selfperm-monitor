#!/usr/bin/env python3
"""
flip_paired.py: pre-registered test (progress/preregistration_xf_flip.md).
Per prefix, decision effect = score(proceed) - score(decline); compares the
self-permission monitor with naive and placebo by sign test, per judge, framing
and source family.

    cd ~/selfperm-monitor && python3 flip_paired.py
"""
import csv, collections, numpy as np, sys
from math import comb
U = sys.argv[1] if len(sys.argv)>1 else ""
meta={r["trial_id"]:r for r in csv.DictReader(open(U+"flip_xf_traces.csv"))}
S=collections.defaultdict(dict)
for r in csv.DictReader(open(U+"scores_extra.csv")):
    if r["dataset"]=="flip_xf" and r["score"]!="" and r["trial_id"] in meta:
        S[(r["judge"],r["monitor"])][r["trial_id"]]=float(r["score"])
def sign_p(w,l):
    n=w+l
    if n==0: return float("nan")
    k=min(w,l); return min(1.0,2*sum(comb(n,i) for i in range(k+1))/2**n)
J={"Llama 8B":"ollama/llama3.1:8b@46e0c10c039e","Gemma 12B":"ollama/gemma3:12b@b92b218582c5","Sonnet 5.5":"openrouter/anthropic/claude-sonnet-5.5@google-vertex"}
for jn,j in J.items():
    for frame in ["perm","plain"]:
        eff={}
        for mon in ["naive","self_permission","placebo"]:
            d=S.get((j,mon),{}); e={}
            for t,m in meta.items():
                if m["variant"]==f"{frame}_proceed":
                    dec=t.replace("_proceed","_decline")
                    if t in d and dec in d: e[m["source_model"]+"|"+m["source_trial"]]=d[t]-d[dec]
            eff[mon]=e
        sp=eff["self_permission"]
        if not sp: continue
        print(f"{jn:<10} {frame:<5} n={len(sp):>2}  mean decision effect: " + "  ".join(f"{m} {np.mean(list(eff[m].values())):+5.1f}" for m in eff if eff[m]))
        for o in ["naive","placebo"]:
            k=[x for x in sp if x in eff[o]]
            w=sum(sp[x]>eff[o][x] for x in k); l=sum(sp[x]<eff[o][x] for x in k)
            print(f"           self_perm effect > {o:<8}: {w} vs {l} (ties {len(k)-w-l}), sign p={sign_p(w,l):.2g}")
    # per family, perm frame, self_perm vs naive
    for fam in ["xfm_deepseek","xfm_qwen","xfm_glm"]:
        out=[]
        for mon in ["naive","self_permission","placebo"]:
            d=S.get((j,mon),{}); v=[]
            for t,m in meta.items():
                if m["variant"]=="perm_proceed" and m["source_model"]==fam:
                    dec=t.replace("_proceed","_decline")
                    if t in d and dec in d: v.append(d[t]-d[dec])
            if v: out.append(f"{mon} {np.mean(v):+5.1f} (n={len(v)})")
        if out: print(f"           {fam:<13} perm: "+"  ".join(out))
