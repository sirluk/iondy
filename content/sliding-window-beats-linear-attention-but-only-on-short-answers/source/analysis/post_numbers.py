"""Every number the post cites for the four matched-memory students.

Usage (from anywhere):
    python -I post_numbers.py

Reads data/memory_matched_per_problem.csv and data/memory_matched_math_completions.csv
(written from the evaluation output bundles by load_samples.py, math_rollouts.py and
write_tables.py; the bundles themselves are not included) and writes data/post_numbers.json,
which make_figures.py reads.

Main comparison: Bifocal (kv448) against SWA with a learned sink (swa576). Both use learned
scalar sinks, so they differ only in how the 576 tokens of cache per layer are spent.
swa572 uses the SWA paper's 4 sink tokens. Our eval code mishandles sink tokens when the
prompt is longer than the cache, so its scores here cover only prompts that fit: the math
benchmarks (prompts under 200 tokens) and 154 of HumanEval's 164 prompts. The post reports
only its math scores, as its "What we left out" note explains.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
ARMS = ["swa576", "swa572", "kv448", "query64"]
MAIN = ("kv448", "swa576")
MATH = ["aime2024", "aime2025", "hmmt_202502", "hmmt_202511"]
SHORT = ["gsm8k", "humaneval", "ifeval"]
CAP = 31744 - 8  # re-tokenized text can be a few tokens shorter than the generated sequence
CACHE = 576  # tokens of KV cache per layer, all four students
LONG = 4000
LAYERS = 36

prob = defaultdict(dict)  # (arm, bench) -> doc_hash -> row
for r in csv.DictReader(open(DATA / "memory_matched_per_problem.csv")):
    r["score"] = float(r["score"])
    r["prompt_tokens"] = int(r["prompt_tokens"]) if r["prompt_tokens"] else None
    prob[(r["arm"], r["benchmark"])][r["doc_hash"]] = r
comps = [dict(r, tokens=int(r["tokens"]), correct=int(r["correct"])) for r in csv.DictReader(open(DATA / "memory_matched_math_completions.csv"))]

rng = np.random.default_rng(0)


def problems(bench, arms):
    """Problems to compare. With the sink-token student, only prompts that fit in its cache."""
    hashes = set.intersection(*(set(prob[(a, bench)]) for a in arms))
    if "swa572" in arms:
        hashes = {h for h in hashes if prob[("swa572", bench)][h]["prompt_tokens"] is None
                  or prob[("swa572", bench)][h]["prompt_tokens"] <= CACHE}
    return sorted(hashes)


def score(arm, bench, arms=None):
    hashes = problems(bench, arms or [arm])
    return 100 * float(np.mean([prob[(arm, bench)][h]["score"] for h in hashes])) if hashes else None


def paired(a1, a2, benches, B=20000):
    diffs = [np.array([prob[(a1, b)][h]["score"] - prob[(a2, b)][h]["score"] for h in problems(b, [a1, a2])]) for b in benches]
    if any(len(d) == 0 for d in diffs):
        return None  # e.g. GSM8K with the sink-token student: no prompt fits in its cache
    point = np.mean([d.mean() for d in diffs])
    boots = np.mean([d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1) for d in diffs], axis=0)
    perm = np.mean([(d * rng.choice((-1.0, 1.0), size=(5000, len(d)))).mean(axis=1) for d in diffs], axis=0)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"diff": 100 * point, "ci": [100 * lo, 100 * hi], "p": float(np.mean(np.abs(perm) >= abs(point) - 1e-12)),
            "better": int(sum((d > 0).sum() for d in diffs)), "worse": int(sum((d < 0).sum() for d in diffs)),
            "n": int(sum(len(d) for d in diffs))}


out = {"scores": {}, "median_tokens": {}, "n_problems": {}, "paired": {}, "math": {}}
for b in MATH + SHORT:
    out["n_problems"][b] = len(problems(b, ["swa576"]))
    for a in ARMS:
        out["scores"][f"{a}/{b}"] = score(a, b, ["swa572"] if a == "swa572" else None)
    if b in MATH:
        lens = [c["tokens"] for c in comps if c["benchmark"] == b and c["arm"] in MAIN]
    else:
        lens = [float(prob[(a, b)][h]["median_tokens"]) for a in MAIN for h in problems(b, list(MAIN))]
    out["median_tokens"][b] = float(np.median(lens))
out["n_problems"]["humaneval_fit"] = len(problems("humaneval", ["swa572"]))
for a in ARMS:
    out["scores"][f"{a}/math"] = float(np.mean([out["scores"][f"{a}/{b}"] for b in MATH]))
pairs = [MAIN, ("kv448", "swa572"), ("swa576", "swa572"), ("kv448", "query64"), ("swa576", "query64")]
for a1, a2 in pairs:
    out["paired"][f"{a1}-{a2}/math"] = paired(a1, a2, MATH)
    for b in MATH + SHORT:
        r = paired(a1, a2, [b])
        if r:
            out["paired"][f"{a1}-{a2}/{b}"] = r

# Math completions: caps, lengths, where the correct answers are.
for a in ARMS:
    c = [x for x in comps if x["arm"] == a]
    done = [x for x in c if x["tokens"] < CAP]
    short = [x for x in c if x["tokens"] < LONG]
    long_ = [x for x in c if x["tokens"] >= LONG]
    window = {"swa576": 576, "swa572": 572, "kv448": 448, "query64": 64}[a]
    out["math"][a] = {
        "completions": len(c),
        "capped_share": 100 * (len(c) - len(done)) / len(c),
        "accuracy_finished": 100 * float(np.mean([x["correct"] for x in done])),
        "median_tokens": float(np.median([x["tokens"] for x in c])),
        "median_tokens_finished": float(np.median([x["tokens"] for x in done])),
        "short_n": len(short), "short_correct": int(sum(x["correct"] for x in short)),
        "long_n": len(long_), "long_correct": int(sum(x["correct"] for x in long_)),
        "long_correct_problems": len({(x["benchmark"], x["doc_hash"]) for x in long_ if x["correct"]}),
        "reach": LAYERS * window,
        "shorter_than_reach": 100 * float(np.mean([x["tokens"] < LAYERS * window for x in c])),
    }
for a1, a2 in [MAIN, ("kv448", "swa572")]:
    per = {a: defaultdict(int) for a in (a1, a2)}
    for x in comps:
        if x["arm"] in per and x["correct"] and x["tokens"] >= LONG:
            per[x["arm"]][(x["benchmark"], x["doc_hash"])] += 1
    d = np.array([per[a1][k] - per[a2][k] for k in set(per[a1]) | set(per[a2])])
    out["math"][f"long_correct_problems_{a1}_vs_{a2}"] = {"more": int((d > 0).sum()), "fewer": int((d < 0).sum())}
out["math"]["median_tokens_main"] = float(np.median([x["tokens"] for x in comps if x["arm"] in MAIN]))

# Split the math problems by how long their answers typically are: the median over all 32
# answers of the two main students. The split is the same for both, so a model that merely
# writes longer cannot move problems between groups.
lens = defaultdict(list)
for x in comps:
    if x["arm"] in MAIN:
        lens[(x["benchmark"], x["doc_hash"])].append(x["tokens"])
typical = {k: float(np.median(v)) for k, v in lens.items()}
out["problem_split"] = {}
for a1, a2 in [MAIN, ("kv448", "swa572")]:
    for group, sel in (("short", lambda m: m < LONG), ("long", lambda m: m >= LONG)):
        ks = sorted(k for k, m in typical.items() if sel(m))
        d = np.array([prob[(a1, b)][h]["score"] - prob[(a2, b)][h]["score"] for b, h in ks])
        boots = d[rng.integers(0, len(d), size=(20000, len(d)))].mean(axis=1)
        perm = (d * rng.choice((-1.0, 1.0), size=(5000, len(d)))).mean(axis=1)
        out["problem_split"][f"{a1}-{a2}/{group}"] = {
            "n": len(ks),
            a1: 100 * float(np.mean([prob[(a1, b)][h]["score"] for b, h in ks])),
            a2: 100 * float(np.mean([prob[(a2, b)][h]["score"] for b, h in ks])),
            "diff": 100 * float(d.mean()), "ci": [100 * float(q) for q in np.quantile(boots, [0.025, 0.975])],
            "p": float(np.mean(np.abs(perm) >= abs(d.mean()) - 1e-12)),
        }

json.dump(out, open(DATA / "post_numbers.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("scores", "median_tokens", "n_problems", "math")}, indent=1))
for k, v in out["paired"].items():
    if k.split("/")[0] in ("kv448-swa576", "swa576-swa572", "kv448-query64"):
        print(f"{k:28s} {v['diff']:+6.2f} ({v['ci'][0]:+6.2f}, {v['ci'][1]:+6.2f}) p={v['p']:.3f} better/worse {v['better']}/{v['worse']} of {v['n']}")
