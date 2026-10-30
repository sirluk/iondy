"""Write one eval bundle's rows into the data tables, replacing earlier rows of the same arms.

Usage: python -I write_tables.py <samples.json> <math_rollouts.json> <data_dir>

The inputs are the outputs of load_samples.py and math_rollouts.py for one bundle.
"""

import csv
import json
import os
import sys

import numpy as np

samples_path, math_path, data = sys.argv[1:4]
rows = json.load(open(samples_path))
comps = json.load(open(math_path))
arms = {r["arm"] for r in rows}

tables = {
    "memory_matched_per_problem.csv": (
        ["arm", "benchmark", "doc_hash", "doc_id", "score", "n_completions", "mean_tokens", "median_tokens", "prompt_tokens"],
        [[r["arm"], r["bench"], r["doc_hash"], r["doc_id"], r["score"], len(r["lengths"]), round(float(np.mean(r["lengths"])), 1),
          float(np.median(r["lengths"])), r["prompt_len"] if r["prompt_len"] is not None else ""] for r in rows],
    ),
    "memory_matched_math_completions.csv": (
        ["arm", "benchmark", "doc_hash", "tokens", "correct", "answered", "tail_repetition"],
        [[x["arm"], x["bench"], x["doc_hash"], x["len"], int(x["correct"]), int(x["answered"]), round(x["repetition"], 3)] for x in comps],
    ),
}
for name, (header, new) in tables.items():
    path = os.path.join(data, name)
    old = []
    if os.path.exists(path):
        with open(path) as f:
            r = csv.reader(f)
            assert next(r) == header, f"unexpected header in {path}"
            old = [row for row in r if row[0] not in arms]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(old + new)
    print(f"{name}: kept {len(old)} rows, wrote {len(new)} rows for {sorted(arms)}")
