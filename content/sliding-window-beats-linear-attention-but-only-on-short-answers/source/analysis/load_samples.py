"""Load the matched-memory eval samples into one per-problem table.

Usage: python -I load_samples.py <bundle_dir> <tokenizer.json> <out.json>

Keeps resps[0][:16] for math and resps[0][:1] for short tasks (distributed padding),
as the bundle README says. Lengths are re-tokenized from the saved text.
"""

import csv
import json
import sys

from tokenizers import Tokenizer

bundle, tok_path, out_path = sys.argv[1:4]
tok = Tokenizer.from_file(tok_path)

MATH = ("aime2024", "aime2025", "hmmt_202502", "hmmt_202511")
METRIC = {"gsm8k": "exact_match", "gpqa": "exact_match", "ifeval": "prompt_level_strict_acc", "humaneval": "pass@1"}

rows = []
with open(f"{bundle}/artifacts.tsv") as f:
    for a in csv.DictReader(f, delimiter="\t"):
        bench = next(b for b in (*MATH, *METRIC) if a["benchmark"].startswith(b))
        is_math = bench in MATH
        with open(f"{bundle}/{a['samples_jsonl']}") as g:
            for line in g:
                r = json.loads(line)
                resps = r["resps"][0][: 16 if is_math else 1]
                lengths = [len(e.ids) for e in tok.encode_batch(resps, add_special_tokens=False)]
                score = float(r["pass@1"] if is_math else r[METRIC[bench]])
                rows.append({
                    "arm": a["arm"], "bench": bench, "doc_hash": r["doc_hash"], "doc_id": r["doc_id"],
                    "score": score, "lengths": lengths, "n_resps_saved": len(r["resps"][0]),
                    "no_answer_rate": float(r["no_answer_rate"]) if "no_answer_rate" in r else None,
                    "extracted": r.get("extracted_answers"), "target": r.get("target"),
                    "tails": [x[-400:] for x in resps] if bench == "gsm8k" else None,
                    "prompt_len": len(tok.encode(r["arguments"]["gen_args_0"]["arg_0"], add_special_tokens=False).ids)
                    if not is_math or r["doc_id"] in (0, "0") else None,
                })
        print(a["arm"], bench, "done", flush=True)

json.dump(rows, open(out_path, "w"))
print(len(rows), "rows")
