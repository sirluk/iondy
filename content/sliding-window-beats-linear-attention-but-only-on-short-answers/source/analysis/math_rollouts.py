"""Per-completion view of the math rollouts.

Usage: python -I math_rollouts.py <bundle_dir> <tokenizer.json> <out.json>

Correctness per completion: extracted answer == target after light normalization.
Checked against the logged per-problem pass@1.
"""

import csv
import json
import re
import sys

from tokenizers import Tokenizer

bundle, tok_path, out_path = sys.argv[1:4]
tok = Tokenizer.from_file(tok_path)
MATH = ("aime2024", "aime2025", "hmmt_202502", "hmmt_202511")


def norm(x):
    x = str(x).strip().strip("$").replace(" ", "").replace("\\!", "").replace("dfrac", "frac").replace("tfrac", "frac")
    x = re.sub(r"\\text\{[^}]*\}", "", x).rstrip(".")
    try:
        return str(int(float(x))) if re.fullmatch(r"-?\d+(\.0+)?", x) else x
    except ValueError:
        return x


def repetition(text, n=200):
    """Share of the last 4,000 characters covered by 200-character chunks seen before in that tail."""
    t = text[-4000:]
    chunks = [t[i:i + n] for i in range(0, len(t) - n, n)]
    return 1 - len(set(chunks)) / max(1, len(chunks))


out, mismatch = [], 0
with open(f"{bundle}/artifacts.tsv") as f:
    for a in csv.DictReader(f, delimiter="\t"):
        bench = next((b for b in MATH if a["benchmark"].startswith(b)), None)
        if not bench:
            continue
        with open(f"{bundle}/{a['samples_jsonl']}") as g:
            for line in g:
                r = json.loads(line)
                resps = r["resps"][0][:16]
                ext = r["extracted_answers"][:16]
                lens = [len(e.ids) for e in tok.encode_batch(resps, add_special_tokens=False)]
                correct = [norm(e) == norm(r["target"]) for e in ext]
                if abs(sum(correct) / 16 - float(r["pass@1"])) > 1e-6:
                    mismatch += 1
                for i in range(16):
                    out.append({"arm": a["arm"], "bench": bench, "doc_hash": r["doc_hash"], "len": lens[i],
                                "correct": correct[i], "answered": ext[i] not in (None, "", "None", "[invalid]"),
                                "repetition": repetition(resps[i]), "logged_pass1": float(r["pass@1"])})
print("problems where my grading differs from the logged pass@1:", mismatch)
json.dump(out, open(out_path, "w"))
print(len(out), "completions")
