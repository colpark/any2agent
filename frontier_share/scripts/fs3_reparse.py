"""FS-E1: re-parse the saved raw reader outputs with the fixed parser (no model call)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from reader_parse import parse
FS = os.path.expanduser("~/Documents/langres/frontier_share")
QIDS = [q["id"] for q in json.load(open(f"{FS}/questions.json"))]
L = [json.loads(l) for l in open(f"{FS}/reader/answers.jsonl")]
for r in L:
    r["answers"], r["valid"] = parse(r["raw_output"], QIDS)
open(f"{FS}/reader/answers.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in L))
m = json.load(open(f"{FS}/reader/run_meta.json")); m["n_valid"] = sum(r["valid"] for r in L); m["parser"] = "reader_parse.py (FS-E1)"
json.dump(m, open(f"{FS}/reader/run_meta.json", "w"), indent=1)
print("valid", m["n_valid"], "/", len(L))
