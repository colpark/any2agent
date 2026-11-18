"""FS-E2: rerun reader items whose greedy output hit max_new_tokens=700 before any JSON, with max_new_tokens=2000.
Greedy decoding is deterministic, so the first 700 tokens are unchanged; the output simply continues. Same prompt, model, revision."""
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from reader_parse import parse
H = os.path.expanduser("~"); FS = f"{H}/Documents/langres/frontier_share"; REPO = f"{H}/Documents/langres"
READER = "microsoft/phi-4"; REV = json.load(open(f"{REPO}/model_revisions.json"))[READER]["revision"]
QS = json.load(open(f"{FS}/questions.json")); QIDS = [q["id"] for q in QS]
HEAD = ("Here is a description of a Sentinel-2 satellite image patch: {text}. Answer every question below from the description only, "
        "with yes or no. When the description does not settle a question, give your best guess. Return JSON with a list `answers` "
        "of 16 entries with `id` and `answer`.")
QBLOCK = "\n".join(f"{q['id']}. {q['text']}" for q in QS)
L = [json.loads(l) for l in open(f"{FS}/reader/answers.jsonl")]
idx = [i for i, r in enumerate(L) if all(v is None for v in r["answers"].values())]
print("rerun", [(L[i]["role"], L[i]["uid"][:8], L[i]["cond"]) for i in idx], flush=True)
subprocess.run([sys.executable, f"{REPO}/scripts/evict_cache.py", "~/.cache/huggingface/hub", "~/langres_data", "~/langres_state"])
tok = AutoTokenizer.from_pretrained(READER, revision=REV)
m = AutoModelForCausalLM.from_pretrained(READER, revision=REV, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
for i in idx:
    r = L[i]; cap = json.load(open(f"{FS}/raw/caption_{r['role']}_{r['uid']}.json"))
    text = json.dumps(cap, indent=1, ensure_ascii=False) if r["cond"] == "full_json" else cap["summaries"][r["cond"]]
    prompt = HEAD.format(text=text) + "\n\n" + QBLOCK
    import hashlib; assert hashlib.sha256(prompt.encode()).hexdigest() == r["prompt_sha256"]
    enc = tok([tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)], return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = m.generate(**enc, max_new_tokens=2000, do_sample=False)
    out = tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    r["raw_output_700"] = r["raw_output"]; r["raw_output"] = out; r["rerun"] = "FS-E2 max_new_tokens 2000"
    r["answers"], r["valid"] = parse(out, QIDS)
    print(r["cond"], r["valid"], len(out), out.startswith(r["raw_output_700"][:200]), flush=True)
open(f"{FS}/reader/answers.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in L))
meta = json.load(open(f"{FS}/reader/run_meta.json")); meta["n_valid"] = sum(r["valid"] for r in L); meta["fs_e2_rerun"] = len(idx)
json.dump(meta, open(f"{FS}/reader/run_meta.json", "w"), indent=1); print("valid", meta["n_valid"])
