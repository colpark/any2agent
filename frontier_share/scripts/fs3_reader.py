"""FS-3: Phi-4 text-only reader, greedy, on six conditions per caption (w25..w400 and the full JSON rendered as text),
for both the main and the retest caption of every patch. Saves every raw reader output.
Output: frontier_share/reader/answers.jsonl (one line per caption x condition) and reader/run_meta.json."""
import glob, hashlib, json, os, re, subprocess, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

H = os.path.expanduser("~"); FS = f"{H}/Documents/langres/frontier_share"; REPO = f"{H}/Documents/langres"
READER = "microsoft/phi-4"; REV = json.load(open(f"{REPO}/model_revisions.json"))[READER]["revision"]
QUESTIONS = json.load(open(f"{FS}/questions.json"))
HEAD = ("Here is a description of a Sentinel-2 satellite image patch: {text}. Answer every question below from the description only, "
        "with yes or no. When the description does not settle a question, give your best guess. Return JSON with a list `answers` "
        "of 16 entries with `id` and `answer`.")
QBLOCK = "\n".join(f"{q['id']}. {q['text']}" for q in QUESTIONS)
CONDS = ["w25", "w50", "w100", "w200", "w400", "full_json"]


def parse(out):
    m = re.search(r"\{.*\}", out, re.S)
    try:
        obj = json.loads(m.group(0)) if m else None
        ans = {str(a["id"]).strip(): str(a["answer"]).strip().lower() for a in obj["answers"]}
    except Exception:
        return {}, False
    ans = {q["id"]: (ans.get(q["id"]) if ans.get(q["id"]) in ("yes", "no") else None) for q in QUESTIONS}
    return ans, all(v is not None for v in ans.values())


subprocess.run([sys.executable, f"{REPO}/scripts/evict_cache.py", "~/.cache/huggingface/hub", "~/langres_data", "~/langres_state"])
jobs = []
for f in sorted(glob.glob(f"{FS}/raw/caption_*_*.json")):
    role, uid = os.path.basename(f)[len("caption_"):-5].split("_", 1)
    cap = json.load(open(f))
    for c in CONDS:
        text = json.dumps(cap, indent=1, ensure_ascii=False) if c == "full_json" else cap["summaries"][c]
        jobs.append({"role": role, "uid": uid, "cond": c, "prompt": HEAD.format(text=text) + "\n\n" + QBLOCK})
tok = AutoTokenizer.from_pretrained(READER, revision=REV); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(READER, revision=REV, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
subprocess.run([sys.executable, f"{REPO}/scripts/evict_cache.py", "~/.cache/huggingface/hub"])
texts = [tok.apply_chat_template([{"role": "user", "content": j["prompt"]}], add_generation_prompt=True, tokenize=False) for j in jobs]
order = sorted(range(len(jobs)), key=lambda i: len(texts[i])); BS = 8; t0 = time.time()
for k in range(0, len(order), BS):
    b = order[k:k + BS]
    enc = tok([texts[i] for i in b], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        o = m.generate(**enc, max_new_tokens=700, do_sample=False)
    for i, t in zip(b, tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
        jobs[i]["raw_output"] = t.strip(); jobs[i]["prompt_tokens"] = int(enc["attention_mask"][b.index(i)].sum())
    print("read", min(k + BS, len(order)), "/", len(order), round(time.time() - t0), "s", flush=True)
os.makedirs(f"{FS}/reader", exist_ok=True)
with open(f"{FS}/reader/answers.jsonl", "w") as fo:
    for j in jobs:
        ans, ok = parse(j["raw_output"])
        fo.write(json.dumps({"role": j["role"], "uid": j["uid"], "cond": j["cond"], "prompt_sha256": hashlib.sha256(j["prompt"].encode()).hexdigest(),
                             "prompt_tokens": j["prompt_tokens"], "answers": ans, "valid": ok, "raw_output": j["raw_output"]}) + "\n")
json.dump({"model": READER, "revision": REV, "decoding": "greedy, max_new_tokens 700, bf16, sdpa, batch 8 left-padded",
           "head_template_sha256": hashlib.sha256((HEAD + QBLOCK).encode()).hexdigest(), "n_jobs": len(jobs),
           "n_valid": sum(parse(j["raw_output"])[1] for j in jobs), "seconds": round(time.time() - t0)},
          open(f"{FS}/reader/run_meta.json", "w"), indent=1)
print("DONE", len(jobs))
