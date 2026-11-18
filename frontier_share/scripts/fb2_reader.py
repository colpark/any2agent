"""FB-2: Phi-4 reader (pinned, greedy) for C0 (Q17 only, caption only), C1 (caption + 5 Prithvi neighbors), C2 (caption + 5 random bank patches).
Pilot reader template; the entry count follows the number of questions asked (16 in the pilot, 17 here, 1 for C0-Q17).
The neighbor block (verbatim from DESIGN_FB_v1) sits between the description paragraph and the questions.
Output: frontier_share/fb/reader_answers.jsonl, fb/reader_meta.json."""
import hashlib, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from reader_parse import parse

H = os.path.expanduser("~"); FS = f"{H}/Documents/langres/frontier_share"; REPO = f"{H}/Documents/langres"; FB = f"{H}/langres_state/fb"
READER = "microsoft/phi-4"; REV = json.load(open(f"{REPO}/model_revisions.json"))[READER]["revision"]
QS = json.load(open(f"{FS}/fb/questions17.json"))
HEAD = ("Here is a description of a Sentinel-2 satellite image patch: {text}. Answer every question below from the description only, "
        "with yes or no. When the description does not settle a question, give your best guess. Return JSON with a list `answers` "
        "of {n} entries with `id` and `answer`.")
BLOCK = ("A foundation model that sees six Sentinel-2 bands, including shortwave infrared, which the descriptions never saw, found these five "
         "patches {how}. Their descriptions come from another model: {items}. Use them as evidence about this patch, especially for "
         "properties its own description does not settle.")
HOW = {"C1": "most similar to this one", "C2": "chosen at random from the dataset"}
sel = json.load(open(f"{H}/langres_state/fs/selection.json"))["uid_order"]
NB = json.load(open(f"{FB}/neighbors.json"))
BANK = [json.loads(l) for l in open(f"{FB}/bank_captions.jsonl")]


def qblock(qs):
    return "\n".join(f"{q['id']}. {q['text']}" for q in qs)


jobs = []
for u in sel:
    w100 = json.load(open(f"{FS}/raw/caption_main_{u}.json"))["summaries"]["w100"]
    q17 = [QS[-1]]
    jobs.append({"uid": u, "cond": "C0_Q17", "qids": ["Q17"], "prompt": HEAD.replace("{n}", "1").format(text=w100) + "\n\n" + qblock(q17)})
    for c, key in [("C1", "neighbors"), ("C2", "random")]:
        items = " ".join(f"[{r + 1}] cosine {cs:.2f}: {BANK[k]['caption']}" for r, (k, cs) in enumerate(NB[u][key]))
        p = HEAD.replace("{n}", "17").format(text=w100) + "\n\n" + BLOCK.format(how=HOW[c], items=items) + "\n\n" + qblock(QS)
        jobs.append({"uid": u, "cond": c, "qids": [q["id"] for q in QS], "prompt": p, "bank_ids": [BANK[k]["bank_id"] for k, _ in NB[u][key]]})

subprocess.run([sys.executable, f"{REPO}/scripts/evict_cache.py", "~/.cache/huggingface/hub", "~/langres_data", "~/langres_state"])
tok = AutoTokenizer.from_pretrained(READER, revision=REV); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(READER, revision=REV, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
subprocess.run([sys.executable, f"{REPO}/scripts/evict_cache.py", "~/.cache/huggingface/hub"])
texts = [tok.apply_chat_template([{"role": "user", "content": j["prompt"]}], add_generation_prompt=True, tokenize=False) for j in jobs]
order = sorted(range(len(jobs)), key=lambda i: len(texts[i])); BS = 8; t0 = time.time()
for k in range(0, len(order), BS):
    b = order[k:k + BS]
    enc = tok([texts[i] for i in b], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        o = m.generate(**enc, max_new_tokens=2000, do_sample=False)
    for pos, (i, t) in enumerate(zip(b, tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))):
        jobs[i]["raw_output"] = t.strip(); jobs[i]["prompt_tokens"] = int(enc["attention_mask"][pos].sum())
        jobs[i]["hit_limit"] = bool((o[pos, enc["input_ids"].shape[1]:] != tok.pad_token_id).sum() >= 2000)
    print("read", min(k + BS, len(order)), "/", len(order), round(time.time() - t0), "s", flush=True)
with open(f"{FS}/fb/reader_answers.jsonl", "w") as fo:
    for j in jobs:
        ans, ok = parse(j["raw_output"], j["qids"])
        fo.write(json.dumps({**{k: v for k, v in j.items() if k != "prompt"}, "prompt_sha256": hashlib.sha256(j["prompt"].encode()).hexdigest(),
                             "answers": ans, "valid": ok}) + "\n")
json.dump({"model": READER, "revision": REV, "decoding": "greedy, max_new_tokens 2000, bf16, sdpa, batch 8 left-padded",
           "n_jobs": len(jobs), "n_valid": sum(parse(j["raw_output"], j["qids"])[1] for j in jobs), "n_hit_limit": sum(j["hit_limit"] for j in jobs),
           "seconds": round(time.time() - t0), "example_prompt_C1": jobs[1]["prompt"]}, open(f"{FS}/fb/reader_meta.json", "w"), indent=1)
print("DONE")
