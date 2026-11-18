"""LM-based readouts for one condition's final summaries (RUN_DIR/{cond}_final.jsonl).
--task qa   : Llama-3.1-8B-Instruct answers 10 code-keyed yes/no questions from w alone (greedy).
--task para : Qwen2.5-7B-Instruct paraphrases each summary (greedy) -> summaries_{split}_para.jsonl
--task ppl  : perplexity of each summary under the base text model (LR-X2: frozen Qwen2.5-VL-7B-Instruct, text only,
              summary tokens scored after a bare BOS-free start; mean NLL over summary tokens).
Usage: eval_lm.py --task qa --run DIR --cond A_search --split val [--gpu_mem 0.4]
"""
import argparse, json, os, sys, math, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True); ap.add_argument("--run", required=True); ap.add_argument("--cond", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--gpu_mem", type=float, default=0.40); ap.add_argument("--infile", default=None)
a = ap.parse_args()
from vllm import LLM, SamplingParams

src = a.infile or f"{a.run}/{a.cond}_final.jsonl"
rows = {r["patch_id"]: r for r in map(json.loads, open(src))}
pids = [p for p in ids(a.split) if p in rows]
OUTP = f"{a.run}/{a.cond}"
texts = [rows[p]["summary"] for p in pids]

LLAMA = "unsloth/Llama-3.1-8B-Instruct"
QWEN_T = "Qwen/Qwen2.5-7B-Instruct"


def questions():
    d = pd.read_parquet(f"{ARR}/descriptors.parquet")
    tr = d[d.split == "rlpool"]
    ndmi_med = float(tr.ndmi.median())
    from collections import Counter
    top6 = [l for l, _ in Counter(l for ls in tr["labels"] for l in ls).most_common(6)]
    qs = [("ndvi_gt_0.5", "Is the average NDVI (normalized difference vegetation index) of this patch above 0.5?", lambda r: r.ndvi > 0.5),
          ("ndvi_lt_0.2", "Is the average NDVI (normalized difference vegetation index) of this patch below 0.2?", lambda r: r.ndvi < 0.2),
          ("ndmi_gt_med", f"Is the average NDMI (normalized difference moisture index) of this patch above {ndmi_med:.3f}?", lambda r: r.ndmi > ndmi_med),
          ("ndwi_gt_0", "Is the average NDWI (normalized difference water index) of this patch above 0?", lambda r: r.ndwi > 0)]
    for l in top6:
        qs.append((f"label:{l}", f"Does this patch contain the land cover class '{l}'?", (lambda L: lambda r: L in set(r["labels"]))(l)))
    return qs, d.set_index("patch_id"), {"ndmi_rlpool_median": ndmi_med, "top6_rlpool": top6}


if a.task == "qa":
    qs, d, meta = questions()
    llm = LLM(LLAMA, revision=rev(LLAMA), dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=a.gpu_mem, seed=0)
    tok = llm.get_tokenizer()
    reqs, keys = [], []
    for p, w in zip(pids, texts):
        for qid, q, f in qs:
            msg = [{"role": "user", "content": f"Here is a description of a Sentinel-2 satellite image patch.\n\nDescription: {w}\n\n"
                    f"Based only on the description, answer the question with a single word, yes or no.\nQuestion: {q}"}]
            reqs.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)); keys.append((p, qid))
    outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=3))
    ans = [o.outputs[0].text.strip().lower() for o in outs]
    dtr = d[d.split == "rlpool"]
    res, raw = {}, []
    for qi, (qid, q, f) in enumerate(qs):
        kt = np.array([bool(f(r)) for _, r in dtr.iterrows()]); maj = bool(kt.mean() >= 0.5)
        k_ev = np.array([bool(f(d.loc[p])) for p in pids])
        a_ev = [ans[i] for i in range(qi, len(ans), len(qs))]
        pred = np.array([x.startswith("yes") for x in a_ev]); valid = np.array([x.startswith(("yes", "no")) for x in a_ev])
        res[qid] = {"question": q, "acc": float(((pred == k_ev) & valid).mean()), "majority_acc": float((k_ev == maj).mean()),
                    "majority_answer": "yes" if maj else "no", "n": int(len(k_ev)), "n_unparsed": int((~valid).sum()),
                    "key_yes_rate": float(k_ev.mean()), "pred_yes_rate": float(pred.mean())}
    for (p, qid), x in zip(keys, ans):
        raw.append({"patch_id": p, "qid": qid, "answer": x})
    with open(f"{OUTP}_qa_raw.jsonl", "w") as fo:
        for r in raw: fo.write(json.dumps(r) + "\n")
    out = {"meta": meta, "per_question": res, "mean_acc": float(np.mean([v["acc"] for v in res.values()])),
           "mean_majority_acc": float(np.mean([v["majority_acc"] for v in res.values()]))}
    json.dump(out, open(f"{OUTP}_qa.json", "w"), indent=1); print(json.dumps(out, indent=1))

elif a.task == "para":
    llm = LLM(QWEN_T, revision=rev(QWEN_T), dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=a.gpu_mem, seed=0)
    tok = llm.get_tokenizer()
    reqs = [tok.apply_chat_template([{"role": "user", "content": "Paraphrase the following text. Keep every fact and roughly the same length. "
                                      f"Output only the paraphrase.\n\n{w}"}], tokenize=False, add_generation_prompt=True) for w in texts]
    outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=200))
    with open(f"{a.run}/{a.cond}_para_final.jsonl", "w") as fo:
        for p, o in zip(pids, outs):
            w = o.outputs[0].text.strip()
            fo.write(json.dumps({**rows[p], "summary": w, "n_words": n_words(w), "condition": a.cond + "_para", "paraphrase_of": rows[p]["summary"]}) + "\n")
    print("para done", len(pids))

elif a.task == "ppl":
    llm = LLM(VLM, revision=rev(VLM), dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=a.gpu_mem, seed=0,
              limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    # score summary tokens; first token conditioned on <|im_start|> only (no image, no instruction)
    reqs = [{"prompt_token_ids": tok("<|im_start|>").input_ids + tok(w).input_ids} for w in texts]
    outs = llm.generate(reqs, SamplingParams(max_tokens=1, prompt_logprobs=0))
    ppl = []
    for o in outs:
        lps = [x[t].logprob for x, t in zip(o.prompt_logprobs[1:], o.prompt_token_ids[1:])]
        ppl.append(math.exp(-np.mean(lps)) if lps else float("nan"))
    with open(f"{OUTP}_ppl.jsonl", "w") as fo:
        for p, v in zip(pids, ppl): fo.write(json.dumps({"patch_id": p, "ppl": v}) + "\n")
    pp = np.array(ppl)
    out = {"mean": float(np.nanmean(pp)), "median": float(np.nanmedian(pp)), "n": int(len(pp))}
    json.dump(out, open(f"{OUTP}_ppl.json", "w"), indent=1); print(out)
