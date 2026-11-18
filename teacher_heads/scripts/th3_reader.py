"""TH-3 (E3): Phi-4 reader on Test300 via vLLM, greedy, guided JSON (list `answers` of 8 {id, answer}); conditions R0, R1, R1s.
Readout from the Astra + code heads (classes with a trained 'both' head). Grading by true labels only. Usage: th3_reader.py run | grade"""
import hashlib, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import binomtest
from th_common import *

HEAD = ("Here is a description of a Sentinel-2 satellite image patch: {text}. Answer every question below from the description only, "
        "with yes or no. When the description does not settle a question, give your best guess. Return JSON with a list `answers` "
        "of 8 entries with `id` and `answer`.")
QS = [("G1", "Is any forest present?"), ("G2", "Is any farmland present?"), ("G3", "Is any built-up land present?"),
      ("G4", "Is any open water present?"), ("G5", "Is any wetland present?"),
      ("G6", "Is any natural open land present, such as grassland, heath, shrub or sand?"),
      ("G7", "Does the patch mix farmland with forest, wetland or natural open land?"), ("G8", "Does the patch contain more than three land cover types?")]
QIDS = [q for q, _ in QS]
T300 = sets()["test300"]


def keys(Y):  # grading only
    g = lambda *c: Y[:, [x - 1 for x in c]].any(1)
    G2 = g(3, 4, 5, 6, 7, 8); G1, G5, G6 = g(9, 10, 11), g(16, 17), g(12, 13, 14, 15)
    return np.stack([G1, G2, g(1, 2), g(18, 19), G5, G6, G2 & (G1 | G5 | G6), Y.sum(1) > 3], 1)


def readout(probs, rel, i):
    items = sorted(((CLASSES[c - 1], float(probs[c][i]), rel[str(c)]["tier"]) for c in probs), key=lambda x: -x[1])
    word = lambda p: "likely" if p > 0.7 else "possible" if p >= 0.3 else "unlikely"
    return "Prithvi readout (heads trained from teacher labels, no human labels): " + " | ".join(
        f"{n} {p:.2f} ({word(p)}, reliability {t})" for n, p, t in items)


def run():
    E1 = json.load(open(f"{TH}/e1.json")); hp = np.load(f"{ST}/head_probs.npy", allow_pickle=True).item()
    probs = {int(k.split(":")[1]): v for k, v in hp.items() if k.startswith("astra+code:")}
    caps = {json.loads(l)["test_index"]: json.loads(l)["caption"] for l in open(f"{ST}/qwen_captions_test300.jsonl")}
    rng = np.random.default_rng(0)
    while True:
        perm = rng.permutation(len(T300))
        if not (perm == np.arange(len(T300))).any(): break
    qb = "\n".join(f"{q}. {t}" for q, t in QS); jobs = []
    for j, i in enumerate(T300):
        base = HEAD.format(text=caps[i])
        jobs.append(("R0", i, base + "\n\n" + qb))
        jobs.append(("R1", i, base + "\n\n" + readout(probs, E1["reliability"], i) + "\n\n" + qb))
        jobs.append(("R1s", i, base + "\n\n" + readout(probs, E1["reliability"], T300[perm[j]]) + "\n\n" + qb))
    subprocess.run([PY, f"{REPO}/scripts/evict_cache.py", "~/.cache/huggingface/hub", "~/langres_data", "~/langres_state"])
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    R = json.load(open(f"{REPO}/model_revisions.json"))["microsoft/phi-4"]["revision"]
    llm = LLM("microsoft/phi-4", revision=R, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.5, seed=0)
    tok = llm.get_tokenizer()
    schema = {"type": "object", "additionalProperties": False, "required": ["answers"], "properties": {"answers": {
        "type": "array", "minItems": 8, "maxItems": 8, "items": {"type": "object", "additionalProperties": False, "required": ["id", "answer"],
        "properties": {"id": {"type": "string", "enum": QIDS}, "answer": {"type": "string", "enum": ["yes", "no"]}}}}}}
    sp = SamplingParams(temperature=0, max_tokens=300, seed=0, structured_outputs=StructuredOutputsParams(json=schema, disable_any_whitespace=True))
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for _, _, p in jobs]
    t0 = time.time(); outs = llm.generate(prompts, sp); sec = round(time.time() - t0)
    with open(f"{TH}/reader_answers.jsonl", "w") as f:
        for (c, i, p), o in zip(jobs, outs):
            try:
                a = {e["id"]: e["answer"] for e in json.loads(o.outputs[0].text)["answers"]}
            except Exception:
                a = {}
            f.write(json.dumps({"cond": c, "test_index": i, "answers": a, "raw": o.outputs[0].text, "prompt_sha256": hashlib.sha256(p.encode()).hexdigest()}) + "\n")
    json.dump({"model": "microsoft/phi-4", "revision": R, "n": len(jobs), "sec": sec, "n_readout_classes": len(probs), "derangement_seed": 0,
               "example_R1": jobs[1][2]}, open(f"{TH}/reader_meta.json", "w"), indent=1)
    print("DONE", len(jobs), sec)


def grade():
    s = sets(); K = keys(true_labels([s["test_ids"][i] for i in T300])); pos = {i: j for j, i in enumerate(T300)}
    A = {}
    for l in open(f"{TH}/reader_answers.jsonl"):
        r = json.loads(l); A.setdefault(r["cond"], {})[r["test_index"]] = r["answers"]
    right = {c: np.array([[A[c][i].get(q) == ("yes" if K[pos[i], k] else "no") for k, q in enumerate(QIDS)] for i in T300]) for c in A}
    out = {"per_question": {c: dict(zip(QIDS, right[c].mean(0).round(3).tolist())) for c in right}, "pooled": {c: float(right[c].mean()) for c in right},
           "missing_answers": {c: int(sum(len(A[c][i]) < 8 for i in T300)) for c in A},
           "yes_rate": {c: float(np.mean([v == "yes" for i in T300 for v in A[c][i].values()])) for c in A},
           "key_yes_rate": dict(zip(QIDS, K.mean(0).round(3).tolist()))}
    for a_, b_ in (("R1", "R0"), ("R1", "R1s")):
        x, y = right[a_].ravel(), right[b_].ravel(); w, l = int((x & ~y).sum()), int((~x & y).sum())
        out[f"{a_}_vs_{b_}"] = {"diff": float(x.mean() - y.mean()), f"{a_}_only": w, f"{b_}_only": l, "mcnemar_p": float(binomtest(w, w + l, 0.5).pvalue) if w + l else 1.0}
    out["gate_E3"] = out["R1_vs_R0"]["diff"] >= 0.03 and out["R1_vs_R0"]["mcnemar_p"] < 0.05 and out["R1_vs_R1s"]["diff"] > 0
    json.dump(out, open(f"{TH}/e3.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "per_question"}, indent=1))


if __name__ == "__main__":
    {"run": run, "grade": grade}[sys.argv[1]]()
