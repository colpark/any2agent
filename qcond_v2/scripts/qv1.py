"""QV-1. Usage: qv1.py prompts | rewrite (node A2) | check | embed (host A GPU)
prompts: one rewrite prompt per question in qcond/questions.json (141 practice incl. the 2 skipped by v1, 33 held out) -> ~/langres_state/qv/rw_prompts.json
rewrite: Qwen3.5-9B (stage 1 revision), vLLM bf16, text only, greedy, thinking off, guided JSON -> rw_out.jsonl (one retry at 2x tokens on schema failure)
check:   signs on the 31 physical questions vs known directions (stop below 0.90); semantic sign -1 list -> qcond_v2/rewrites.json, rewrite_check.json
embed:   Qwen3-Embedding-4B as v1 on positive forms and negations -> emb_rewrites.npz"""
import hashlib, json, os, subprocess, sys, time
H = os.path.expanduser("~"); SQV = f"{H}/langres_state/qv"
PROMPT = ("Rewrite the yes or no question below about a satellite image patch. First give its positive form: the same property, asked in the direction of "
          "presence, a high value, or the direction the property names. Give sign 1 if the original already asks in that direction, and sign -1 if it asks "
          "about absence, a low value, the bottom of a range, or the reverse direction. Keep the property and its definition, and change only the direction. "
          "Then give the natural negation of the original question, with a definition. Question: {question} Definition: {definition}")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["positive_question", "positive_definition", "sign", "negation_question", "negation_definition"],
          "properties": {"positive_question": {"type": "string"}, "positive_definition": {"type": "string"}, "sign": {"type": "integer", "enum": [1, -1]},
                         "negation_question": {"type": "string"}, "negation_definition": {"type": "string"}}}
QWEN, QREV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def prompts():
    sys.path.insert(0, os.path.dirname(__file__)); from qv_common import questions
    P = [{"id": q["id"], "split": q["split"], "kind": q["kind"], "prompt": PROMPT.replace("{question}", q["question"]).replace("{definition}", q["definition"])} for q in questions()]
    json.dump(P, open(f"{SQV}/rw_prompts.json", "w"), indent=1); print(len(P), "prompts")


def rewrite():
    import jsonschema
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    subprocess.run([f"{H}/langres_state/venv/bin/python", f"{H}/Documents/langres/scripts/evict_cache.py", "~/.cache/huggingface/hub", "~/langres_data", "~/langres_state"])
    P = json.load(open(f"{SQV}/rw_prompts.json")); tok = AutoTokenizer.from_pretrained(QWEN, revision=QREV); V = jsonschema.Draft202012Validator(SCHEMA)
    chat = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in P]
    llm = LLM(QWEN, revision=QREV, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.6, seed=0, limit_mm_per_prompt={"image": 0})
    sp = lambda m: SamplingParams(temperature=0, max_tokens=m, seed=0, structured_outputs=StructuredOutputsParams(json=SCHEMA))
    def ok(t):
        try: return not list(V.iter_errors(json.loads(t)))
        except Exception: return False
    t0 = time.time(); o1 = [o.outputs[0].text for o in llm.generate(chat, sp(800))]; bad = [k for k, t in enumerate(o1) if not ok(t)]
    o2 = dict(zip(bad, [o.outputs[0].text for o in llm.generate([chat[k] for k in bad], sp(1600))])) if bad else {}
    with open(f"{SQV}/rw_out.jsonl", "w") as f:
        for k, p in enumerate(P):
            r = {"id": p["id"], "attempt": 2 if k in o2 else 1, "raw": o2.get(k, o1[k]), "prompt_sha256": hashlib.sha256(chat[k].encode()).hexdigest()}
            r["valid"] = ok(r["raw"]); f.write(json.dumps(r) + "\n")
    print(json.dumps({"n": len(P), "retried": len(bad), "invalid": sum(not ok(t) for t in o2.values()), "sec": round(time.time() - t0)}))


def check():
    sys.path.insert(0, os.path.dirname(__file__)); from qv_common import QV, questions
    Q = {q["id"]: q for q in questions()}; R = {r["id"]: r for r in map(json.loads, open(f"{SQV}/rw_out.jsonl"))}
    out = [{"id": i, "split": Q[i]["split"], "kind": Q[i]["kind"], "question": Q[i]["question"], **json.loads(R[i]["raw"]), "attempt": R[i]["attempt"]} for i in Q if R[i]["valid"]]
    known = {r["id"]: (-1 if r["id"].endswith("_bottom") else 1) for r in out if r["kind"] == "physical"}
    ag = [r["sign"] == known[r["id"]] for r in out if r["id"] in known]
    res = {"n_valid": len(out), "n_total": len(Q), "physical_compared": len(ag), "sign_agreement": sum(ag) / len(ag), "pass": len(ag) == 31 and sum(ag) / len(ag) >= .9,
           "physical_disagree": [r["id"] for r in out if r["id"] in known and r["sign"] != known[r["id"]]],
           "semantic_sign_minus1": [{"id": r["id"], "split": r["split"], "question": r["question"]} for r in out if r["kind"] == "semantic" and r["sign"] == -1]}
    json.dump(out, open(f"{QV}/rewrites.json", "w"), indent=1); json.dump(res, open(f"{QV}/rewrite_check.json", "w"), indent=1); print(json.dumps(res, indent=1))


def embed():
    import numpy as np, torch
    from sentence_transformers import SentenceTransformer
    sys.path.insert(0, os.path.dirname(__file__)); from qv_common import QV, thc, REPO
    import vs_common
    subprocess.run([thc.PY, f"{REPO}/scripts/evict_cache.py", "~/.cache/huggingface/hub", "~/langres_data", "~/langres_state"])
    m = SentenceTransformer(vs_common.EMB, revision=vs_common.EREV, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}); R = json.load(open(f"{QV}/rewrites.json"))
    enc = lambda T: m.encode(T, batch_size=32, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    np.savez(f"{SQV}/emb_rewrites.npz", ids=[r["id"] for r in R], pos=enc([f"{r['positive_question']} {r['positive_definition']}" for r in R]),
             neg=enc([f"{r['negation_question']} {r['negation_definition']}" for r in R])); print(len(R), "x 2 embedded")


if __name__ == "__main__":
    {"prompts": prompts, "rewrite": rewrite, "check": check, "embed": embed}[sys.argv[1]]()
