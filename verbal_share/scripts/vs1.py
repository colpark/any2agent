"""VS-1. Usage: vs1.py captions (node A2) | prithvi | fields | embed (host A).
captions: Qwen3.5-9B (pinned), vLLM bf16, greedy, thinking off, JSON-schema guided, arm B images, max 3,000 tokens (VS-X1); schema failures rerun
          once at 4,500, then dropped. -> ~/langres_state/vs/captions_{fit,test}.jsonl
prithvi:  scripts/prithvi.py encoder and preprocessing unchanged, forward hooks on all 24 blocks, mean over spatial tokens (CLS dropped)
          -> ~/langres_state/vs/layers_{fit,test}.npy (n, 24, 1024) raw; Fit-standardized in VS-2.
fields:   7 text fields per patch -> verbal_share/fields_{fit,test}.jsonl.   embed: Qwen3-Embedding-4B -> ~/langres_state/vs/emb_{fit,test}.npy"""
import hashlib, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from vs_common import *
N = int(os.environ.get("VS_SMOKE", 0)) or None  # smoke: first N patches per set, outputs tagged smoke_
SPLITS = lambda: tuple(((f"smoke_{t}" if N else t), s, i[:N]) for t, s, i in (("fit", "rlpool", sets()["fit_rl"]), ("test", "test", sets()["test"])))
evict = lambda: subprocess.run([sys.executable, f"{REPO}/scripts/evict_cache.py", "~/.cache/huggingface/hub", "~/langres_data", "~/langres_state"])


def captions():
    import jsonschema
    sys.path.insert(0, f"{REPO}/scripts"); from common import arm_images
    evict()
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    schema = json.load(open(f"{VS}/schemas/caption_vs.json")); V = jsonschema.Draft202012Validator(schema); text = open(f"{VS}/caption_prompt.txt").read()
    proc = AutoProcessor.from_pretrained(QWEN, revision=QREV)
    prompt = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": text}]}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=False)
    llm = LLM(QWEN, revision=QREV, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.6, limit_mm_per_prompt={"image": 2}, seed=0)
    sp = lambda m: SamplingParams(temperature=0, max_tokens=m, seed=0, structured_outputs=StructuredOutputsParams(json=schema))  # default guided decoding (VS-E1 reverted)

    def ok(t):
        try: return not list(V.iter_errors(json.loads(t)))
        except Exception: return False
    meta = {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
    for tag, split, idx in SPLITS():
        rgb = np.load(f"{ARR}/{split}_rgb.npy", mmap_mode="r"); fcc = np.load(f"{ARR}/{split}_fcc.npy", mmap_mode="r")
        req = [{"prompt": prompt, "multi_modal_data": {"image": arm_images("B", rgb[i], fcc[i])}} for i in idx]
        t0 = time.time(); o1 = [o.outputs[0] for o in llm.generate(req, sp(3000))]; t1 = time.time()
        bad = [k for k, o in enumerate(o1) if not ok(o.text)]; o2 = {}
        if bad: o2 = dict(zip(bad, [o.outputs[0] for o in llm.generate([req[k] for k in bad], sp(4500))]))  # VS-X1 retry budget
        with open(f"{ST}/captions_{tag}.jsonl", "w") as f:
            for k, (i, o) in enumerate(zip(idx, o1)):
                r = {"k": k, "split_index": int(i), "attempt": 1, "raw": o.text, "finish": o.finish_reason, "n_tokens": len(o.token_ids)}
                if k in o2: r |= {"attempt": 2, "raw1": o.text, "raw": o2[k].text, "finish": o2[k].finish_reason, "n_tokens": len(o2[k].token_ids)}
                r["valid"] = ok(r["raw"]); f.write(json.dumps(r) + "\n")
        meta[tag] = {"n": len(idx), "sec_pass1": round(t1 - t0), "sec_total": round(time.time() - t0), "retried": len(bad),
                     "dropped": sum(not ok(o.text) for o in o2.values())}
        print(tag, meta[tag], flush=True)
    json.dump(meta, open(f"{ST}/captions_meta.json", "w"), indent=1)


def prithvi():
    import torch, torch.nn.functional as Fn
    evict(); src = open(f"{REPO}/scripts/prithvi.py").read()
    ns = {"__name__": "prithvi_lib"}; sys.argv = ["prithvi.py", ARR, ST]
    exec(src[:src.index("@torch.no_grad()")], ns)  # stage-1 model build and normalization, unchanged
    enc, MS_IDX, P_MEAN, P_STD = ns["enc"], ns["MS_IDX"], ns["P_MEAN"], ns["P_STD"]; buf = {}
    for b, blk in enumerate(enc.blocks): blk.register_forward_hook(lambda m, i, o, b=b: buf.__setitem__(b, o[:, 1:].float().mean(1).cpu()))
    for tag, split, idx in SPLITS():
        ms = np.load(f"{ARR}/{split}_ms12.npy", mmap_mode="r"); L, last = [], []; t0 = time.time()
        with torch.no_grad():
            for k in range(0, len(idx), 128):
                y = torch.from_numpy(np.asarray(ms[idx[k:k + 128]])[:, MS_IDX]).cuda()
                y = ((Fn.interpolate(y, size=224, mode="bilinear", align_corners=False) - P_MEAN) / P_STD).unsqueeze(2)
                last.append(enc.forward_features(y)[-1][:, 1:].float().mean(1).cpu()); L.append(torch.stack([buf[b] for b in range(24)], 1))
        L = torch.cat(L).numpy(); np.save(f"{ST}/layers_{tag}.npy", L.astype(np.float32)); np.save(f"{ST}/stage1_{tag}.npy", torch.cat(last).numpy())
        print(tag, L.shape, round(time.time() - t0), "s", flush=True)
    ref = np.load(f"{H}/langres_data/prithvi/test_prithvi_raw.npy"); new = np.load(f"{ST}/stage1_test.npy")
    print("stage-1 test token reproduced: max abs diff", float(np.abs(ref - new).max()), "corr", float(np.corrcoef(ref.ravel(), new.ravel())[0, 1]))


def fields():
    S = sets(); vf = descriptors("rlpool", S["fit_rl"]); meta = {}
    for tag, split, idx in SPLITS():
        v = vf if tag == "fit" else descriptors(split, idx); C = [json.loads(l) for l in open(f"{ST}/captions_{tag}.jsonl")]
        with open(f"{VS}/fields_{tag}.jsonl", "w") as f:
            for k, c in enumerate(C):
                r = {"k": k, "split_index": c["split_index"], "valid": c["valid"]}
                if c["valid"]:
                    j = json.loads(c["raw"]); r["fields"] = dict(zip(FIELDS, caption_fields(j) + [code_text(v[k], vf)])); r["uncertainties"] = j["uncertainties"]
                f.write(json.dumps(r) + "\n")
        meta[tag] = {"n": len(C), "valid": sum(c["valid"] for c in C)}
    np.save(f"{ST}/descriptors_fit.npy", vf); print(meta)


def embed():
    import torch
    from sentence_transformers import SentenceTransformer
    evict(); m = SentenceTransformer(EMB, revision=EREV, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}); t0 = time.time()
    for tag in ("fit", "test"):
        R = [json.loads(l) for l in open(f"{VS}/fields_{tag}.jsonl")]; E = np.full((len(R), 7, m.get_sentence_embedding_dimension()), np.nan, np.float32)
        ok = [k for k, r in enumerate(R) if r["valid"]]
        for j, fl in enumerate(FIELDS):
            E[ok, j] = m.encode([R[k]["fields"][fl] for k in ok], batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        np.save(f"{ST}/emb_{tag}.npy", E); print(tag, E.shape, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    {"captions": captions, "prithvi": prithvi, "fields": fields, "embed": embed}[sys.argv[1]]()
