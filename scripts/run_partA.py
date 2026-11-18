"""Part A runner for one arm on one split: pass 1 (z), B0, B1, B2 zero-shot, and per-instance search.
Also used for LR-2 calibration (--calib) and for RL greedy / RL+search (--lora).
vLLM generates; HF Qwen2.5-VL (frozen, no LoRA) scores pass 1 / pass 2 at layer 21.
Usage:
  run_partA.py --arm A --split calib --out DIR --calib          # zero-shot greedy + means + AUC + layer diag
  run_partA.py --arm A --split val --out DIR [--limit N]        # B0,B1,B2,A_search
  run_partA.py --arm A --split test --out DIR --lora L --tag RL # RL greedy + RL_search
"""
import argparse, json, os, sys, time, zlib
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True); ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--calib", action="store_true"); ap.add_argument("--lora", default=None); ap.add_argument("--tag", default="RL")
ap.add_argument("--limit", type=int, default=None); ap.add_argument("--gpu_mem", type=float, default=0.35)
ap.add_argument("--no_search", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
LOG = open(f"{a.out}/timing.jsonl", "a")
T0 = time.time()


def tick(name, **kw):
    LOG.write(json.dumps({"stage": name, "t": round(time.time() - T0, 1), **kw}) + "\n"); LOG.flush()
    print(name, round(time.time() - T0, 1), kw, flush=True)


import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

pids = ids(a.split)
rgb = np.load(f"{ARR}/{a.split}_rgb.npy", mmap_mode="r"); fcc = np.load(f"{ARR}/{a.split}_fcc.npy", mmap_mode="r")
N = min(len(pids), a.limit or len(pids)); pids = pids[:N]
IMGS = [arm_images(a.arm, rgb[i], fcc[i]) for i in range(N)]
desc = pd.read_parquet(f"{ARR}/descriptors.parquet").set_index("patch_id")

sc = QwenScorer()
tick("hf_loaded")
llm = LLM(VLM, revision=rev(VLM), dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=a.gpu_mem,
          limit_mm_per_prompt={"image": 2}, enable_lora=a.lora is not None, max_lora_rank=16, seed=0,
          mm_processor_kwargs={"min_pixels": IMG_SIDE * IMG_SIDE, "max_pixels": IMG_SIDE * IMG_SIDE})
LR = LoRARequest("policy", 1, a.lora) if a.lora else None
tick("vllm_loaded")


def vreq(i, text):
    p = sc.proc.apply_chat_template([image_user(a.arm, text)], tokenize=False, add_generation_prompt=True)
    return {"prompt": p, "multi_modal_data": {"image": IMGS[i]}}


def seed_of(*xs):
    return zlib.crc32("|".join(map(str, xs)).encode()) & 0x7FFFFFFF


def generate(reqs, n, temp, seeds):
    sps = [SamplingParams(n=n, temperature=temp, max_tokens=160, seed=s) for s in seeds]
    outs = llm.generate(reqs, sps, lora_request=LR)
    return [[c.text.strip() for c in o.outputs] for o in outs]


# ---- pass 1
z = sc.pass1(a.arm, IMGS); np.save(f"{a.out}/z.npy", z.astype(np.float32))
tick("pass1", n=N)

if a.calib:
    # zero-shot greedy summaries on calibration split; all-layer states for the diagnostic sweep
    W = [o[0] for o in generate([vreq(i, DESCRIBE) for i in range(N)], 1, 0.0, [0] * N)]
    tick("zeroshot_gen")
    zh = sc.pass2(W); np.save(f"{a.out}/zhat.npy", zh.astype(np.float32))
    mu_img, mu_txt = z.mean(0), zh.mean(0)
    os.makedirs(CAL, exist_ok=True)
    np.savez(f"{CAL}/means_{a.arm}.npz", mu_img=mu_img, mu_txt=mu_txt, patch_ids=np.array(pids))
    ui, ut = unit(z, mu_img), unit(zh, mu_txt)
    S = ui @ ut.T
    from sklearn.metrics import roc_auc_score
    lab = np.eye(N, dtype=int).ravel()
    auc = float(roc_auc_score(lab, S.ravel()))
    top1 = float((S.argmax(1) == np.arange(N)).mean())
    with open(f"{a.out}/summaries.jsonl", "w") as f:
        for i in range(N):
            f.write(json.dumps({"patch_id": pids[i], "arm": a.arm, "condition": "B2_zeroshot", "split": a.split, "round": 0,
                                "summary": W[i], "n_words": n_words(W[i]), "s": float(S[i, i])}) + "\n")
    # diagnostic only: AUC per layer (never used to pick the layer)
    diag = {}
    def all_layers(texts, images=None):
        kw = dict(text=texts, padding=True, return_tensors="pt")
        if images is not None:
            kw.update(images=images, images_kwargs={"min_pixels": IMG_SIDE * IMG_SIDE, "max_pixels": IMG_SIDE * IMG_SIDE})
        enc = sc.proc(**kw).to("cuda")
        with torch.no_grad():
            hs = sc.model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        return np.stack([h[:, -1].float().cpu().numpy() for h in hs], 1)
    Z1 = np.concatenate([all_layers([sc._prompt(image_user(a.arm, Q))] * len(IMGS[k:k + 16]), [im for ims in IMGS[k:k + 16] for im in ims]) for k in range(0, N, 16)])
    Z2 = np.concatenate([all_layers([sc._prompt(pass2_user(w)) for w in W[k:k + 32]]) for k in range(0, N, 32)])
    for L in range(Z1.shape[1]):
        S_L = unit(Z1[:, L], Z1[:, L].mean(0)) @ unit(Z2[:, L], Z2[:, L].mean(0)).T
        diag[L] = float(roc_auc_score(lab, S_L.ravel()))
    res = {"arm": a.arm, "n": N, "auc_own_vs_other": auc, "top1_retrieval": top1, "mean_s_own": float(np.diag(S).mean()),
           "mean_s_other": float((S.sum() - np.trace(S)) / (N * N - N)), "layer": LAYER, "diag_auc_per_layer": diag,
           "check_layer21_matches_sweep_value": abs(diag[LAYER] - auc) < 1e-3}
    json.dump(res, open(f"{a.out}/calib_result.json", "w"), indent=1)
    tick("calib_done", auc=auc)
    sys.exit(0)

mu_img, mu_txt = load_means(a.arm)
ui = unit(z, mu_img); np.save(f"{a.out}/u_img.npy", ui.astype(np.float32))
CAND = open(f"{a.out}/candidates.jsonl", "a")


def score(texts, idx):
    ut = unit(sc.pass2(texts), mu_txt)
    return np.einsum("nd,nd->n", ui[idx], ut), ut


def save_final(cond, W, s, ut, extra=None):
    np.save(f"{a.out}/{cond}_u_txt.npy", ut.astype(np.float32))
    with open(f"{a.out}/{cond}_final.jsonl", "w") as f:
        for i in range(N):
            f.write(json.dumps({"patch_id": pids[i], "arm": a.arm, "condition": cond, "split": a.split, "summary": W[i],
                                "n_words": n_words(W[i]), "s": float(s[i]), **(extra[i] if extra else {})}) + "\n")


ALL = np.arange(N)
if not a.lora:
    W0 = [B0_TEXT] * N
    s0, ut0 = score(W0, ALL); save_final("B0_generic", W0, s0, ut0)
    W1 = ["Land cover: " + ", ".join(desc.loc[p, "labels"]) + "." for p in pids]  # LR-X19: comma-joined, final period
    s1, ut1 = score(W1, ALL); save_final("B1_labels", W1, s1, ut1)
    tick("B0_B1")
greedy_cond = f"{a.tag}_greedy" if a.lora else "B2_zeroshot"
W2 = [o[0] for o in generate([vreq(i, DESCRIBE) for i in range(N)], 1, 0.0, [0] * N)]
s2, ut2 = score(W2, ALL); save_final(greedy_cond, W2, s2, ut2)
tick("greedy", cond=greedy_cond)
if a.no_search:
    sys.exit(0)

# ---- search: round 0 = 32 describe samples, rounds 1-3 = 8 rewrite samples from current top 4
search_cond = f"{a.tag}_search" if a.lora else "A_search"
pool = [[] for _ in range(N)]  # (w, s, round, j)
utpool = [[] for _ in range(N)]
for rnd in range(4):
    if rnd == 0:
        reqs = [vreq(i, DESCRIBE) for i in range(N)]; n = 32
    else:
        reqs = []
        for i in range(N):
            top = sorted(pool[i], key=lambda t: -t[1])[:4]
            reqs.append(vreq(i, rewrite_text([(w, s) for w, s, _, _ in top])))
        n = 8
    outs = generate(reqs, n, 1.0, [seed_of(search_cond, a.arm, pids[i], rnd) for i in range(N)])
    texts = [w for o in outs for w in o]; idx = np.repeat(ALL, n)
    s, ut = score(texts, idx)
    k = 0
    for i in range(N):
        for j in range(n):
            pool[i].append((texts[k], float(s[k]), rnd, j)); utpool[i].append(ut[k].astype(np.float16))
            CAND.write(json.dumps({"patch_id": pids[i], "arm": a.arm, "condition": search_cond, "split": a.split, "round": rnd,
                                   "j": j, "summary": texts[k], "n_words": n_words(texts[k]), "s": float(s[k])}) + "\n")
            k += 1
    CAND.flush()
    tick(f"search_round{rnd}", n_gen=len(texts), mean_best=float(np.mean([max(t[1] for t in p) for p in pool])))

Wf, sf, utf, extra = [], [], [], []
for i in range(N):
    ok = [k for k, t in enumerate(pool[i]) if n_words(t[0]) <= 100]
    fallback = not ok
    cand = ok if ok else list(range(len(pool[i])))  # LR-X20: if no candidate <=100 words, take best overall, flagged
    b = max(cand, key=lambda k: pool[i][k][1])
    Wf.append(pool[i][b][0]); sf.append(pool[i][b][1]); utf.append(utpool[i][b].astype(np.float32))
    extra.append({"pick_round": pool[i][b][2], "n_candidates": len(pool[i]), "n_le100": len(ok), "fallback_over100": fallback})
save_final(search_cond, Wf, np.array(sf), np.stack(utf), extra)
tick("search_done", n=N)
