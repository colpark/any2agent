"""Compare policies on one patch with one generator (HF generate) and both scorers (LR-X24/25 follow-up, eval only).
For each policy (base, RFT, RL checkpoints): greedy + K samples at T=1 (fixed seed), Qwen s (LoRA off), word stats,
diversity, base-model NLL per token of the text given the image prompt (fluency under the original model), then InternVL s'.
Usage: eval_single.py --idx 0 --out FILE.json --policy name=path ... [--ref name=jsonl ...]
"""
import argparse, json, os, re, sys, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="A"); ap.add_argument("--idx", type=int, default=0); ap.add_argument("--out", required=True)
ap.add_argument("--policy", nargs="+", required=True); ap.add_argument("--ref", nargs="*", default=[]); ap.add_argument("--k", type=int, default=32)
a = ap.parse_args()

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

rgb = np.load(f"{ARR}/rlpool_rgb.npy", mmap_mode="r"); fcc = np.load(f"{ARR}/rlpool_fcc.npy", mmap_mode="r")
IMGS = arm_images(a.arm, rgb[a.idx], fcc[a.idx]); pid = ids("rlpool")[a.idx]
IK = {"min_pixels": IMG_SIDE * IMG_SIDE, "max_pixels": IMG_SIDE * IMG_SIDE}
proc = AutoProcessor.from_pretrained(VLM, revision=rev(VLM)); proc.tokenizer.padding_side = "left"
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM, revision=rev(VLM), dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
evict_model_cache(VLM)
PROMPT = proc.apply_chat_template([image_user(a.arm, DESCRIBE)], tokenize=False, add_generation_prompt=True)
mu_img, mu_txt = load_means(a.arm)


@torch.no_grad()
def gen(model, n, greedy):
    enc = proc(text=[PROMPT] * n, images=IMGS * n, images_kwargs=IK, padding=True, return_tensors="pt").to("cuda")
    kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
    out = model.generate(**enc, max_new_tokens=160, **kw)
    return [t.strip() for t in proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]


@torch.no_grad()
def nll(texts):
    r = []
    plen = proc(text=[PROMPT], images=IMGS, images_kwargs=IK, return_tensors="pt")["input_ids"].shape[1]
    for t in texts:
        enc = proc(text=[PROMPT + t + "<|im_end|>\n"], images=IMGS, images_kwargs=IK, return_tensors="pt").to("cuda")
        lab = enc["input_ids"].clone(); lab[:, :plen] = -100
        r.append(float(base(**enc, labels=lab).loss))
    return r


texts = {}  # name -> {"greedy": str, "samples": [..]}
model = base
for spec in a.policy:
    name, path = spec.split("=", 1)
    torch.manual_seed(0)
    if path == "base":
        m = base
    else:
        m = PeftModel.from_pretrained(base, path).eval()
    texts[name] = {"greedy": gen(m, 1, True)[0], "samples": gen(m, a.k, False)}
    if path != "base":
        base = m.unload()  # back to the plain model, LoRA removed
    print("generated", name, flush=True)
for spec in a.ref:
    name, path = spec.split("=", 1)
    w = json.loads(open(path).readline())["summary"]
    texts[name] = {"greedy": w, "samples": []}

sc = QwenScorer(model=base, proc=proc)
UI = unit(sc.pass1(a.arm, [IMGS]), mu_img)[0]
flat = [(n, kind, j, w) for n, d in texts.items() for kind, ws in [("greedy", [d["greedy"]]), ("samples", d["samples"])] for j, w in enumerate(ws)]
S = unit(sc.pass2([f[3] for f in flat]), mu_txt) @ UI
NLL = nll([f[3] for f in flat])
del sc, base; torch.cuda.empty_cache()

from check_scorer import InternScorer
cs = InternScorer(); cm = np.load(f"{CAL}/check_means_{a.arm}.npz")
UIc = unit(cs.pass1(a.arm, [IMGS]), cm["mu_img"])[0]
SP = unit(cs.pass2([f[3] for f in flat]), cm["mu_txt"]) @ UIc

tok = lambda t: set(re.findall(r"[a-z]+", t.lower()))
J = lambda x, y: len(x & y) / max(1, len(x | y))
res = {"patch_id": pid, "policies": {}}
for n in texts:
    g = [k for k, f in enumerate(flat) if f[0] == n and f[1] == "greedy"][0]
    ss = [k for k, f in enumerate(flat) if f[0] == n and f[1] == "samples"]
    r = {"greedy": {"text": flat[g][3], "s": float(S[g]), "s_prime": float(SP[g]), "words": n_words(flat[g][3]), "base_nll": NLL[g]}}
    if ss:
        W = [flat[k][3] for k in ss]; nw = np.array([n_words(w) for w in W]); sets = [tok(w) for w in W]
        ok = [k for k in ss if n_words(flat[k][3]) <= 100]
        r["samples"] = {"s_mean": float(S[ss].mean()), "s_max_le100": float(S[ok].max()) if ok else None,
                        "s_prime_mean": float(SP[ss].mean()), "corr_s_sprime": float(np.corrcoef(S[ss], SP[ss])[0, 1]),
                        "words_mean": float(nw.mean()), "frac_over100": float((nw > 100).mean()),
                        "distinct": len(set(W)), "jaccard_within": float(np.mean([J(x, y) for x, y in itertools.combinations(sets, 2)])),
                        "base_nll_mean": float(np.mean([NLL[k] for k in ss]))}
    res["policies"][n] = r
json.dump({"texts": texts, **res}, open(a.out, "w"), indent=1)
for n, r in res["policies"].items():
    print(n, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in {**{f"g_{x}": y for x, y in r["greedy"].items() if x != "text"}, **r.get("samples", {})}.items()}))
