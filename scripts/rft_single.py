"""LR-X25 single-patch rejection-sampling fine-tuning (expert iteration) sanity test, user-approved 2026-11-18.
Each round: sample K describe summaries from the current policy (LoRA on, T=1), score s with the frozen scorer
(LoRA off), add to the pool, take the best <=100-word candidate in the pool as the target, run SFT steps on it
(loss on completion tokens only), then measure greedy s and the mean s of fresh samples.
Usage: rft_single.py --arm A --idx 0 --out DIR [--rounds 4] [--k 32] [--sft_steps 30] [--lr 1e-4]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="A"); ap.add_argument("--idx", type=int, default=0); ap.add_argument("--out", required=True)
ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--k", type=int, default=32)
ap.add_argument("--sft_steps", type=int, default=30); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

pid = ids("rlpool")[a.idx]
rgb = np.load(f"{ARR}/rlpool_rgb.npy", mmap_mode="r"); fcc = np.load(f"{ARR}/rlpool_fcc.npy", mmap_mode="r")
IMGS = arm_images(a.arm, rgb[a.idx], fcc[a.idx])
mu_img, mu_txt = load_means(a.arm)
proc = AutoProcessor.from_pretrained(VLM, revision=rev(VLM)); proc.tokenizer.padding_side = "left"
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM, revision=rev(VLM), dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
evict_model_cache(VLM)
pm = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)"))
sc = QwenScorer(model=pm, proc=proc)
IK = {"min_pixels": IMG_SIDE * IMG_SIDE, "max_pixels": IMG_SIDE * IMG_SIDE}
PROMPT = proc.apply_chat_template([image_user(a.arm, DESCRIBE)], tokenize=False, add_generation_prompt=True)
LOG = open(f"{a.out}/rounds.jsonl", "a"); CAND = open(f"{a.out}/candidates.jsonl", "a")

with torch.no_grad(), pm.disable_adapter():
    pm.eval(); UI = unit(sc.pass1(a.arm, [IMGS]), mu_img)[0]


def score(texts):
    pm.eval()
    with torch.no_grad(), pm.disable_adapter():
        return unit(sc.pass2(texts), mu_txt) @ UI


@torch.no_grad()
def gen(n, greedy=False):
    pm.eval()
    enc = proc(text=[PROMPT] * n, images=IMGS * n, images_kwargs=IK, padding=True, return_tensors="pt").to("cuda")
    kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
    out = pm.generate(**enc, max_new_tokens=160, **kw)
    return [t.strip() for t in proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]


def sft(target, steps):
    pm.train()
    full = proc(text=[PROMPT + target + "<|im_end|>\n"], images=IMGS, images_kwargs=IK, return_tensors="pt").to("cuda")
    plen = proc(text=[PROMPT], images=IMGS, images_kwargs=IK, return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone(); labels[:, :plen] = -100
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        loss = pm(**full, labels=labels).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(pm.parameters(), 1.0); opt.step(); opt.zero_grad()
        losses.append(float(loss))
    return losses


pool = []  # (w, s, round)
T0 = time.time()
for rnd in range(a.rounds + 1):
    g = gen(1, greedy=True)[0]; gs = float(score([g])[0])
    samp = gen(a.k); ss = score(samp)
    for j, (w, v) in enumerate(zip(samp, ss)):
        pool.append((w, float(v), rnd))
        CAND.write(json.dumps({"patch_id": pid, "arm": a.arm, "condition": "RFT_sample", "round": rnd, "j": j, "summary": w,
                               "n_words": n_words(w), "s": float(v)}) + "\n")
    CAND.flush()
    nw = np.array([n_words(w) for w in samp])
    rec = {"round": rnd, "t": round(time.time() - T0, 1), "greedy_s": gs, "greedy_words": n_words(g), "greedy": g,
           "sample_mean_s": float(ss.mean()), "sample_max_s": float(ss.max()), "sample_frac_over100": float((nw > 100).mean()),
           "sample_mean_words": float(nw.mean())}
    if rnd < a.rounds:
        ok = [p for p in pool if n_words(p[0]) <= 100]
        tw, tsc, tr = max(ok, key=lambda p: p[1])
        rec.update({"target_s": tsc, "target_round": tr, "target": tw, "sft_loss": sft(tw, a.sft_steps)})
    LOG.write(json.dumps(rec) + "\n"); LOG.flush()
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("greedy", "target", "sft_loss")}), flush=True)
pm.save_pretrained(f"{a.out}/lora_final")
print("DONE", round(time.time() - T0, 1))
