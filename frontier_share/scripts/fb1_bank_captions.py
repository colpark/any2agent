"""FB-1: caption the 1,000 bank patches with Qwen2.5-VL-7B-Instruct (pinned) on vLLM, greedy, max 160 new tokens,
arm B describe prompt and inputs (true color + false color, preamble) from scripts/common.py. Output: ~/langres_state/fb/bank_captions.jsonl"""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.expanduser("~/Documents/langres/scripts"))
import numpy as np
from common import *
subprocess.run([sys.executable, f"{REPO}/scripts/evict_cache.py", "~/.cache/huggingface/hub", "~/langres_data", "~/langres_state"])
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
bank = json.load(open(f"{STATE}/fb/bank.json")); idx = bank["rlpool_index"]
rgb = np.load(f"{ARR}/rlpool_rgb.npy", mmap_mode="r"); fcc = np.load(f"{ARR}/rlpool_fcc.npy", mmap_mode="r")
proc = AutoProcessor.from_pretrained(VLM, revision=rev(VLM))
prompt = proc.apply_chat_template([image_user("B", DESCRIBE)], tokenize=False, add_generation_prompt=True)
llm = LLM(VLM, revision=rev(VLM), dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.5, limit_mm_per_prompt={"image": 2}, seed=0,
          mm_processor_kwargs={"min_pixels": IMG_SIDE * IMG_SIDE, "max_pixels": IMG_SIDE * IMG_SIDE})
evict_model_cache(VLM)
t0 = time.time()
reqs = [{"prompt": prompt, "multi_modal_data": {"image": arm_images("B", rgb[i], fcc[i])}} for i in idx]
outs = llm.generate(reqs, SamplingParams(n=1, temperature=0.0, max_tokens=160, seed=0))
W = [o.outputs[0].text.strip() for o in outs]; FR = [o.outputs[0].finish_reason for o in outs]
with open(f"{STATE}/fb/bank_captions.jsonl", "w") as f:
    for k, (p, w, fr) in enumerate(zip(bank["patch_ids"], W, FR)):
        f.write(json.dumps({"bank_id": f"b{k:04d}", "patch_id": p, "caption": w, "n_words": n_words(w), "finish_reason": fr}) + "\n")
nw = np.array([n_words(w) for w in W])
print(json.dumps({"n": len(W), "sec": round(time.time() - t0), "words_mean": float(nw.mean()), "words_max": int(nw.max()),
                  "over100": int((nw > 100).sum()), "finish_length": FR.count("length"), "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest()}))
