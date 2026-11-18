"""Part B: GRPO for one arm (TRL GRPOTrainer).
Design: LoRA r16 a32 on LM q,k,v,o; 4 samples/patch; 32 patches/step; LoRA lr 1e-5; clip 0.2; beta 0.04 (adapter off as ref);
450 steps over the RL pool; one seed per arm; reward = s(w) - 0.5*[words > 100]; scorer = same model with LoRA OFF
(pass 1 cached per patch, pass 2 per rollout), frozen calibration means.
LR-X11 knobs: loss_type grpo, scale_rewards group, num_iterations 1, constant LR, no warmup, max_grad_norm 1.0,
LoRA dropout 0, top_p 1, top_k off, Qwen default system prompt.
Usage: grpo_train.py --arm A --out DIR [--steps 450] [--save_steps 150]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=450); ap.add_argument("--save_steps", type=int, default=150)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--micro_bs", type=int, default=16)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainerCallback
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

pids = ids("rlpool")
rgb = np.load(f"{ARR}/rlpool_rgb.npy", mmap_mode="r"); fcc = np.load(f"{ARR}/rlpool_fcc.npy", mmap_mode="r")
mu_img, mu_txt = load_means(a.arm)


class PatchDS(torch.utils.data.Dataset):
    def __len__(self):
        return len(pids)

    def __getitem__(self, i):
        msg = [image_user(a.arm, DESCRIBE)]
        it = iter(arm_images(a.arm, rgb[i], fcc[i]))
        for part in msg[0]["content"]:
            if part["type"] == "image":
                part["image"] = next(it)
        return {"prompt": msg, "idx": i}


proc = AutoProcessor.from_pretrained(VLM, revision=rev(VLM)); proc.tokenizer.padding_side = "left"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM, revision=rev(VLM), dtype=torch.bfloat16, attn_implementation="sdpa")
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                  target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)")
ROLL = open(f"{a.out}/rollouts.jsonl", "a")
UI = {}  # patch index -> u_img (base model, adapter off)
TR = {}


@torch.no_grad()
def reward_fn(prompts, completions, idx, **kw):
    pm = TR["t"].model
    texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    sc = TR.get("sc") or QwenScorer(model=pm, proc=proc); TR["sc"] = sc
    was_training = pm.training; pm.eval()
    with pm.disable_adapter():
        new = sorted({i for i in idx if i not in UI})
        if new:
            z = sc.pass1(a.arm, [arm_images(a.arm, rgb[i], fcc[i]) for i in new])
            for i, u in zip(new, unit(z, mu_img)):
                UI[i] = u
        ut = unit(sc.pass2(texts), mu_txt)
    if was_training:
        pm.train()
    s = np.einsum("nd,nd->n", np.stack([UI[i] for i in idx]), ut)
    nw = np.array([n_words(t) for t in texts])
    rew = s - 0.5 * (nw > 100)
    st = kw.get("trainer_state"); step = st.global_step if st is not None else -1
    for j, t in enumerate(texts):
        ROLL.write(json.dumps({"step": step, "arm": a.arm, "condition": "RL_rollout", "patch_id": pids[idx[j]], "summary": t,
                               "n_words": int(nw[j]), "s": float(s[j]), "reward": float(rew[j])}) + "\n")
    ROLL.flush()
    return rew.tolist()


class LogCB(TrainerCallback):
    def __init__(self):
        self.f = open(f"{a.out}/train_log.jsonl", "a"); self.t = time.time()

    def on_log(self, args, st, control, logs=None, **kw):
        self.f.write(json.dumps({"step": st.global_step, "wall_s": round(time.time() - self.t, 1), **(logs or {})}) + "\n"); self.f.flush()


N_GEN, PATCHES = 4, 32
GEN_BS = N_GEN * PATCHES
cfg = GRPOConfig(
    output_dir=a.out, max_steps=a.steps, learning_rate=1e-5, beta=0.04, epsilon=0.2, num_generations=N_GEN,
    per_device_train_batch_size=a.micro_bs, gradient_accumulation_steps=GEN_BS // a.micro_bs, generation_batch_size=GEN_BS,
    num_iterations=1, temperature=1.0, top_p=1.0, top_k=0, max_completion_length=160, loss_type="grpo", scale_rewards="group",
    lr_scheduler_type="constant", warmup_steps=0, weight_decay=0.0, max_grad_norm=1.0, bf16=True, gradient_checkpointing=True,
    logging_steps=1, save_strategy="steps", save_steps=a.save_steps, report_to=[], remove_unused_columns=False,
    seed=a.seed, data_seed=a.seed, use_vllm=False, log_completions=False, dataloader_num_workers=2,
)
trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn], args=cfg, train_dataset=PatchDS(), processing_class=proc,
                      peft_config=lcfg, callbacks=[LogCB()])
TR["t"] = trainer
json.dump({"args": vars(a), "grpo_config": {k: str(v) for k, v in cfg.to_dict().items()}}, open(f"{a.out}/config.json", "w"), indent=1)
t0 = time.time(); trainer.train()
trainer.model.save_pretrained(f"{a.out}/lora_final")
json.dump({"train_sec": round(time.time() - t0, 1), "steps": a.steps}, open(f"{a.out}/done.json", "w"))
print("DONE", time.time() - t0)
