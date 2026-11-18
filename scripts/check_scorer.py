"""Check VLM (InternVL3-8B, frozen, eval only): pass 1 / pass 2 at its own layer 21 of 28, own calibration means.
Usage:
  check_scorer.py calib --arm A --calib_dir RUN_DIR      # means from calib split + the calib zero-shot summaries
  check_scorer.py score --arm A --split test --run_dir DIR [--files X_final.jsonl ...]  # adds s_prime
Prompt (LR-X21): the -hf processor's shipped chat template (no system message), same user content and prefill as Qwen;
each 448x448 image = 1 tile = 256 tokens.
"""
import argparse, json, os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from common import *

CHECK = "OpenGVLab/InternVL3-8B-hf"  # LR-X22: official HF-native conversion of OpenGVLab/InternVL3-8B
LAYER_C = 21


class InternScorer:
    def __init__(self, dev="cuda"):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.proc = AutoProcessor.from_pretrained(CHECK, revision=rev(CHECK))
        self.proc.tokenizer.padding_side = "left"
        self.m = AutoModelForImageTextToText.from_pretrained(CHECK, revision=rev(CHECK), dtype=torch.bfloat16,
                                                             attn_implementation="sdpa").to(dev).eval()
        self.dev = dev

    def prompt(self, user):
        return self.proc.apply_chat_template([user], tokenize=False, add_generation_prompt=True) + PREFILL

    @torch.no_grad()
    def _run(self, texts, images=None):
        enc = self.proc(text=texts, images=images, return_tensors="pt", padding=True).to(self.dev)
        if "pixel_values" in enc:
            enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        o = self.m(**enc, output_hidden_states=True, use_cache=False)
        return o.hidden_states[LAYER_C][:, -1].float().cpu().numpy()

    def pass1(self, arm, img_lists, bs=8):
        return np.concatenate([self._run([self.prompt(image_user(arm, Q))] * len(img_lists[k:k + bs]),
                                         [im for ims in img_lists[k:k + bs] for im in ims]) for k in range(0, len(img_lists), bs)])

    def pass2(self, summaries, bs=32):
        return np.concatenate([self._run([self.prompt(pass2_user(w)) for w in summaries[k:k + bs]]) for k in range(0, len(summaries), bs)])


def load_imgs(arm, split, n=None):
    rgb = np.load(f"{ARR}/{split}_rgb.npy", mmap_mode="r"); fcc = np.load(f"{ARR}/{split}_fcc.npy", mmap_mode="r")
    n = n or len(rgb)
    return [arm_images(arm, rgb[i], fcc[i]) for i in range(n)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--arm", required=True)
    ap.add_argument("--calib_dir"); ap.add_argument("--split"); ap.add_argument("--run_dir"); ap.add_argument("--files", nargs="*")
    a = ap.parse_args()
    sc = InternScorer()
    if a.mode == "calib":
        rows = [json.loads(l) for l in open(f"{a.calib_dir}/summaries.jsonl")]
        assert [r["patch_id"] for r in rows] == ids("calib")
        z = sc.pass1(a.arm, load_imgs(a.arm, "calib")); zh = sc.pass2([r["summary"] for r in rows])
        np.savez(f"{CAL}/check_means_{a.arm}.npz", mu_img=z.mean(0), mu_txt=zh.mean(0))
        S = unit(z, z.mean(0)) @ unit(zh, zh.mean(0)).T
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(np.eye(len(S), dtype=int).ravel(), S.ravel()))
        json.dump({"arm": a.arm, "check_auc_own_vs_other": auc, "top1": float((S.argmax(1) == np.arange(len(S))).mean())},
                  open(f"{a.calib_dir}/check_calib_result.json", "w"), indent=1)
        print("check calib auc", auc)
    else:
        m = np.load(f"{CAL}/check_means_{a.arm}.npz")
        files = a.files or sorted(glob.glob(f"{a.run_dir}/*_final.jsonl"))
        rows0 = [json.loads(l) for l in open(files[0])]
        n = len(rows0)
        ui = unit(sc.pass1(a.arm, load_imgs(a.arm, a.split, n)), m["mu_img"])
        np.save(f"{a.run_dir}/check_u_img.npy", ui.astype(np.float32))
        for f in files:
            rows = [json.loads(l) for l in open(f)]
            assert [r["patch_id"] for r in rows] == [r["patch_id"] for r in rows0]
            ut = unit(sc.pass2([r["summary"] for r in rows]), m["mu_txt"])
            sp = np.einsum("nd,nd->n", ui, ut)
            with open(f.replace("_final.jsonl", "_final_check.jsonl"), "w") as fo:
                for r, v in zip(rows, sp):
                    fo.write(json.dumps({"patch_id": r["patch_id"], "s_prime": float(v)}) + "\n")
            print(os.path.basename(f), "mean s'", float(sp.mean()), flush=True)
