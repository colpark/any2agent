"""Rescore a summaries file (e.g. paraphrases) with the frozen Qwen pass 2 against cached u_img.
Usage: rescore.py --run DIR --arm A --infile DIR/X_para_final.jsonl
Writes X_para_final.jsonl in place with updated s, and X_para_u_txt.npy.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--arm", required=True); ap.add_argument("--infile", required=True)
a = ap.parse_args()
R = [json.loads(l) for l in open(a.infile)]
ui = np.load(f"{a.run}/u_img.npy")[:len(R)]
_, mu_txt = load_means(a.arm)
ut = unit(QwenScorer().pass2([r["summary"] for r in R]), mu_txt)
s = np.einsum("nd,nd->n", ui, ut)
np.save(a.infile.replace("_final.jsonl", "_u_txt.npy"), ut.astype(np.float32))
with open(a.infile, "w") as f:
    for r, v in zip(R, s):
        f.write(json.dumps({**r, "s": float(v)}) + "\n")
print(os.path.basename(a.infile), "mean s", float(s.mean()))
