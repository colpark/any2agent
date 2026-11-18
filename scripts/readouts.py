"""Readouts 1-4 for one arm's run dir (CPU). Usage: readouts.py RUN_DIR SPLIT [--conds c1 c2 ...]
1 s, g: mean, 95% paired bootstrap CI (1,000 resamples, same indices for every condition), paired diffs.
2 s': mean per condition, Pearson and Spearman corr(s, s') across patches.
3 Prithvi decomposition: ridge u_img -> z_prithvi and u_txt -> z_prithvi; 5-fold CV, fixed folds (KFold shuffle seed 0),
  alpha chosen inside each training fold by RidgeCV (efficient LOO) over a fixed grid; R2 on pooled out-of-fold preds.
4 Probes: same folds; ridge (RidgeCV inside fold) for NDVI/NDWI/NDMI; logistic per label (C=1, LR-X12);
  features standardized inside fold; (a) u_txt (b) r=u_img-u_txt (c) u_img (d) [u_txt, r].
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, f1_score, average_precision_score
from joblib import Parallel, delayed
from common import *

ALPHAS = np.logspace(-2, 5, 15)
ORDER = ["B0_generic", "B1_labels", "B2_zeroshot", "A_search", "RL_greedy", "RL_search"]


def rows(path):
    return [json.loads(l) for l in open(path)]


def boot_idx(n, B=1000, seed=0):
    return np.random.default_rng(seed).integers(0, n, (B, n))


def ci(x, I):
    m = x[I].mean(1); return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def folds(n):
    return list(KFold(5, shuffle=True, random_state=0).split(np.arange(n)))


def cv_ridge(X, Y, F):
    P = np.zeros_like(Y, dtype=np.float64)
    for tr, te in F:
        sc = StandardScaler().fit(X[tr])
        P[te] = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), Y[tr]).predict(sc.transform(X[te]))
    return P


def _logit_fold(X, y, tr, te):
    if y[tr].min() == y[tr].max():
        return te, np.full(len(te), float(y[tr][0]))
    sc = StandardScaler().fit(X[tr])
    return te, LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]


def probes(X, desc, labels, F):
    out = {}
    D = desc[["ndvi", "ndwi", "ndmi"]].to_numpy()
    P = cv_ridge(X, D, F)
    for j, q in enumerate(["ndvi", "ndwi", "ndmi"]):
        out[f"{q}_r2"] = float(r2_score(D[:, j], P[:, j]))
    Y = np.array([[l in set(ls) for l in labels] for ls in desc["labels"]], dtype=int)
    present = [j for j in range(len(labels)) if 0 < Y[:, j].sum() < len(Y)]
    jobs = [(j, tr, te) for j in present for tr, te in F]
    res = Parallel(n_jobs=24)(delayed(_logit_fold)(X, Y[:, j], tr, te) for j, tr, te in jobs)
    PR = np.zeros(Y.shape)
    for (j, _, _), (te, p) in zip(jobs, res):
        PR[te, j] = p
    out["label_macroF1"] = float(f1_score(Y[:, present], (PR[:, present] >= 0.5).astype(int), average="macro", zero_division=0))
    out["label_mAP"] = float(np.mean([average_precision_score(Y[:, j], PR[:, j]) for j in present]))
    out["n_labels_scored"] = len(present)
    return out


def main(run, split, conds=None, skip_probes=False):
    pids = ids(split)
    conds = conds or [c for c in ORDER if os.path.exists(f"{run}/{c}_final.jsonl")]
    N = len(rows(f"{run}/{conds[0]}_final.jsonl")); pids = pids[:N]
    desc = pd.read_parquet(f"{ARR}/descriptors.parquet").set_index("patch_id").loc[pids]
    rl = pd.read_parquet(f"{ARR}/descriptors.parquet"); rl = rl[rl.split == "rlpool"]
    labels = sorted({l for ls in rl["labels"] for l in ls})
    zp = np.load(f"{ARR}/../prithvi/{split}_prithvi.npy")[:N]
    ui = np.load(f"{run}/u_img.npy")[:N]
    I, F = boot_idx(N), folds(N)
    S, SP, G, out = {}, {}, {}, {"split": split, "n": N, "conditions": conds}
    for c in conds:
        R = rows(f"{run}/{c}_final.jsonl"); assert [r["patch_id"] for r in R] == pids
        S[c] = np.array([r["s"] for r in R])
        cf = f"{run}/{c}_final_check.jsonl"
        if os.path.exists(cf):
            SP[c] = np.array([r["s_prime"] for r in rows(cf)])
    if "B0_generic" in S:
        for c in conds:
            G[c] = (S[c] - S["B0_generic"]) / (1 - S["B0_generic"])
    out["s"] = {c: ci(S[c], I) for c in conds}
    out["g"] = {c: ci(G[c], I) for c in G}
    out["s_prime"] = {c: ci(SP[c], I) for c in SP}
    out["corr_s_sprime"] = {c: {"pearson": float(pearsonr(S[c], SP[c])[0]), "spearman": float(spearmanr(S[c], SP[c])[0])} for c in SP}
    pairs = [(x, y) for x in conds for y in conds if ORDER.index(x) > ORDER.index(y)]
    out["paired_diff_s"] = {f"{x}-{y}": ci(S[x] - S[y], I) for x, y in pairs}
    out["paired_diff_sprime"] = {f"{x}-{y}": ci(SP[x] - SP[y], I) for x, y in pairs if x in SP and y in SP}
    # 3 Prithvi decomposition
    out["prithvi_r2"] = {"u_img": float(r2_score(zp, cv_ridge(ui, zp, F)))}
    UT = {c: np.load(f"{run}/{c}_u_txt.npy")[:N] for c in conds}
    for c in conds:
        out["prithvi_r2"][f"u_txt|{c}"] = float(r2_score(zp, cv_ridge(UT[c], zp, F))) if c != "B0_generic" else None
    # 4 probes
    if not skip_probes:
        out["probes"] = {"c_u_img": probes(ui, desc, labels, F)}
        for c in conds:
            if c == "B0_generic":
                continue  # constant text: (a) carries nothing; (b) == (c) up to a constant shift
            r = ui - UT[c]
            pa, pb, pd_ = probes(UT[c], desc, labels, F), probes(r, desc, labels, F), probes(np.concatenate([UT[c], r], 1), desc, labels, F)
            out["probes"][c] = {"a_u_txt": pa, "b_r": pb, "d_u_txt_r": pd_,
                                "gain_d_minus_a": {k: pd_[k] - pa[k] for k in pa if k != "n_labels_scored"}}
    json.dump(out, open(f"{run}/readouts_{split}.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("split"); ap.add_argument("--conds", nargs="*")
    ap.add_argument("--skip_probes", action="store_true")
    a = ap.parse_args()
    o = main(a.run, a.split, a.conds, a.skip_probes)
    print(json.dumps({k: o[k] for k in ["s", "g", "s_prime", "prithvi_r2"]}, indent=1))
