"""QC-3 (CPU): answerers, formula, temperature (practice labels only), held-out scoring and comparisons -> ~/langres_state/qc/qc3.npz, qcond/qc3.json.
True labels enter only the true-label heads (labels equivalent), never the answerers, formula or temperature."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold
from qc_common import *
DRY = os.environ.get("QC_DRY", "")  # dry run: soft labels with gaps filled, outputs suffixed, discarded
CS, ALPHAS = [0.01, 0.1, 1], np.logspace(-3, 3, 13)


def practice_labels(Q):
    """Binary label and weight per practice question: semantic soft (P >= 0.5, |2P - 1|), physical exact (weight 1)."""
    P = np.load(f"{ST}/soft_practice{DRY}.npy"); K = np.load(f"{ST}/phys_practice.npy"); pid = [q["id"] for q in json.load(open(f"{QC}/physical.json"))["questions"]]
    sem = [q["id"] for q in Q if q["kind"] == "semantic"]; Y, W = [], []
    for q in Q:
        if q["kind"] == "semantic": p = P[:, sem.index(q["id"])]; Y.append(p >= .5); W.append(np.abs(2 * p - 1))
        else: Y.append(K[:, pid.index(q["id"])]); W.append(np.ones(300))
    return np.column_stack(Y), np.column_stack(W)


def wll(y, p, w):
    p = np.clip(p, 1e-7, 1 - 1e-7); return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())


def cv_loss(X, y, w, C):
    out = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        a, b = logit_fit(X[tr], y[tr], w[tr], C); out.append(wll(y[te], sig(X[te] @ a + b), w[te]))
    return float(np.mean(out))


def ridge(E, T, a):
    """Ridge with intercept from embeddings E (q x 2560) to parameters T (q x 4097), kernel form. Returns M (2560 x 4097), c (4097)."""
    me, mt = E.mean(0), T.mean(0); Ec = E - me; M = Ec.T @ np.linalg.solve(Ec @ Ec.T + a * np.eye(len(E)), T - mt); return M, mt - me @ M


def main():
    t0 = time.time(); Qall = questions(); Fp, Ft = feats(); Eall = np.load(f"{ST}/emb_all.npy")
    PQ = [q for q in Qall if q["split"] == "practice"]; HQ = [q for q in Qall if q["split"] == "held_out"]
    Y, W = practice_labels(PQ); n1 = Y.sum(0); keep = np.flatnonzero((n1 >= 10) & (300 - n1 >= 10)); skipped = [PQ[j]["id"] for j in range(len(PQ)) if j not in keep]
    L = {C: Parallel(12)(delayed(cv_loss)(Fp, Y[:, j], W[:, j], C) for j in keep) for C in CS}; C = min(CS, key=lambda c: np.mean(L[c]))
    TH_ = np.array(Parallel(12)(delayed(lambda j: np.r_[logit_fit(Fp, Y[:, j], W[:, j], C)])(j) for j in keep))  # 4096 weights + bias
    Ep = Eall[[PQ[j]["emb_row"] for j in keep]]; Yk, Wk = Y[:, keep], W[:, keep]; Fp1 = np.c_[Fp, np.ones(300)]
    loqo = {}
    for a in ALPHAS:
        Z = np.zeros((300, len(keep)))
        for i in range(len(keep)):
            o = np.r_[:i, i + 1:len(keep)]; M, c = ridge(Ep[o], TH_[o], a); Z[:, i] = Fp1 @ (Ep[i] @ M + c)
        loqo[float(a)] = (Z, [auc(Yk[:, i], Z[:, i]) for i in range(len(keep))])
    alpha = max(loqo, key=lambda a: np.nanmean(loqo[a][1])); Z = loqo[alpha][0]
    T = float(np.exp(minimize_scalar(lambda lt: wll(Yk.ravel(), sig((Z / np.exp(lt)).ravel()), Wk.ravel()), bounds=(-6, 6), method="bounded").x))
    M, c = ridge(Ep, TH_, alpha); Eh = Eall[[q["emb_row"] for q in HQ]]; Ph = Eh @ M + c
    qc = sig((np.c_[Ft, np.ones(len(Ft))] @ Ph.T) / T)
    # comparisons
    R = rows(); Et = np.load(f"{VSS}/emb_test.npy")[R["test"]]; lang = np.einsum("nfd,qd->nq", Et, Eh) / Et.shape[1]
    nsem = len(SEM); assert [q["id"] for q in HQ[:nsem]] == SEM
    astra = derive(astra19("test200", R["t200_test"], DIRECT)); qwen = derive(qwen19(R["test"], DIRECT))
    Pa = derive(astra19("bank", R["practice"], SOFT)); Kp = np.load(f"{ST}/phys_practice.npy"); pid = [q["id"] for q in json.load(open(f"{QC}/physical.json"))["questions"]]
    ref_lab = [(Pa[:, j] >= .5, np.abs(2 * Pa[:, j] - 1)) for j in range(nsem)] + [(Kp[:, pid.index(q["id"])], np.ones(300)) for q in HQ[nsem:]]
    ref = np.column_stack(Parallel(12)(delayed(lambda y, w: (lambda a, b: sig(Ft @ a + b))(*logit_fit(Fp, y, w, C)))(y, w) for y, w in ref_lab))
    Yp = truth23("rlpool", R["practice"]); tl = {}
    for n in (30, 100, 300):
        for s in (0, 1, 2):
            idx = np.sort(np.random.default_rng(s).choice(300, n, replace=False))
            tl[f"{n}_{s}"] = np.column_stack([sig(Ft @ a + b) if 0 < Yp[idx, j].sum() < n else np.full(len(Ft), np.nan)
                                              for j, (a, b) in enumerate(Parallel(12)(delayed(lambda j: logit_fit(Fp[idx], Yp[idx, j], np.ones(n), C) if 0 < Yp[idx, j].sum() < n else (np.zeros(4096), 0.))(j) for j in range(nsem)))])
    maxcos = (Eh @ Ep.T).max(1)
    np.savez(f"{ST}/qc3{DRY}.npz", qc=qc, lang=lang, astra=astra, qwen=qwen, ref=ref, M=M, c=c, T=T, alpha=alpha, C=C, theta=TH_, Ph=Ph, maxcos=maxcos, **{f"tl_{k}": v for k, v in tl.items()})
    out = {"C": C, "cv_logloss": {str(k): float(np.mean(v)) for k, v in L.items()}, "alpha": alpha, "loqo_mean_auc": {str(a): float(np.nanmean(v[1])) for a, v in loqo.items()},
           "T": T, "n_answerers": len(keep), "skipped": skipped, "practice_ids": [PQ[j]["id"] for j in keep], "loqo_auc": dict(zip([PQ[j]["id"] for j in keep], loqo[alpha][1])),
           "held_ids": [q["id"] for q in HQ], "max_cos_practice": maxcos.tolist(), "sec": round(time.time() - t0)}
    json.dump(out, open(f"{ST}/qc3{DRY}.json" if DRY else f"{QC}/qc3.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k not in ("loqo_auc", "practice_ids", "max_cos_practice", "held_ids")}))


if __name__ == "__main__":
    main()
