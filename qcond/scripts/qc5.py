"""QC-5 (CPU, no Astra): the by-product projection g(f) of the v1 formula. Reads v1 artifacts only -> qcond/qc5.json.
v1 stores the formula as params(e) = e @ M + c with M (2560 x 4097), so M_w = M[:, :4096], m_b = M[:, 4096] and g(f) = M_w f + m_b."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.model_selection import KFold
from qc_common import *
DRY = os.environ.get("QC_DRY", "")  # dry run on the dry qc3 outputs; writes under ~/langres_state/qc/dry


def ranks(S):  # rank of the own candidate (diagonal), 1 = best
    return 1 + (S > np.diag(S)[:, None]).sum(1)


def kridge(X, Y, a):
    mx, my = X.mean(0), Y.mean(0); Xc = X - mx; B = Xc.T @ np.linalg.solve(Xc @ Xc.T + a * np.eye(len(X)), Y - my); return lambda Z: (Z - mx) @ B + my


def nrm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def main():
    Z = np.load(f"{ST}/qc3{DRY}.npz"); J = json.load(open(f"{ST}/qc3{DRY}.json" if DRY else f"{QC}/qc3.json")); M, c = Z["M"], Z["c"]; Mw, mb, cw, cb = M[:, :4096], M[:, 4096], c[:4096], c[4096]
    Fp, Ft = (x.astype(np.float64) for x in feats()); R = rows(); E = np.load(f"{ST}/emb_all.npy").astype(np.float64); Q = questions()
    H = [q for q in Q if q["split"] == "held_out"]; Eh = E[[q["emb_row"] for q in H]]; g = lambda F: F @ Mw.T + mb
    # 1. reproduce the v1 logits on 5 Test patches x 3 held-out questions
    v1 = (np.c_[Ft, np.ones(len(Ft))] @ Z["Ph"].T)[:5, :3]; new = (g(Ft[:5]) @ Eh[:3].T) + (Ft[:5] @ cw + cb)[:, None]
    chk = {"max_abs_diff": float(np.abs(v1 - new).max()), "v1_vs_stored_qc_maxdiff": float(np.abs(sig(v1 / float(Z["T"])) - Z["qc"][:5, :3]).max())}
    chk["pass"] = chk["max_abs_diff"] <= 1e-6; print(chk); assert chk["pass"]
    # A. caption retrieval
    Et = np.load(f"{VSS}/emb_test.npy")[R["test"]].astype(np.float64); C = nrm(Et[:, 5]); Gt = g(Ft)
    Eb = np.load(f"{ah.ST}/emb_bank.npy")[ah.bank_ok()][practice()][:, 5].astype(np.float64); Yb = nrm(Eb); cvs = {}
    for a in np.logspace(-3, 3, 13):
        s = [np.mean(np.sum(nrm(kridge(Fp[tr], Yb[tr], a)(Fp[te])) * Yb[te], 1)) for tr, te in KFold(5, shuffle=True, random_state=0).split(Fp)]; cvs[float(a)] = float(np.mean(s))
    alpha = max(cvs, key=cvs.get); P = nrm(kridge(Fp, Yb, alpha)(Ft)); perm = np.random.default_rng(0).permutation(len(Ft))
    rk = {"qc": ranks(Gt @ C.T), "caption_ridge": ranks(P @ C.T), "control_shuffled": ranks(g(Ft[perm]) @ C.T)}
    stat = lambda r: {"R@1": float((r <= 1).mean()), "R@10": float((r <= 10).mean()), "median_rank": float(np.median(r))}
    B = boot_idx(len(Ft)); ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    A = {k: {**stat(r), "ci": {m: ci([stat(r[b])[m] for b in B]) for m in ("R@1", "R@10", "median_rank")}} for k, r in rk.items()}
    A["qc_minus_caption_ridge_ci"] = {m: ci([stat(rk["qc"][b])[m] - stat(rk["caption_ridge"][b])[m] for b in B]) for m in ("R@1", "R@10", "median_rank")}
    A["qc_minus_control_ci"] = {m: ci([stat(rk["qc"][b])[m] - stat(rk["control_shuffled"][b])[m] for b in B]) for m in ("R@1", "R@10", "median_rank")}
    A["caption_ridge_alpha"], A["caption_ridge_cv_cos"] = alpha, cvs
    # B. ranking the 23 held-out semantic questions per patch
    ns = len(SEM); Y = truth23("test", R["test"]); keep = (Y.sum(1) > 0) & (Y.sum(1) < ns)
    sq, sl = Gt @ Eh[:ns].T, Z["lang"][:, :ns]; aq = np.array([auc(Y[i], sq[i]) for i in np.flatnonzero(keep)]); al = np.array([auc(Y[i], sl[i]) for i in np.flatnonzero(keep)])
    Bk = boot_idx(len(aq)); Bres = {"n_patches": int(keep.sum()), "skipped_all_same": int((~keep).sum()), "qc": float(aq.mean()), "lang": float(al.mean()),
                                    "diff": float((aq - al).mean()), "diff_ci": ci([(aq[b] - al[b]).mean() for b in Bk]), "qc_ci": ci([aq[b].mean() for b in Bk]), "lang_ci": ci([al[b].mean() for b in Bk])}
    sv = np.linalg.svd(Mw, compute_uv=False); rank = {"effective_rank_1pct": int((sv > .01 * sv[0]).sum()), "n_practice_questions": J["n_answerers"], "top10_sv_share": float((sv[:10] ** 2).sum() / (sv ** 2).sum())}
    out = {"check": chk, "A": A, "B": Bres, "rank_Mw": rank}; json.dump(out, open(f"{ST}/dry/qc5.json" if DRY else f"{QC}/qc5.json", "w"), indent=1)
    print(json.dumps({"A": {k: {m: A[k][m] for m in ("R@1", "R@10", "median_rank")} for k in rk}, "B": Bres, "rank": rank}, indent=1))


if __name__ == "__main__":
    main()
