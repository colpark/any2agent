"""AH-3 fusion, heads, references, direct priors and measures (true labels: grading, reference heads and the trust check only).
Outputs agent_heads/ah3.json, agent_heads/figs/fused_vs_prior.png, ~/langres_state/ah/{fused_bank,scores_test}.npy. Usage: ah3.py [figs]"""
import json, os, sys, time, warnings
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from ah_common import *
from th_common import ba
warnings.filterwarnings("ignore")


def fusion(SRC):
    P, info = np.zeros((len(SRC["astra"]), len(CL))), {}
    for j, c in enumerate(CL):
        L = np.stack([SRC[s][:, j] for s in SRC], 1); P[:, j], par, it, fb = dawid_skene(L)
        info[c] = {"iterations": it, "fallback_majority": fb, **par, "fused_yes_rate": float((P[:, j] >= .5).mean()),
                   "mean_weight": float(np.abs(2 * P[:, j] - 1).mean()), "sources": list(SRC)}
    return P, info


def main():
    t0 = time.time(); S = sets(); ok = bank_ok(); bidx = np.array(S["bank_rl"])[ok]; I = inputs(); t200 = np.array(S["t200"])
    SRC = {"astra": astra("bank", bidx)[0], "qwen": qwen("bank", bidx)[0], "code": np.load(f"{ST}/code_bank.npy")[ok]}
    Pf, dsinfo = fusion(SRC); np.save(f"{ST}/fused_bank.npy", Pf)
    jobs, keys = [], []
    for j, c in enumerate(CL):
        y, w = Pf[:, j] >= .5, np.abs(2 * Pf[:, j] - 1)
        for rep in ("F", "VC", "D"):
            if min(y.sum(), (~y).sum()) >= 2: keys.append(("fused_" + rep, c)); jobs.append((I[rep][0], y, I[rep][1], w, True))
        for s, M in SRC.items():
            m = M[:, j] >= 0; ys = M[m, j] == 1
            if ys.sum() >= 10 and (~ys).sum() >= 10: keys.append(("src_" + s, c)); jobs.append((I["F"][0][m], ys, I["F"][1], None, True))
    Yb = truth("rlpool", bidx); refsub = {}
    for n in (30, 100, 500):
        for seed in (0, 1, 2):
            i = np.sort(np.random.default_rng(seed).choice(len(bidx), min(n, len(bidx)), replace=False)); refsub[(n, seed)] = i
            for j, c in enumerate(CL):
                if 0 < Yb[i, j].sum() < len(i) and (n > 30 or min(Yb[i, j].sum(), len(i) - Yb[i, j].sum()) >= 1):
                    keys.append((f"ref_{n}_{seed}", c)); jobs.append((I["F"][0][i], Yb[i, j], I["F"][1], None, n > 30))
    out = Parallel(n_jobs=12)(delayed(head)(Xb, y, Xt, w, cv) for Xb, y, Xt, w, cv in jobs); print("heads", len(jobs), round(time.time() - t0), "s", flush=True)
    SC = {}
    for (k, c), p in zip(keys, out): SC.setdefault(k, np.full((997, len(CL)), np.nan))[:, CL.index(c)] = p
    A2, Pa = astra("test200", S["t200_test"]); _, Pq = qwen("test", S["test"]); CT = np.load(f"{ST}/code_test.npy")
    SC["qwen_direct"] = Pq; SC["code_direct"] = np.where(CT == 1, 1., np.where(CT == 0, 0., .5))
    if os.path.exists(f"{ST}/free_scores.npy"): SC["free_agent"] = np.load(f"{ST}/free_scores.npy").astype(float)
    SC["astra_direct"] = np.full((997, len(CL)), np.nan); SC["astra_direct"][t200] = Pa
    np.save(f"{ST}/scores_test.npy", SC, allow_pickle=True)
    Yt = truth("test", S["test"]); Y2 = Yt[t200]; h1 = [j for j, c in enumerate(CL) if Y2[:, j].sum() >= 10]
    AUC = lambda Y, Sx, i: np.array([auc(Y[i, j], Sx[i, j]) if not np.isnan(Sx[i, j]).any() else np.nan for j in range(len(CL))])
    full = {k: AUC(Yt, v, np.arange(997)) for k, v in SC.items() if k != "astra_direct"}
    full200 = {k: AUC(Y2, SC[k][t200], np.arange(len(t200))) for k in ("fused_F", "astra_direct", "qwen_direct", "code_direct", "src_astra", "free_agent") if k in SC}

    def stats(A, A2_):  # contrasts from one (re)sample: dicts of per-class AUC on Test and on Test200
        d = {}
        bp = np.nanmax(np.stack([A2_["astra_direct"], A2_["qwen_direct"], A2_["code_direct"]]), 0); d["H1_test200_vs_best_direct"] = (A2_["fused_F"] - bp)[h1]
        for nm, ks in (("H1_test_vs_best_qwen_code_direct", ("qwen_direct", "code_direct")), ("H1_test_vs_qwen_direct", ("qwen_direct",)), ("H1_test_vs_code_direct", ("code_direct",))):
            d[nm] = (A["fused_F"] - np.nanmax(np.stack([A[k] for k in ks]), 0))[h1]
        d["H2_vs_best_single_source_head"] = A["fused_F"] - np.nanmax(np.stack([A[k] for k in ("src_astra", "src_qwen", "src_code") if k in A]), 0)
        d["substrate_F_minus_VC"] = A["fused_F"] - A["fused_VC"]; d["substrate_F_minus_D"] = A["fused_F"] - A["fused_D"]
        if "free_agent" in A: d["free_agent_minus_fused_F"] = A["free_agent"] - A["fused_F"]
        for k in ("src_astra", "free_agent"):  # report only, not design measures: H1-style comparison for the Astra-label head and the free agent
            if k in A2_: d[f"extra_H1_{k}_minus_best_direct_test200"] = (A2_[k] - bp)[h1]
        return d
    point = stats(full, full200); Bt, B2 = boot_idx(997), boot_idx(len(t200), seed=1)
    keep = [k for k in full if not k.startswith("ref_")]
    def one(i, i2): return stats({k: AUC(Yt, SC[k], i) for k in keep}, {k: AUC(Y2, SC[k][t200], i2) for k in full200})
    bs = Parallel(n_jobs=12)(delayed(one)(i, i2) for i, i2 in zip(Bt, B2)); print("bootstrap", round(time.time() - t0), "s", flush=True)
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    meas = {}
    for k in point:
        cls = [CL[j] for j in h1] if "H1" in k else CL; b = np.array([x[k] for x in bs])
        meas[k] = {"mean": float(np.nanmean(point[k])), "ci": ci(np.nanmean(b, 1)), "classes": cls,
                   "per_class": {c: {"diff": float(point[k][m]), "ci": ci(b[:, m])} for m, c in enumerate(cls)}}
    verdict = {"H1_pass": meas["H1_test200_vs_best_direct"]["ci"][0] > 0, "H2_pass": meas["H2_vs_best_single_source_head"]["ci"][0] > 0,
               "prithvi_adds_as_learner": meas["substrate_F_minus_VC"]["ci"][0] > 0 and meas["substrate_F_minus_D"]["ci"][0] > 0}
    ref = {n: np.nanmean([full[f"ref_{n}_{s}"] for s in range(3) if f"ref_{n}_{s}" in full], 0) for n in (30, 100, 500)}

    def eq(b, curve):
        ns = sorted(curve)
        if curve[ns[0]] >= b: return "at most 30"
        for n0, n1 in zip(ns, ns[1:]):
            if curve[n0] < b <= curve[n1]: return float(np.exp(np.log(n0) + (b - curve[n0]) / (curve[n1] - curve[n0]) * np.log(n1 / n0)))
        return "more than 500"
    label_value = {c: eq(full["fused_F"][j], {n: ref[n][j] for n in ref}) for j, c in enumerate(CL)}
    fc = ~np.isnan(full["fused_F"]); label_value["mean_classes_with_fused_head"] = eq(float(full["fused_F"][fc].mean()), {n: float(np.nanmean(ref[n][fc])) for n in ref})
    trust = {}
    for j, c in enumerate(CL):
        est, tru = [], []
        for s_i, (s, M) in enumerate(SRC.items()):
            m = M[:, j] >= 0; b_ = ba(Yb[m, j], M[m, j] == 1) if m.sum() >= 2 else None
            if b_ is not None and m.any(): est.append((dsinfo[c]["sens"][s_i] + dsinfo[c]["spec"][s_i]) / 2); tru.append(b_)
        trust[c] = {"sources": [s for s, M in SRC.items() if (M[:, j] >= 0).sum() >= 2 and ba(Yb[M[:, j] >= 0, j], M[M[:, j] >= 0, j] == 1) is not None],
                    "ds_est_accuracy": est, "true_bank_ba": tru, "spearman": float(spearmanr(est, tru).statistic) if len(est) >= 2 else None}
    pe, pt = sum((t["ds_est_accuracy"] for t in trust.values()), []), sum((t["true_bank_ba"] for t in trust.values()), [])
    trust["pooled_class_source_pairs"] = {"n": len(pe), "spearman": float(spearmanr(pe, pt).statistic)}
    res = {"n_bank": int(ok.sum()), "n_test": 997, "n_test200": len(t200), "h1_classes": [CL[j] for j in h1], "dawid_skene": dsinfo,
           "auc_test": {k: dict(zip(CL, map(float, v))) for k, v in full.items()}, "auc_test200": {k: dict(zip(CL, map(float, v))) for k, v in full200.items()},
           "ref_curve_mean_auc": {n: dict(zip(CL, map(float, v))) for n, v in ref.items()}, "measures": meas, "verdict": verdict, "label_value": label_value,
           "trust": trust, "n_heads": len(jobs), "sec": round(time.time() - t0)}
    json.dump(res, open(f"{AH}/ah3.json", "w"), indent=1); figs()
    print(json.dumps({"verdict": verdict, **{k: (round(v["mean"], 3), [round(x, 3) for x in v["ci"]]) for k, v in meas.items()}, "label_value_mean": label_value["mean_classes_with_fused_head"]}, indent=1))


def figs():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r = json.load(open(f"{AH}/ah3.json")); a, a2 = r["auc_test"], r["auc_test200"]; ink, mut, bg = "#2b2a27", "#7a7974", "#fcfcfb"
    plt.rcParams.update({"axes.edgecolor": mut, "axes.labelcolor": ink, "xtick.color": mut, "ytick.color": mut, "figure.facecolor": bg, "axes.facecolor": bg})
    fig, ax = plt.subplots(figsize=(5.2, 5.0)); lo = 0.45
    for c in CL:
        c_ = str(c); best = max(a["qwen_direct"][c_], a["code_direct"][c_]); f = a["fused_F"][c_]
        if c in r["h1_classes"]:
            best200 = max(v for v in (a2["astra_direct"][c_], a2["qwen_direct"][c_], a2["code_direct"][c_]) if v == v)
            ax.plot(best200, a2["fused_F"][c_], "o", color="#2a6fdb", ms=7, mec=bg, mew=1.5)
            ax.annotate(c_, (best200, a2["fused_F"][c_]), textcoords="offset points", xytext=(5, -3), fontsize=7, color="#2a6fdb")
        ax.plot(best, f, "s", color="#e8a33d", ms=6, mec=bg, mew=1.5); ax.annotate(c_, (best, f), textcoords="offset points", xytext=(5, -3), fontsize=7, color=mut)
    ax.plot([lo, 1], [lo, 1], "-", color=mut, lw=1); ax.set_xlim(lo, 1); ax.set_ylim(lo, 1); ax.set_aspect("equal")
    ax.plot([], [], "o", color="#2a6fdb", label="Test200: best of Astra, Qwen, code direct"); ax.plot([], [], "s", color="#e8a33d", label="Test: best of Qwen, code direct")
    ax.set_xlabel("Best direct prior AUC"); ax.set_ylabel("Fused F head AUC"); ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{AH}/figs/fused_vs_prior.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    figs() if sys.argv[1:] == ["figs"] else main()
