"""VS-3 (host A, CPU): stage-1 logistic probes on V, VC, F, F-hat, R, VC+F; graded classes and Q17; paired bootstrap contrasts;
figures. Labels enter here only. Outputs verbal_share/vs3.json, verbal_share/figs/*.png."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from vs_common import *


def probe(Xf, yf, Xt):
    s = StandardScaler().fit(Xf)
    m = LogisticRegressionCV(Cs=[0.01, 0.1, 1], cv=5, scoring="neg_log_loss", max_iter=3000).fit(s.transform(Xf), yf)
    return m.predict_proba(s.transform(Xt))[:, 1]


def auc_boot(y, P, B=1000, seed=0):  # same resamples for every input -> paired contrasts
    rng = np.random.default_rng(seed); n = len(y); out = np.full((B, len(P)), np.nan)
    for b in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max(): out[b] = [roc_auc_score(y[i], p[i]) for p in P]
    return out


def main():
    t0 = time.time(); S = sets(); ok_f, ok_t = np.load(f"{ST}/ok_fit.npy"), np.load(f"{ST}/ok_test.npy")
    fit = np.array(S["fit_rl"])[ok_f]; test = np.array(S["test"])[ok_t]
    Ef, Et = np.load(f"{ST}/emb_fit.npy")[ok_f], np.load(f"{ST}/emb_test.npy")[ok_t]
    Lf, Lt = np.load(f"{ST}/layers_fit.npy")[ok_f][:, LAYERS_F], np.load(f"{ST}/layers_test.npy")[ok_t][:, LAYERS_F]
    mu, sd = Lf.mean(0), Lf.std(0) + 1e-6; Ff, Ft = ((Lf - mu) / sd).reshape(len(Lf), -1), ((Lt - mu) / sd).reshape(len(Lt), -1)
    X = {"V": (Ef[:, :6].reshape(len(Ef), -1), Et[:, :6].reshape(len(Et), -1)), "VC": (Ef.reshape(len(Ef), -1), Et.reshape(len(Et), -1)),
         "F": (Ff, Ft), "Fhat": (np.load(f"{ST}/fhat_fit.npy"), np.load(f"{ST}/fhat_test.npy")), "R": (np.load(f"{ST}/resid_fit.npy"), np.load(f"{ST}/resid_test.npy"))}
    X["VC+F"] = (np.hstack([X["VC"][0], Ff]), np.hstack([X["VC"][1], Ft]))
    Yf, Yt = labels("rlpool", fit), labels("test", test); graded = json.load(open(f"{VS}/vs0.json"))["graded_classes"]
    T = {f"c{c}": (Yf[:, c - 1], Yt[:, c - 1]) for c in graded}; T["Q17"] = (q17("rlpool", fit), q17("test", test))
    INP = list(X); jobs = [(t, k) for t in T for k in INP]
    P = dict(zip(jobs, Parallel(n_jobs=12)(delayed(probe)(X[k][0], T[t][0], X[k][1]) for t, k in jobs)))
    print("probes", round(time.time() - t0), "s", flush=True)
    res, Bs = {}, {}
    for t in T:
        y = T[t][1]; Bs[t] = auc_boot(y, [P[(t, k)] for k in INP]); res[t] = {"auc": {k: float(roc_auc_score(y, P[(t, k)])) for k in INP}, "n_pos_test": int(y.sum())}
    ix = {k: i for i, k in enumerate(INP)}; ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    C = {"fm_gain": ("VC+F", "VC"), "fm_gain_Fcoords": ("F", "Fhat"), "phys_gain": ("VC", "V")}
    for t in T:
        b = Bs[t]; a = res[t]["auc"]
        res[t]["contrasts"] = {c: {"diff": a[p] - a[q], "ci": ci(b[:, ix[p]] - b[:, ix[q]])} for c, (p, q) in C.items()}
        res[t]["contrasts"]["resid_content"] = {"auc": a["R"], "ci": ci(b[:, ix["R"]])}
    cl = [f"c{c}" for c in graded]; mean = {}
    for c, (p, q) in C.items():
        d = np.nanmean([Bs[t][:, ix[p]] - Bs[t][:, ix[q]] for t in cl], 0)
        mean[c] = {"diff": float(np.mean([res[t]["auc"][p] - res[t]["auc"][q] for t in cl])), "ci": ci(d)}
    mean["auc"] = {k: float(np.mean([res[t]["auc"][k] for t in cl])) for k in INP}
    mean["resid_content"] = {"auc": mean["auc"]["R"], "ci": ci(np.nanmean([Bs[t][:, ix["R"]] for t in cl], 0))}
    h2 = {t: res[t]["contrasts"]["fm_gain"]["ci"][0] > 0 and res[t]["contrasts"]["fm_gain_Fcoords"]["ci"][0] > 0 for t in T}
    h2["mean_classes"] = mean["fm_gain"]["ci"][0] > 0 and mean["fm_gain_Fcoords"]["ci"][0] > 0
    h3 = {t: res[t]["contrasts"]["resid_content"]["ci"][0] > 0.5 for t in T if h2[t]}
    out = {"targets": res, "mean_over_graded": mean, "H2": h2, "H3_on_H2_targets": h3, "graded": graded, "n_fit": len(fit), "n_test": len(test),
           "sec": round(time.time() - t0)}
    json.dump(out, open(f"{VS}/vs3.json", "w"), indent=1); figures(); print(json.dumps({"mean": mean, "Q17": res["Q17"], "H2": h2, "H3": h3}, indent=1))


def figures():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    v2 = json.load(open(f"{VS}/vs2.json")); fam = v2["selected"]; ink, mut, bg = "#2b2a27", "#7a7974", "#fcfcfb"
    plt.rcParams.update({"axes.edgecolor": mut, "axes.labelcolor": ink, "xtick.color": mut, "ytick.color": mut, "figure.facecolor": bg, "axes.facecolor": bg})
    def clean(ax):
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    v, vc = v2["maps"]["V"][fam]["test_r2"], v2["maps"]["VC"][fam]["test_r2"]
    fig, ax = plt.subplots(figsize=(6.4, 2.3)); parts = [("Vision language (V)", max(v, 0), "#2a6fdb"), ("Added physical language (VC − V)", max(vc - v, 0), "#e8a33d"),
                                                      ("Residual (1 − VC)", 1 - max(vc, 0), "#c9c7bf")]; x = 0
    for name, w, c in parts:
        ax.barh(0, w, left=x, color=c, edgecolor=bg, linewidth=2, label=f"{name}: {w:.2f}"); x += w
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xlabel(f"Share of Prithvi F variance on Test ({fam} map)"); clean(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=7.5); fig.tight_layout(); fig.savefig(f"{VS}/figs/shares.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.6)); L = np.arange(1, 25)
    for (rep, f), (c, ls) in {("VC", fam): ("#2a6fdb", "-"), ("V", fam): ("#2a6fdb", "--"), ("VC", "linear"): ("#e8a33d", "-"), ("V", "linear"): ("#e8a33d", "--")}.items():
        if f in v2["layer_curve"][rep]: ax.plot(L, v2["layer_curve"][rep][f], ls, color=c, lw=2, marker="o", ms=4, label=f"{rep}, {f}")
    ax.set_xlabel("Prithvi block"); ax.set_ylabel("Held-out R² (Test)"); ax.set_xticks([1, 6, 12, 18, 24]); ax.legend(frameon=False, fontsize=8); clean(ax)
    fig.tight_layout(); fig.savefig(f"{VS}/figs/layer_curve.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.8, 3.2)); n = [int(k) for k in v2["learning_curve"]]
    for f, c in ((fam, "#2a6fdb"), ("linear", "#e8a33d")):
        if f in v2["learning_curve"]["250"]: ax.plot(n, [v2["learning_curve"][str(k)][f] for k in n], "-o", color=c, lw=2, ms=5, label=f)
    ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels(n); ax.minorticks_off(); ax.set_xlabel("Fit patches"); ax.set_ylabel("Held-out R² of VC → F"); ax.legend(frameon=False, fontsize=8); clean(ax)
    fig.tight_layout(); fig.savefig(f"{VS}/figs/learning_curve.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    figures() if sys.argv[1:] == ["figs"] else main()
