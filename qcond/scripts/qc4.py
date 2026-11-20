"""QC-4 measures (grading with true labels and code keys) -> qcond/qc4.json, figs/generalization.png, figs/cascade.png."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import spearmanr
from qc_common import *
DRY = os.environ.get("QC_DRY", "")  # dry run: reads the dry qc3 outputs, writes under ~/langres_state/qc
if DRY: QC_OUT = f"{ST}/dry"
else: QC_OUT = QC


def ba(y, d):
    y = np.asarray(y, bool); d = np.asarray(d, bool); return np.nan if y.all() or (~y).all() else .5 * (d[y].mean() + (~d[~y]).mean())


def ece(y, p, k=15):
    b = np.minimum((p * k).astype(int), k - 1); return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(k) if (b == i).any()))


def main():
    Z = np.load(f"{ST}/qc3{DRY}.npz"); J = json.load(open(f"{ST}/qc3{DRY}.json" if DRY else f"{QC}/qc3.json")); R = rows(); ids = J["held_ids"]; ns = len(SEM)
    phys = json.load(open(f"{QC}/physical.json"))["questions"]; pid = [q["id"] for q in phys]; K = np.load(f"{ST}/phys_test.npy")
    Y = np.column_stack([truth23("test", R["test"])] + [K[:, pid.index(q)] for q in ids[ns:]])
    t2 = R["t200"]; tl = {n: np.nanmean([Z[f"tl_{n}_{s}"] for s in (0, 1, 2)], 0) for n in (30, 100, 300)}
    A = lambda S, y=Y: np.array([auc(y[:, j], S[:, j]) for j in range(S.shape[1])])
    per = {"qc": A(Z["qc"]), "lang": A(Z["lang"]), "ref": A(Z["ref"]), "qwen": A(Z["qwen"], Y[:, :ns]), "astra_t200": A(Z["astra"], Y[t2, :ns]),
           "qc_t200": A(Z["qc"][t2, :ns], Y[t2, :ns]), **{f"true{n}": np.array([np.nanmean([auc(Y[:, j], Z[f"tl_{n}_{s}"][:, j]) for s in (0, 1, 2)]) for j in range(ns)]) for n in (30, 100, 300)}}
    r1, r2 = [], []
    for b in boot_idx(len(Y)):
        q, l, f = (A(Z[k][b][:, :ns], Y[b][:, :ns]) for k in ("qc", "lang", "ref")); r1.append(np.nanmean(q - l)); r2.append(np.nanmean(q) - .9 * np.nanmean(f))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    s = slice(0, ns); out = {"R1": {"mean_diff": float(np.mean(per["qc"][s] - per["lang"][s])), "ci": ci(r1)}}
    out["R1"]["pass"] = out["R1"]["ci"][0] > 0
    out["R2"] = {"qc": float(per["qc"][s].mean()), "ref": float(per["ref"][s].mean()), "ratio": float(per["qc"][s].mean() / per["ref"][s].mean()),
                 "qc_minus_0.9ref_ci": ci(r2)}; out["R2"]["pass"] = out["R2"]["ratio"] >= .9
    # cascade on Test200, 15 classes
    p, ay, yt = Z["qc"][t2, :15], Z["astra"][:, :15] > .5, Y[t2, :15]; abase = float(np.nanmean([ba(yt[:, j], ay[:, j]) for j in range(15)])); cas = []
    for tau in np.round(np.arange(0, .501, .05), 2):
        conf = np.abs(p - .5) >= tau; d = np.where(conf, p >= .5, ay)
        cas.append({"tau": float(tau), "mean_ba": float(np.nanmean([ba(yt[:, j], d[:, j]) for j in range(15)])), "astra_share_patches": float((~conf).any(1).mean()),
                    "astra_share_answers": float((~conf).mean())})
    out["cost"] = {"astra_direct_mean_ba": abase, "cascade": cas, "pass": any(c["astra_share_patches"] <= .3 and c["mean_ba"] >= abase - .01 for c in cas)}
    out["calibration"] = {k: {"ece_sem": float(np.mean([ece(Y[:, j], Z[k][:, j]) for j in range(ns)])), "brier_sem": float(np.mean((Z[k][:, :ns] - Y[:, :ns]) ** 2)),
                              "ece_phys": float(np.mean([ece(Y[:, j], Z[k][:, j]) for j in range(ns, len(ids))])), "brier_phys": float(np.mean((Z[k][:, ns:] - Y[:, ns:]) ** 2))} for k in ("qc", "ref")}
    mc = Z["maxcos"]; out["generalization"] = {"spearman_all": float(spearmanr(mc, per["qc"]).statistic), "spearman_sem": float(spearmanr(mc[:ns], per["qc"][:ns]).statistic)}
    eq = []
    for j in range(ns):
        c = [per[f"true{n}"][j] for n in (30, 100, 300)]; v = per["qc"][j]
        eq.append("<=30" if v <= c[0] else ">300" if v > c[2] else "30-100" if v <= c[1] else "100-300")
    out["labels_equivalent"] = dict(zip(ids[:ns], eq)); out["loqo_practice_auc_mean"] = float(np.mean(list(J["loqo_auc"].values())))
    out["per_question"] = {q: {k: (None if j >= len(v) or np.isnan(v[j]) else round(float(v[j]), 4)) for k, v in per.items()} | {"max_cos": round(float(mc[j]), 3)} for j, q in enumerate(ids)}
    out["means_sem"] = {k: float(np.nanmean(v[:ns])) for k, v in per.items()}; out["means_phys"] = {k: float(np.nanmean(per[k][ns:])) for k in ("qc", "lang", "ref")}
    json.dump(out, open(f"{QC_OUT}/qc4.json", "w"), indent=1); figs(out, mc, per, ns, ids)
    print(json.dumps({k: out[k] for k in ("R1", "R2", "means_sem", "means_phys", "calibration", "generalization", "loqo_practice_auc_mean")}, indent=1))
    print(json.dumps(out["cost"]["astra_direct_mean_ba"]), [(c["tau"], round(c["mean_ba"], 3), round(c["astra_share_patches"], 2), round(c["astra_share_answers"], 2)) for c in cas], out["cost"]["pass"])


def figs(out, mc, per, ns, ids):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(f"{QC_OUT}/figs", exist_ok=True); ink, mut = "#1f2328", "#6e7781"
    f, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(mc[:ns], per["qc"][:ns], s=40, color="#2563eb", label="semantic (23)", edgecolor="white", lw=1.5, zorder=3)
    ax.scatter(mc[ns:], per["qc"][ns:], s=40, color="#d97706", marker="s", label="physical (10)", edgecolor="white", lw=1.5, zorder=3)
    ax.axhline(.5, color=mut, lw=1, ls="--"); ax.set_xlabel("max cosine to any practice question"); ax.set_ylabel("QC AUC on Test")
    ax.set_title(f"Held-out questions: Spearman {out['generalization']['spearman_all']:.2f}", color=ink, fontsize=11); ax.legend(frameon=False)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    f.tight_layout(); f.savefig(f"{QC_OUT}/figs/generalization.png", dpi=150); plt.close(f)
    c = out["cost"]["cascade"]; f, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot([x["astra_share_patches"] for x in c], [x["mean_ba"] for x in c], "-o", color="#2563eb", lw=2, ms=8, mec="white", label="QC + Astra cascade")
    for x in c[::2]: ax.annotate(f"τ={x['tau']:.1f}", (x["astra_share_patches"], x["mean_ba"]), textcoords="offset points", xytext=(6, -12), fontsize=8, color=mut)
    a = out["cost"]["astra_direct_mean_ba"]; ax.axhline(a, color=ink, lw=1, label=f"Astra direct {a:.3f}"); ax.axhline(a - .01, color=mut, lw=1, ls="--", label="Astra − 0.01")
    ax.axvline(.3, color=mut, lw=1, ls=":"); ax.set_xlabel("share of Test200 patches sent to Astra"); ax.set_ylabel("mean balanced accuracy (15 classes)")
    ax.legend(frameon=False, loc="lower right")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    f.tight_layout(); f.savefig(f"{QC_OUT}/figs/cascade.png", dpi=150); plt.close(f)


if __name__ == "__main__":
    main()
