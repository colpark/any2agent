"""QV-3 measures (grading with true labels and code keys) -> qcond_v2/qv3.json, figs/learning_curve.png, figs/cascade_per_answer.png."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from qv_common import *

CONDS = ["A1B1", "A1B2", "A2B1", "A2B2"]; SIZES = (35, 70, 139)


def main():
    V = v1(); Z = V["Z"]; R = rows(); hids = V["hids"]; ns = len(SEM); t2 = R["t200"]
    phys = [q["id"] for q in json.load(open(f"{QC}/physical.json"))["questions"]]; K = np.load(f"{ST}/phys_test.npy")
    Y = np.column_stack([truth23("test", R["test"])] + [K[:, phys.index(q)] for q in hids[ns:]])
    fit = {n: np.load(f"{SQV}/fit_{n}.npz") for n in SIZES}; P = {c: fit[139][c] for c in CONDS}
    S = {**P, "v1": Z["qc"], "lang": Z["lang"], "ref": Z["ref"], "A2B2_70": fit[70]["A2B2"]}
    per = {k: auc_cols(Y, v) for k, v in S.items()}; ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    bs = {k: [] for k in S}
    for b in boot_idx(len(Y)):
        for k, v in S.items(): bs[k].append(np.nanmean(auc_cols(Y[b, :ns], v[b, :ns])))
    bs = {k: np.array(v) for k, v in bs.items()}; m = {k: float(np.nanmean(per[k][:ns])) for k in S}
    out = {"mean_sem": {k: {"auc": m[k], "ci": ci(bs[k])} for k in S}}
    out["R1"] = {c: {"diff": m[c] - m["lang"], "ci": ci(bs[c] - bs["lang"]), "pass": ci(bs[c] - bs["lang"])[0] > 0} for c in CONDS}
    out["R2"] = {c: {"ratio": m[c] / m["ref"], "ci_minus_0.9ref": ci(bs[c] - .9 * bs["ref"]), "pass": m[c] / m["ref"] >= .9} for c in CONDS}
    eff = {"polarity (A1B2 - A1B1)": ("A1B2", "A1B1"), "MLP (A2B1 - A1B1)": ("A2B1", "A1B1"), "both (A2B2 - A1B1)": ("A2B2", "A1B1"), "MLP given polarity (A2B2 - A1B2)": ("A2B2", "A1B2")}
    out["effects"] = {k: {"diff": m[a] - m[b], "ci": ci(bs[a] - bs[b])} for k, (a, b) in eff.items()}
    out["more_data"] = {"A2B2_139_minus_70": m["A2B2"] - m["A2B2_70"], "ci": ci(bs["A2B2"] - bs["A2B2_70"]), "pays_off": m["A2B2"] - m["A2B2_70"] >= .01}
    out["polarity"] = {c: {"n_below_0.5": int((per[c] < .5).sum()), "below": [h for h, a in zip(hids, per[c]) if a < .5], "phys_mean": float(np.mean(per[c][ns:]))} for c in CONDS + ["v1"]}
    out["learning_curve"] = {c: {n: float(np.nanmean(auc_cols(Y[:, :ns], fit[n][c][:, :ns]))) for n in SIZES} for c in CONDS}
    out["learning_curve_phys"] = {c: {n: float(np.nanmean(auc_cols(Y[:, ns:], fit[n][c][:, ns:]))) for n in SIZES} for c in CONDS}
    ay, yt = Z["astra"][:, :15] > .5, Y[t2, :15]; abase = float(np.nanmean([v1meas.ba(yt[:, j], ay[:, j]) for j in range(15)])); cas = {}
    for c in CONDS:
        p = P[c][t2, :15]; cas[c] = []
        for tau in np.round(np.arange(0, .501, .05), 2):
            d = np.abs(p - .5) < tau; dec = np.where(d, ay, p >= .5)
            cas[c].append({"tau": float(tau), "mean_ba": float(np.nanmean([v1meas.ba(yt[:, j], dec[:, j]) for j in range(15)])), "share_answers": float(d.mean())})
    out["cost"] = {"astra_direct_mean_ba": abase, "cascade": cas, "pass": {c: any(x["share_answers"] <= .3 and x["mean_ba"] >= abase - .01 for x in cas[c]) for c in CONDS}}
    out["calibration"] = {c: {"ece_sem": float(np.mean([v1meas.ece(Y[:, j], S[c][:, j]) for j in range(ns)])), "brier_sem": float(np.mean((S[c][:, :ns] - Y[:, :ns]) ** 2)),
                              "ece_phys": float(np.mean([v1meas.ece(Y[:, j], S[c][:, j]) for j in range(ns, len(hids))])), "brier_phys": float(np.mean((S[c][:, ns:] - Y[:, ns:]) ** 2))} for c in CONDS + ["v1", "ref"]}
    keep = (Y[:, :ns].sum(1) > 0) & (Y[:, :ns].sum(1) < ns); pp = {}
    for c in CONDS + ["lang"]:
        a = auc_cols(Y[keep, :ns].T, S[c][keep, :ns].T); pp[c] = a
    Bk = boot_idx(int(keep.sum())); out["describe"] = {c: {"mean": float(np.mean(pp[c])), "ci": ci([pp[c][b].mean() for b in Bk]), "minus_lang_ci": ci([(pp[c][b] - pp["lang"][b]).mean() for b in Bk])} for c in CONDS}
    out["describe"]["lang"] = {"mean": float(np.mean(pp["lang"]))}; out["describe_skipped"] = int((~keep).sum())
    out["per_question"] = {h: {k: (None if np.isnan(per[k][j]) else round(float(per[k][j]), 4)) for k in per} for j, h in enumerate(hids)}
    for k, v in (("qwen", Z["qwen"]), ("astra_t200", None)):
        a = auc_cols(Y[:, :ns], v) if v is not None else auc_cols(Y[t2, :ns], Z["astra"])
        for j, h in enumerate(hids[:ns]): out["per_question"][h][k] = round(float(a[j]), 4)
    json.dump(out, open(f"{QV}/qv3.json", "w"), indent=1); figs(out)
    print(json.dumps({k: out[k] for k in ("mean_sem", "R1", "R2", "effects", "more_data", "learning_curve", "describe")}, indent=1))
    print(json.dumps({c: {k: v for k, v in out["polarity"][c].items()} for c in out["polarity"]}))
    print("astra", abase, json.dumps(out["cost"]["pass"]), {c: [(x["tau"], round(x["mean_ba"], 3), round(x["share_answers"], 2)) for x in cas[c]] for c in ("A2B2", "A1B1")})


def figs(out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(f"{QV}/figs", exist_ok=True); col = {"A1B1": "#6e7781", "A1B2": "#d97706", "A2B1": "#7c3aed", "A2B2": "#2563eb"}
    nm = {"A1B1": "A1B1 base, as written (v1)", "A1B2": "A1B2 base, polarity", "A2B1": "A2B1 +ZeroMLP, as written", "A2B2": "A2B2 +ZeroMLP, polarity (primary)"}
    f, ax = plt.subplots(figsize=(6.4, 4.2))
    for c in CONDS:
        lc = out["learning_curve"][c]; ax.plot(SIZES, [lc[n] for n in SIZES], "-o", color=col[c], lw=2.5 if c == "A2B2" else 1.8, ms=7, mec="white", label=nm[c])
    ax.axhline(out["mean_sem"]["lang"]["auc"], color="#1f2328", lw=1, ls="--", label="language route"); ax.set_xticks(SIZES)
    ax.set_xlabel("practice questions"); ax.set_ylabel("mean AUC, 23 held-out semantic questions"); ax.legend(frameon=False, fontsize=8)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    f.tight_layout(); f.savefig(f"{QV}/figs/learning_curve.png", dpi=150); plt.close(f)
    f, ax = plt.subplots(figsize=(6.4, 4.2))
    for c in CONDS:
        cs = out["cost"]["cascade"][c]; ax.plot([x["share_answers"] for x in cs], [x["mean_ba"] for x in cs], "-o", color=col[c], lw=2.5 if c == "A2B2" else 1.5, ms=5, mec="white", label=nm[c])
    a = out["cost"]["astra_direct_mean_ba"]; ax.axhline(a, color="#1f2328", lw=1, label=f"Astra direct {a:.3f}"); ax.axhline(a - .01, color="#6e7781", lw=1, ls="--", label="Astra − 0.01")
    ax.axvline(.3, color="#6e7781", lw=1, ls=":"); ax.set_xlabel("share of Test200 answers sent to Astra"); ax.set_ylabel("mean balanced accuracy (15 classes)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    f.tight_layout(); f.savefig(f"{QV}/figs/cascade_per_answer.png", dpi=150); plt.close(f)


if __name__ == "__main__":
    main()
