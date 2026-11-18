"""TH-1 grading (E1) and TH-2 heads (E2). Usage: th2_heads.py e1 | e2
True labels appear only in grading and in the label-efficiency reference. Code-teacher BA is graded on the patches it does not abstain on."""
import json, os, sys, warnings
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from th_common import *
warnings.filterwarnings("ignore")
s = sets(); YT = true_labels(s["test_ids"]); T200 = np.array(s["test200"])
GR = [c for c in CID if YT[:, c - 1].sum() >= 20]; GR_A = [c for c in GR if YT[T200, c - 1].sum() >= 10]
enc = lambda v: {"yes": 1, "no": 0}.get(v, -1)


def astra(tag, idx):
    A = json.load(open(f"{ST}/astra_{tag}.json"))
    return np.array([[enc(A.get(str(i), {}).get(str(c))) for c in CID] for i in idx], np.int8)


def qwen(tag, idx):
    R = {r["split_index"]: r["answers"] for r in map(json.loads, open(f"{ST}/qwen_{tag}.jsonl"))}
    return np.array([[enc((R.get(i) or {}).get(str(c))) for c in CID] for i in idx], np.int8)


AB, QB, CB = astra("bank", s["bank_rl"]), qwen("bank", s["bank_rl"]), np.load(f"{ST}/code_bank.npy")
A200, QT, CT = astra("test200", T200), qwen("test", range(1000)), np.load(f"{ST}/code_test.npy")
PRACTICE = {"astra": AB, "qwen": QB, "code": CB, "astra+code": np.where(CB < 0, AB, np.where(CB == AB, AB, -1))}


def grade(y, M, c):  # teacher M (-1 = abstain/missing) on class c
    m = M[:, c - 1] >= 0
    if m.sum() < 2 or ba(y[m], M[m, c - 1] == 1) is None: return {"coverage": float(m.mean()), "ba": None, "ci": None}
    return {"coverage": float(m.mean()), "ba": ba(y[m], M[m, c - 1] == 1), "ci": boot(y[m], M[m, c - 1] == 1)}


def e1():
    g = {}
    for c in GR:
        y, y2 = YT[:, c - 1], YT[T200, c - 1]
        g[c] = {"n_pos": int(y.sum()), "n_pos_test200": int(y2.sum()), "astra_test200": grade(y2, A200, c) if c in GR_A else None,
                "qwen_test": grade(y, QT, c), "code_test": grade(y, CT, c), "qwen_test200": grade(y2, QT[T200], c), "code_test200": grade(y2, CT[T200], c)}
    rel = {}
    for c in CID:
        k = c - 1; pairs = [float((AB[:, k] == QB[:, k]).mean())]; m = CB[:, k] >= 0
        if m.any(): pairs.append(float((AB[m, k] == CB[m, k]).mean()))
        rel[c] = {"astra_qwen": pairs[0], "astra_code": pairs[1] if len(pairs) > 1 else None, "reliability": float(np.mean(pairs))}
    cuts = np.quantile([rel[c]["reliability"] for c in CID], [1 / 3, 2 / 3])
    for c in CID: rel[c]["tier"] = ["low", "medium", "high"][int(np.digitize(rel[c]["reliability"], cuts))]
    lb = lambda d: d["ci"][0] if d and d.get("ci") else -1
    passed = {"astra": [c for c in GR_A if lb(g[c]["astra_test200"]) > 0.5], "qwen": [c for c in GR if lb(g[c]["qwen_test"]) > 0.5],
              "code": [c for c in GR if lb(g[c]["code_test"]) > 0.5]}
    rho = spearmanr([rel[c]["reliability"] for c in GR_A], [g[c]["astra_test200"]["ba"] for c in GR_A])
    out = {"graded": GR, "graded_astra": GR_A, "grades": g, "reliability": rel, "tier_cuts": cuts.tolist(), "passed": passed,
           "classes_passed_E1": sorted(set().union(*passed.values())), "spearman_rel_vs_astra_ba": [float(rho.statistic), float(rho.pvalue)],
           "gate_E1": max(len(v) for v in passed.values()) >= 8,
           "missing": {"astra_bank": int((AB < 0).all(1).sum()), "astra_test200": int((A200 < 0).all(1).sum()), "qwen_bank": int((QB < 0).all(1).sum()), "qwen_test": int((QT < 0).all(1).sum())}}
    json.dump(out, open(f"{TH}/e1.json", "w"), indent=1)
    f = lambda d: "—" if not d or d["ba"] is None else "%.2f[%.2f,%.2f]" % (d["ba"], *d["ci"])
    for c in GR: print(c, CLASSES[c - 1][:22].ljust(22), "A200", f(g[c]["astra_test200"]), "Q", f(g[c]["qwen_test"]), "code cov %.2f" % g[c]["code_test"]["coverage"], f(g[c]["code_test"]), "rel %.2f %s" % (rel[c]["reliability"], rel[c]["tier"]))
    print(json.dumps({k: out[k] for k in ("passed", "spearman_rel_vs_astra_ba", "gate_E1", "missing")}))


def fit(X, y, cv=True):
    if y.all() or (~y).all(): return None
    m = LogisticRegressionCV(Cs=[0.01, 0.1, 1], cv=5, scoring="neg_log_loss", max_iter=3000) if cv else LogisticRegression(C=0.1, max_iter=3000)
    return m.fit(X, y)


def e2():
    E1 = json.load(open(f"{TH}/e1.json")); Xb, Xt = features(); P, heads = {}, {}
    for src, L in PRACTICE.items():
        for c in CID:
            m = L[:, c - 1] >= 0; y = L[m, c - 1] == 1
            if y.sum() < 10 or (~y).sum() < 10: continue
            clf = fit(Xb[m], y); P[(src, c)] = clf.predict_proba(Xt)[:, 1]
            heads[f"{src}:{c}"] = {"n_train": int(m.sum()), "n_yes": int(y.sum()), "C": float(clf.C_[0])}
    np.save(f"{ST}/head_probs.npy", {f"{a}:{b}": v for (a, b), v in P.items()}, allow_pickle=True)
    Yb = true_labels(s["bank_ids"]); res = {}
    for c in GR:
        y = YT[:, c - 1]; r = {}
        for src in PRACTICE:
            if (src, c) not in P: continue
            p = P[(src, c)]; hb = p > 0.5; r[src] = {"ba": ba(y, hb), "auc": float(roc_auc_score(y, p))}
            if src.startswith("astra") and c in GR_A:
                y2, h2, t2 = YT[T200, c - 1], hb[T200], A200[:, c - 1] == 1
                r[src].update({"teacher": "astra@test200", "diff": ba(y2, h2) - ba(y2, t2), "diff_ci": boot(y2, h2, pair=t2)})
            elif src in ("qwen", "code"):
                M = QT if src == "qwen" else CT; m = M[:, c - 1] >= 0; t = M[:, c - 1] == 1
                if ba(y[m], t[m]) is not None: r[src].update({"teacher": f"{src}@test", "diff": ba(y[m], hb[m]) - ba(y[m], t[m]), "diff_ci": boot(y[m], hb[m], pair=t[m])})
        if ("astra+code", c) in P and ("qwen", c) in P:
            a_, q_ = P[("astra+code", c)] > 0.5, P[("qwen", c)] > 0.5; r["strength_ac_minus_qwen"] = {"diff": ba(y, a_) - ba(y, q_), "ci": boot(y, a_, pair=q_)}
        ref = {}
        for n in (10, 30, 100, 300, 500):
            for seed in (0, 1, 2):
                i = np.random.default_rng(seed).choice(500, n, replace=False); yy = Yb[i, c - 1]; clf = fit(Xb[i], yy, cv=n > 30)
                ref.setdefault(n, []).append(ba(y, clf.predict(Xt) if clf is not None else np.full(1000, bool(yy[0]))))
        curve = {n: float(np.mean(v)) for n, v in ref.items()}; eq = ">500"
        if "astra+code" in r:
            b = r["astra+code"]["ba"]; ns = sorted(curve)
            if curve[ns[0]] >= b: eq = ns[0]
            else:
                for n0, n1 in zip(ns, ns[1:]):
                    if curve[n0] < b <= curve[n1]: eq = float(np.exp(np.log(n0) + (b - curve[n0]) / (curve[n1] - curve[n0]) * np.log(n1 / n0))); break
        res[c] = {"heads": r, "ref_curve": curve, "ref_sd": {n: float(np.std(v)) for n, v in ref.items()}, "labels_equivalent_ac": eq}
    passed = [c for c in E1["classes_passed_E1"] if c in GR_A and "diff" in res[c]["heads"].get("astra+code", {})]
    rng = np.random.default_rng(0); y2 = YT[T200]
    def mean_diff(i):
        return np.mean([ba(y2[i, c - 1], (P[("astra+code", c)] > 0.5)[T200][i]) - ba(y2[i, c - 1], A200[i, c - 1] == 1) for c in passed])
    md = float(mean_diff(np.arange(200))); bs = [mean_diff(rng.integers(0, 200, 200)) for _ in range(1000)]; ci = np.nanpercentile(bs, [2.5, 97.5]).tolist()
    qd = [res[c]["heads"]["qwen"]["diff"] for c in GR if "diff" in res[c]["heads"].get("qwen", {})]
    out = {"heads": heads, "per_class": res, "gate": {"classes": passed, "mean_diff_ac_minus_astra": md, "ci": ci, "gate_E2": len(passed) > 0 and ci[0] > -0.02},
           "weak_to_strong_qwen": {"mean_diff": float(np.mean(qd)) if qd else None, "n_classes": len(qd), "n_ci_above_0": sum(res[c]["heads"]["qwen"]["diff_ci"][0] > 0 for c in GR if "diff_ci" in res[c]["heads"].get("qwen", {}))}}
    json.dump(out, open(f"{TH}/e2.json", "w"), indent=1, default=str)
    for c in res: print(c, CLASSES[c - 1][:22].ljust(22), {k: "%.2f%s" % (v["ba"], "(%+.2f)" % v["diff"] if "diff" in v else "") for k, v in res[c]["heads"].items() if "ba" in v}, "eq", res[c]["labels_equivalent_ac"])
    print(json.dumps({k: out[k] for k in ("gate", "weak_to_strong_qwen")}, indent=1, default=str))


if __name__ == "__main__":
    {"e1": e1, "e2": e2}[sys.argv[1]]()
