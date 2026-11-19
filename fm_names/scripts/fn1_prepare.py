"""FN-1: PCA (20) on the discovery features, scores, end assignment, exemplars, held-out pairs, composites, diagnostic.
Outputs: fm_names/fn1.json (no patch ids), ~/langres_state/fn/fn1_private.json (ids), composites in ~/langres_state/fn/renders."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from fn_common import *

disc = json.load(open(f"{H}/langres_state/fb/bank.json"))["rlpool_index"]
Xd = np.load(f"{PR}/rlpool_prithvi.npy")[disc]; Xh = np.load(f"{PR}/test_prithvi.npy")
pca = PCA(n_components=20, svd_solver="full").fit(Xd)
Sd, Sh = pca.transform(Xd), pca.transform(Xh)  # projection after centering on the discovery mean
ends, exem, pairs = {}, {}, {}
for d in range(20):
    a_high = bool(np.random.default_rng(d).integers(0, 2)); ends[d] = "high" if a_high else "low"
    o = np.argsort(Sd[:, d]); A_, B_ = (o[::-1][:6], o[:6]) if a_high else (o[:6], o[::-1][:6])
    A_, B_ = np.random.default_rng(d).permutation(A_), np.random.default_rng(d).permutation(B_)
    exem[d] = {"A": [int(disc[i]) for i in A_], "B": [int(disc[i]) for i in B_]}
    oh = np.argsort(Sh[:, d]); top, bot = oh[::-1][:200], oh[:200]
    rng = np.random.default_rng(d); t, b = rng.choice(top, 40, replace=False), rng.choice(bot, 40, replace=False)
    aend, bend = (t, b) if a_high else (b, t); flip = np.random.default_rng(1000 + d).integers(0, 2, 40)  # 0: A at position 1
    pairs[d] = [{"a": int(x), "b": int(y), "a_pos": int(f) + 1} for x, y, f in zip(aend, bend, flip)]
# diagnostic (report only): band mean and std on the discovery patches
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]; BI = {b: i for i, b in enumerate(BANDS)}
M = np.load(f"{ARR}/rlpool_ms12.npy", mmap_mode="r"); F = []
for k in range(0, len(disc), 250):
    x = np.asarray(M[disc[k:k + 250]], np.float32); F.append(np.concatenate([x[:, [BI[b] for b in BANDS if b in ("B02", "B03", "B04", "B8A", "B11", "B12")]].mean((2, 3)),
                                                                            x[:, [BI[b] for b in BANDS if b in ("B02", "B03", "B04", "B8A", "B11", "B12")]].std((2, 3))], 1))
F = np.concatenate(F)  # columns: means of B02,B03,B04,B8A,B11,B12 then stds in the same order
vis = [0, 1, 2, 3, 6, 7, 8, 9]; diag = {}
for d in range(20):
    rv = LinearRegression().fit(F[:, vis], Sd[:, d]).score(F[:, vis], Sd[:, d]); rf = LinearRegression().fit(F, Sd[:, d]).score(F, Sd[:, d])
    diag[d] = {"visible_r2": float(rv), "full_r2": float(rf), "swir_gain": float(rf - rv)}
need_d = sorted({i for e in exem.values() for i in e["A"] + e["B"]}); need_h = sorted({i for p in pairs.values() for q in p for i in (q["a"], q["b"])})
comp_d, comp_h = composites("rlpool", need_d), composites("test", need_h)
json.dump({"evr": pca.explained_variance_ratio_.tolist(), "evr_sum": float(pca.explained_variance_ratio_.sum()), "ends": ends, "diag": diag,
           "exemplar_images": {d: {g: [comp_d[i] for i in e[g]] for g in "AB"} for d, e in exem.items()},
           "pairs": {d: [{"img_a": comp_h[q["a"]], "img_b": comp_h[q["b"]], "a_pos": q["a_pos"]} for q in p] for d, p in pairs.items()}},
          open(f"{FN}/fn1.json", "w"), indent=1)
json.dump({"exemplars_rlpool_index": exem, "pairs_test_index": pairs}, open(f"{ST}/fn1_private.json", "w"))
print("EVR", np.round(pca.explained_variance_ratio_, 3).tolist(), "sum", round(float(pca.explained_variance_ratio_.sum()), 3))
print("ends", ends); print("swir_gain", {d: round(v["swir_gain"], 3) for d, v in diag.items()}); print("composites", len(comp_d), len(comp_h))
