"""Shared VS helpers (DESIGN_VS_v1.md): paths, sets, labels, Q17 key, code descriptions, caption fields, metrics."""
import json, os, sys
import numpy as np, pandas as pd
H = os.path.expanduser("~"); REPO = f"{H}/Documents/langres"; VS = f"{REPO}/verbal_share"; ST = f"{H}/langres_state/vs"
ARR = f"{H}/langres_data/arrays"; os.makedirs(ST, exist_ok=True)
sys.path.insert(0, f"{REPO}/teacher_heads/scripts")
QWEN, QREV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
EMB, EREV = "Qwen/Qwen3-Embedding-4B", "5cf2132abc99cad020ac570b19d031efec650f2b"
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]; BI = {b: i for i, b in enumerate(BANDS)}
LAYERS_F = [5, 11, 17, 23]  # blocks 6, 12, 18, 24 (0-based)
FIELDS = ["global", "layout", "features", "spectral", "interpretation", "summary", "code"]


def sets():
    """Fit: 2,000 RL-pool patches, make_splits.strat on exact label sets, seed 0. Test: the frozen 1,000."""
    p = f"{VS}/sets.json"
    if os.path.exists(p): return json.load(open(p))
    from th_common import strat
    d = pd.read_parquet(f"{ARR}/descriptors.parquet").set_index("patch_id"); rid = json.load(open(f"{ARR}/rlpool_ids.json"))
    fit = strat(rid, ["|".join(sorted(d.loc[i, "labels"])) for i in rid], 2000, seed=0); pos = {q: i for i, q in enumerate(rid)}
    s = {"fit_rl": sorted(pos[q] for q in fit), "test": list(range(1000))}
    json.dump(s, open(p, "w")); return s


def labels(split, idx):  # VS-3 only
    from th_common import CLASSES
    ids = json.load(open(f"{ARR}/{split}_ids.json")); d = pd.read_parquet(f"{ARR}/descriptors.parquet").set_index("patch_id")
    return np.array([[c in d.loc[ids[i], "labels"] for c in CLASSES] for i in idx], bool)


def nd(a, b): return (a - b) / np.clip(a + b, 1e-6, None)  # build_arrays.py convention


def q17(split, idx):
    """FB key: sign of the NDMI residual after the RL-pool OLS stored in ~/langres_state/fb/q17.npz (FB uses eps 1e-6 in a+b)."""
    z = np.load(f"{H}/langres_state/fb/q17.npz"); X = np.load(f"{ARR}/{split}_ms12.npy", mmap_mode="r"); x = np.asarray(X[idx], np.float32)
    e = lambda a, b: (a - b) / (a + b + 1e-6)
    f = np.stack([x[:, BI["B02"]].mean((1, 2)), x[:, BI["B03"]].mean((1, 2)), x[:, BI["B04"]].mean((1, 2)), x[:, BI["B08"]].mean((1, 2)),
                  e(x[:, BI["B08"]], x[:, BI["B04"]]).mean((1, 2)), e(x[:, BI["B03"]], x[:, BI["B08"]]).mean((1, 2))], 1)
    ndmi = e(x[:, BI["B8A"]], x[:, BI["B11"]]).mean((1, 2))
    return ndmi - (f @ z["coef"] + z["intercept"]) > 0


DESC = ["Brightness", "NDVI mean", "NDVI standard deviation", "Vegetated share", "NDWI mean", "Water share", "NDMI mean",
        "SWIR1 mean", "SWIR2 mean", "NIR texture"]


def descriptors(split, idx):
    X = np.load(f"{ARR}/{split}_ms12.npy", mmap_mode="r"); out = []
    for k in range(0, len(idx), 250):
        x = np.asarray(X[idx[k:k + 250]], np.float32); g = lambda b: x[:, BI[b]]
        ndvi, ndwi, ndmi = nd(g("B08"), g("B04")), nd(g("B03"), g("B08")), nd(g("B8A"), g("B11"))
        out.append(np.stack([(g("B02") + g("B03") + g("B04")).mean((1, 2)) / 3, ndvi.mean((1, 2)), ndvi.std((1, 2)), (ndvi > 0.5).mean((1, 2)),
                             ndwi.mean((1, 2)), (ndwi > 0).mean((1, 2)), ndmi.mean((1, 2)), g("B11").mean((1, 2)), g("B12").mean((1, 2)),
                             g("B08").std((1, 2))], 1))
    return np.concatenate(out)


def code_text(v, fit_v):
    word = lambda p: "very low" if p < 10 else "low" if p < 30 else "typical" if p <= 70 else "high" if p <= 90 else "very high"
    s = []
    for j, name in enumerate(DESC):
        p = int(round(100 * np.mean(fit_v[:, j] <= v[j]))); s.append(f"{name} is {word(p)} ({p}th percentile of the region), value {v[j]:.2f}.")
    return " ".join(s)


def caption_fields(c):
    """Six plain-text fields from one caption JSON (uncertainties are logged, not embedded)."""
    g = c["global"]; cell = lambda s: s.replace("_", " ")
    f1 = " ".join([f"{x['class']} covers about {x['percent']:g} percent." for x in g["cover"]] +
                  [f"Landscape: {g['landscape']}.", f"Season cues: {g['season_cues']}.", f"Heterogeneity: {g['heterogeneity']}."])
    f2 = " ".join([f"The {cell(x['cell'])} cell is {x['cover']}, with {x['texture']} texture; features: {x['features']}." for x in c["grid"]] +
                  [f"Boundaries: {c['boundaries']}."])
    f3 = " ".join(f"{x['count']} {x['type']} in the {cell(x['cell'])} cell, about {x['size_m']} m, {x['shape']}, oriented {x['orientation']}."
                  for x in c["features"]) or "No discrete features listed."
    sp = c["spectral"]
    f4 = " ".join([f"True color: {sp['true_color']}.", f"False color: {sp['false_color']}."] +
                  [f"{x['claim']} ({x['confidence']} confidence)." for x in sp["implications"]] + [f"Not revealed: {sp['not_revealed']}."])
    f5 = " ".join(f"{x['claim']} ({x['confidence']} confidence; evidence: {x['evidence']})." for x in c["interpretation"]) or "No interpretation given."
    return [f1, f2, f3, f4, f5, c["summary"]]


def r2w(F, P, mu):  # variance-weighted R2 against the Fit mean
    return 1 - ((F - P) ** 2).sum() / ((F - mu) ** 2).sum()


def boot_r2(F, P, mu, B=1000, seed=0):
    rng = np.random.default_rng(seed); e, t = ((F - P) ** 2).sum(1), ((F - mu) ** 2).sum(1); n = len(F)
    b = [1 - e[i].sum() / t[i].sum() for i in (rng.integers(0, n, n) for _ in range(B))]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
