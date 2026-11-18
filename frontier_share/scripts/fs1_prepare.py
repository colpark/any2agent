"""FS-1: RL pool statistics and frozen thresholds, the 16 questions, keys for the RL pool and test100,
majority baselines, patch selection (3 x 3 NDVI x NDMI tertile grid on test100), anonymous ids, renderings.
Outputs: frontier_share/questions.json, frontier_share/fs1_facts.json, ~/langres_state/fs/keys_{rlpool,test100}.npz,
frontier_share/private/id_map.json, frontier_share/renders/{uid}_{tc,fc}.png
"""
import json, os, random, uuid, hashlib
from collections import Counter
import numpy as np, pandas as pd
from PIL import Image

H = os.path.expanduser("~"); ARR = f"{H}/langres_data/arrays"; FS = f"{H}/Documents/langres/frontier_share"
ST = f"{H}/langres_state/fs"; os.makedirs(ST, exist_ok=True)
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
B = {b: i for i, b in enumerate(BANDS)}
EPS = 1e-6
desc = pd.read_parquet(f"{ARR}/descriptors.parquet").set_index("patch_id")
frozen = pd.read_parquet(f"{ARR}/splits_frozen.parquet").set_index("patch_id")


def nd(a, b):
    return (a - b) / (a + b + EPS)


def stats(x):
    """x: (n, 12, 120, 120) float32 -> dict of per-patch statistics."""
    ndvi = nd(x[:, B["B08"]], x[:, B["B04"]]); ndwi = nd(x[:, B["B03"]], x[:, B["B08"]])
    ndmi = nd(x[:, B["B8A"]], x[:, B["B11"]]); ndbi = nd(x[:, B["B11"]], x[:, B["B08"]])
    b8 = x[:, B["B08"]]
    cells = (ndwi > 0).reshape(len(x), 3, 40, 3, 40).mean((2, 4)).reshape(len(x), 9)
    dh = np.abs(np.diff(b8, axis=2)); dv = np.abs(np.diff(b8, axis=1))
    edge = (dh.sum((1, 2)) + dv.sum((1, 2))) / (dh[0].size + dv[0].size)
    return {"ndvi": ndvi.mean((1, 2)), "ndmi": ndmi.mean((1, 2)), "ndbi": ndbi.mean((1, 2)),
            "water_maxcell": cells.max(1), "ndvi_left": ndvi[:, :, :60].mean((1, 2)), "ndvi_right": ndvi[:, :, 60:].mean((1, 2)),
            "ndmi_top": ndmi[:, :60].mean((1, 2)), "ndmi_bottom": ndmi[:, 60:].mean((1, 2)),
            "b08_std": b8.std((1, 2)), "b08_edge": edge}


def split_stats(split, ids_sub=None):
    ids = json.load(open(f"{ARR}/{split}_ids.json")); X = np.load(f"{ARR}/{split}_ms12.npy", mmap_mode="r")
    rows = range(len(ids)) if ids_sub is None else [ids.index(i) for i in ids_sub]
    rows = list(rows); out = []
    for k in range(0, len(rows), 500):
        out.append(stats(np.asarray(X[rows[k:k + 500]], dtype=np.float32)))
    S = {key: np.concatenate([o[key] for o in out]) for key in out[0]}
    return [ids[r] for r in rows], S


rl_ids, RL = split_stats("rlpool")
t100 = json.load(open(f"{ARR}/test100_ids.json"))
t_ids, TS = split_stats("test", t100)
assert t_ids == t100

# frozen thresholds (RL pool medians), rounded once and used both in the question text and in the keys
thr = {"ndmi_median": round(float(np.median(RL["ndmi"])), 3), "b08_std_median": round(float(np.median(RL["b08_std"])), 1),
       "b08_edge_median": round(float(np.median(RL["b08_edge"])), 1)}
lab_rl = [list(desc.loc[i, "labels"]) for i in rl_ids]
top6 = [l for l, _ in Counter(l for ls in lab_rl for l in ls).most_common(6)]

QS = [("Q1", "visible", "Is the mean NDVI over the whole patch above 0.5, meaning most of the patch is dense, healthy vegetation?"),
      ("Q2", "visible", "Is the mean NDVI over the whole patch below 0.2, meaning most of the patch is sparse or not vegetated?"),
      ("Q3", "visible", "Is open water present, meaning that in at least one cell of a 3 × 3 grid more than 5% of pixels have NDWI above 0?"),
      ("Q4", "visible", "Is vegetation denser in the left half than in the right half (higher mean NDVI on the left)?")]
for k, l in enumerate(top6):
    QS.append((f"Q{5 + k}", "visible", f"Does the patch contain any {l}, following the CORINE-based BigEarthNet land cover labels?"))
QS += [("Q11", "shortwave", f"Is the mean NDMI (B8A and B11) above {thr['ndmi_median']}, meaning the vegetation and soil are wetter than for a typical patch in this dataset?"),
       ("Q12", "shortwave", "Is the mean NDBI (B11 and B08) above 0, a sign of built-up or bare surfaces?"),
       ("Q13", "shortwave", "Is the mean NDMI higher in the top half than in the bottom half?"),
       ("Q14", "texture", f"Is the near-infrared texture rougher than for a typical patch in this dataset, meaning the B08 spatial standard deviation exceeds {thr['b08_std_median']}?"),
       ("Q15", "texture", "Does the patch carry more than 3 BigEarthNet land cover labels?"),
       ("Q16", "texture", f"Is the edge density higher than for a typical patch, meaning the mean absolute difference between neighboring B08 pixels exceeds {thr['b08_edge_median']}?")]


def keys(S, labels):
    cols = [S["ndvi"] > 0.5, S["ndvi"] < 0.2, S["water_maxcell"] > 0.05, S["ndvi_left"] > S["ndvi_right"]]
    cols += [np.array([l in ls for ls in labels]) for l in top6]
    cols += [S["ndmi"] > thr["ndmi_median"], S["ndbi"] > 0, S["ndmi_top"] > S["ndmi_bottom"],
             S["b08_std"] > thr["b08_std_median"], np.array([len(ls) > 3 for ls in labels]), S["b08_edge"] > thr["b08_edge_median"]]
    return np.stack(cols, 1).astype(bool)


K_rl = keys(RL, lab_rl)
lab_t = [list(desc.loc[i, "labels"]) for i in t100]
K_t = keys(TS, lab_t)
np.savez(f"{ST}/keys_rlpool.npz", keys=K_rl, **RL); np.savez(f"{ST}/keys_test100.npz", keys=K_t, **TS)
json.dump(rl_ids, open(f"{ST}/keys_rlpool_ids.json", "w")); json.dump(t100, open(f"{ST}/keys_test100_ids.json", "w"))
majority = K_rl.mean(0) >= 0.5
questions = [{"id": q, "tag": t, "text": x, "majority": "yes" if m else "no", "rl_yes_rate": round(float(r), 4)}
             for (q, t, x), m, r in zip(QS, majority, K_rl.mean(0))]
json.dump(questions, open(f"{FS}/questions.json", "w"), indent=1)

# selection: tertiles of mean NDVI and NDMI over test100, lowest sha256 per cell, plus lowest-hash remaining in the most populated cell
qv = np.quantile(TS["ndvi"], [1 / 3, 2 / 3]); qm = np.quantile(TS["ndmi"], [1 / 3, 2 / 3])
rv = np.digitize(TS["ndvi"], qv); rm = np.digitize(TS["ndmi"], qm)
sha = np.array([frozen.loc[i, "sha256"] for i in t100])
cells = {(a, b): sorted([j for j in range(100) if rv[j] == a and rm[j] == b], key=lambda j: sha[j]) for a in range(3) for b in range(3)}
pop = {c: len(v) for c, v in cells.items()}
most = max(pop, key=lambda c: (pop[c], -c[0], -c[1]))
sel, log_notes = [], []
for c in sorted(cells):
    if cells[c]:
        sel.append((cells[c][0], c))
    else:
        log_notes.append(f"cell {c} empty, filled from most populated cell {most}")
rest = [j for j in cells[most] if j not in [s for s, _ in sel]]
n_extra = 10 - len(sel)
for j in rest[:n_extra]:
    sel.append((j, most))
assert len(sel) == 10

random.seed(0)
uids = [str(uuid.uuid4()) for _ in sel]  # note: uuid4 draws from os.urandom, so random.seed(0) does not fix it
os.makedirs(f"{FS}/private", exist_ok=True); os.makedirs(f"{FS}/renders", exist_ok=True)
stretch = json.load(open(f"{ARR}/stretch_rlpool_p2p98.json"))
Xt = np.load(f"{ARR}/test_ms12.npy", mmap_mode="r"); tid_all = json.load(open(f"{ARR}/test_ids.json"))
idmap = []
for (j, c), u in zip(sel, uids):
    pid = t100[j]; x = np.asarray(Xt[tid_all.index(pid)])
    for tag, bands in [("tc", ["B04", "B03", "B02"]), ("fc", ["B08", "B04", "B03"])]:
        im = np.stack([(x[B[b]] - stretch[b][0]) / (stretch[b][1] - stretch[b][0]) for b in bands], -1)
        im = (np.clip(im, 0, 1) * 255).round().astype(np.uint8)
        Image.fromarray(im).resize((512, 512), Image.BICUBIC).save(f"{FS}/renders/{u}_{tag}.png")
    idmap.append({"uid": u, "patch_id": pid, "test100_index": j, "cell_ndvi_ndmi": list(c), "sha256": sha[j]})
json.dump(idmap, open(f"{FS}/private/id_map.json", "w"), indent=1)
json.dump({"uid_order": uids, "test100_index": [j for j, _ in sel]}, open(f"{ST}/selection.json", "w"), indent=1)

facts = {"thresholds": thr, "top6_labels": top6, "ndvi_tertile_cuts": qv.tolist(), "ndmi_tertile_cuts": qm.tolist(),
         "cell_population": {f"{a}{b}": pop[(a, b)] for a, b in sorted(pop)}, "most_populated_cell": list(most), "notes": log_notes,
         "rl_yes_rate": {q["id"]: q["rl_yes_rate"] for q in questions}, "majority": {q["id"]: q["majority"] for q in questions},
         "selected_key_yes_rate": {QS[k][0]: float(K_t[[j for j, _ in sel]][:, k].mean()) for k in range(16)},
         "selected_cells": [list(c) for _, c in sel]}
json.dump(facts, open(f"{FS}/fs1_facts.json", "w"), indent=1)
print(json.dumps(facts, indent=1))
