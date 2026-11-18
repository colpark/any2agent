"""FB-1 (CPU): Q17 regression and keys, Q17 Prithvi probe, bank selection (1,000 RL-pool patches stratified by label set,
seed 0, largest remainder as scripts/make_splits.py), neighbors (cosine on standardized Prithvi features) and random controls.
Outputs: ~/langres_state/fb/{q17.npz, bank.json, neighbors.json}, frontier_share/fb/{q17_facts.json, questions17.json}."""
import json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

H = os.path.expanduser("~"); ARR = f"{H}/langres_data/arrays"; PR = f"{H}/langres_data/prithvi"
FS = f"{H}/Documents/langres/frontier_share"; ST = f"{H}/langres_state/fs"; FB = f"{H}/langres_state/fb"
os.makedirs(FB, exist_ok=True); os.makedirs(f"{FS}/fb", exist_ok=True)
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]; B = {b: i for i, b in enumerate(BANDS)}
EPS = 1e-6
nd = lambda a, b: (a - b) / (a + b + EPS)


def feats(split, ids_sub=None):
    ids = json.load(open(f"{ARR}/{split}_ids.json")); X = np.load(f"{ARR}/{split}_ms12.npy", mmap_mode="r")
    rows = list(range(len(ids))) if ids_sub is None else [ids.index(i) for i in ids_sub]
    out = []
    for k in range(0, len(rows), 500):
        x = np.asarray(X[rows[k:k + 500]], dtype=np.float32)
        out.append(np.stack([x[:, B["B02"]].mean((1, 2)), x[:, B["B03"]].mean((1, 2)), x[:, B["B04"]].mean((1, 2)), x[:, B["B08"]].mean((1, 2)),
                             nd(x[:, B["B08"]], x[:, B["B04"]]).mean((1, 2)), nd(x[:, B["B03"]], x[:, B["B08"]]).mean((1, 2)),
                             nd(x[:, B["B8A"]], x[:, B["B11"]]).mean((1, 2))], 1))
    return np.concatenate(out)


rl_ids = json.load(open(f"{ARR}/rlpool_ids.json")); t100 = json.load(open(f"{ARR}/test100_ids.json"))
Frl = feats("rlpool"); Ft = feats("test", t100)
# consistency with the pilot's stored NDMI means
assert np.allclose(Frl[:, 6], np.load(f"{ST}/keys_rlpool.npz")["ndmi"], atol=1e-5)
reg = LinearRegression().fit(Frl[:, :6], Frl[:, 6]); r2 = float(reg.score(Frl[:, :6], Frl[:, 6]))
res_rl = Frl[:, 6] - reg.predict(Frl[:, :6]); res_t = Ft[:, 6] - reg.predict(Ft[:, :6])
y_rl, y_t = res_rl > 0, res_t > 0
np.savez(f"{FB}/q17.npz", y_rlpool=y_rl, y_test100=y_t, res_rlpool=res_rl, res_test100=res_t, coef=reg.coef_, intercept=reg.intercept_)

Xr = np.load(f"{PR}/rlpool_prithvi.npy"); Xt = np.load(f"{PR}/test100_prithvi.npy")
clf = LogisticRegression(C=1.0, max_iter=3000)
cv = cross_val_score(clf, Xr, y_rl, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="accuracy")
clf.fit(Xr, y_rl); pt = clf.predict(Xt)
sel = json.load(open(f"{ST}/selection.json")); U, TI = sel["uid_order"], sel["test100_index"]
q17_text = ("Is the patch wetter than its visible and near-infrared appearance predicts, meaning its NDMI (B8A and B11) exceeds "
            "the value expected from its visible and near-infrared statistics?")
yes_rate = float(y_rl.mean())
facts = {"ols_inputs": ["mean_B02", "mean_B03", "mean_B04", "mean_B08", "mean_NDVI", "mean_NDWI"], "ols_target": "mean_NDMI",
         "r2_rlpool": r2, "coef": reg.coef_.tolist(), "intercept": float(reg.intercept_), "rl_yes_rate": yes_rate,
         "majority": "yes" if yes_rate >= 0.5 else "no", "probe_cv_acc": float(cv.mean()), "probe_cv_folds": cv.tolist(),
         "probe_majority_acc": max(yes_rate, 1 - yes_rate), "probe_test100_acc": float((pt == y_t).mean()), "probe_n_iter": int(clf.n_iter_[0]),
         "key_selected": {u: bool(y_t[j]) for u, j in zip(U, TI)}, "probe_pred_selected": {u: bool(pt[j]) for u, j in zip(U, TI)},
         "residual_selected": {u: float(res_t[j]) for u, j in zip(U, TI)}}
json.dump(facts, open(f"{FS}/fb/q17_facts.json", "w"), indent=1)
qs = json.load(open(f"{FS}/questions.json"))
qs17 = qs + [{"id": "Q17", "tag": "q17", "text": q17_text, "majority": facts["majority"], "rl_yes_rate": round(yes_rate, 4)}]
json.dump(qs17, open(f"{FS}/fb/questions17.json", "w"), indent=1)

# bank: label-set stratified 1,000 from the RL pool, seed 0, largest remainder (same procedure as make_splits.strat)
rng = np.random.default_rng(0)
fr = pd.read_parquet(f"{ARR}/splits_frozen.parquet"); rl = fr[fr.split == "rlpool"].copy()
rl["lset"] = rl["labels"].apply(lambda l: "|".join(sorted(l)))
counts = rl["lset"].value_counts().sort_index(); quota = counts / counts.sum() * 1000
base = np.floor(quota).astype(int); frac = (quota - base) + rng.uniform(0, 1e-9, len(quota))
base[frac.sort_values(ascending=False).index[: 1000 - base.sum()]] += 1
bank = np.concatenate([rng.choice(np.sort(rl.loc[rl.lset == ls, "patch_id"].to_numpy()), k, replace=False) for ls, k in base.items() if k > 0])
assert len(bank) == 1000 and len(set(bank)) == 1000
pos = {p: i for i, p in enumerate(rl_ids)}; bank_idx = [pos[p] for p in bank]
json.dump({"patch_ids": bank.tolist(), "rlpool_index": bank_idx, "n_label_sets": int((base > 0).sum())}, open(f"{FB}/bank.json", "w"))

# neighbors: cosine on standardized features; random control: default_rng(0) per target (as written: seed 0 for each target)
Fb = Xr[bank_idx]; Fb_u = Fb / np.linalg.norm(Fb, axis=1, keepdims=True)
NB = {}
for u, j in zip(U, TI):
    t = Xt[j] / np.linalg.norm(Xt[j]); cs = Fb_u @ t; top = np.argsort(-cs)[:5]
    rnd = np.random.default_rng(0).choice(1000, 5, replace=False)
    NB[u] = {"neighbors": [[int(k), float(cs[k])] for k in top], "random": [[int(k), float(cs[k])] for k in rnd]}
json.dump(NB, open(f"{FB}/neighbors.json", "w"), indent=1)
print(json.dumps({k: facts[k] for k in ["r2_rlpool", "rl_yes_rate", "majority", "probe_cv_acc", "probe_majority_acc", "probe_test100_acc"]}, indent=1))
print("Q17 keys selected:", sum(facts["key_selected"].values()), "/10; probe right:", sum(facts["key_selected"][u] == facts["probe_pred_selected"][u] for u in U))
print("label sets in bank:", int((base > 0).sum()), "; neighbor cos range:", min(c for v in NB.values() for _, c in v["neighbors"]), max(c for v in NB.values() for _, c in v["neighbors"]))
