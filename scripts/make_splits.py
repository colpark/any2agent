"""LR-1: frozen splits from reBEN metadata only (no pixels, no models).
calib 200 + rlpool 20,000 from official train (disjoint), val 100 from official validation, test 1,000 from official test.
Stratified by exact label set: proportional allocation, largest-remainder rounding, random within stratum. Seed 0.
"""
import json, sys
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
meta = pd.read_parquet(sys.argv[1]); out = sys.argv[2]
meta["lset"] = meta["labels"].apply(lambda l: "|".join(sorted(l)))


def strat(df, n):
    counts = df["lset"].value_counts().sort_index()
    quota = counts / counts.sum() * n
    base = np.floor(quota).astype(int)
    frac = (quota - base) + rng.uniform(0, 1e-9, len(quota))
    base[frac.sort_values(ascending=False).index[: n - base.sum()]] += 1
    return np.concatenate([rng.choice(np.sort(df.loc[df.lset == ls, "patch_id"].to_numpy()), k, replace=False)
                           for ls, k in base.items() if k > 0])


tr = meta[meta.split == "train"]
both = strat(tr, 20200)
calib = strat(tr[tr.patch_id.isin(both)], 200)
rlpool = np.setdiff1d(both, calib)
val = strat(meta[meta.split == "validation"], 100)
test = strat(meta[meta.split == "test"], 1000)
S = {"calib": calib, "rlpool": rlpool, "val": val, "test": test}
assert [len(S[k]) for k in S] == [200, 20000, 100, 1000]
allids = np.concatenate(list(S.values())); assert len(set(allids)) == len(allids)
rows = [{"patch_id": p, "split": k} for k, v in S.items() for p in sorted(v)]
df = pd.DataFrame(rows).merge(meta[["patch_id", "labels", "country", "split"]].rename(columns={"split": "official_split"}), on="patch_id")
df["labels"] = df["labels"].apply(list)
df.to_parquet(out, index=False)
print(json.dumps({k: len(v) for k, v in S.items()}),
      {k: int(df[df.split == k].labels.explode().nunique()) for k in S})
