"""LR-1: read extracted tifs -> 12-band 10 m float32 arrays, per-patch sha256, descriptors,
RGB / false-color composites (2-98 pct stretch per band computed on the RL pool).
Usage: build_arrays.py <extracted_root> <splits_membership.parquet> <out_dir>
"""
import sys, json, hashlib, os
import numpy as np, pandas as pd, rasterio, torch
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
BI = {b: i for i, b in enumerate(BANDS)}
root, mem_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
mem = pd.read_parquet(mem_path)
SPLITS = ["calib", "rlpool", "val", "test"]


def load(pid):
    d = os.path.join(root, "BigEarthNet-S2", pid.rsplit("_", 2)[0], pid)
    h = hashlib.sha256(); arr = np.zeros((12, 120, 120), np.float32)
    for i, b in enumerate(BANDS):
        fp = os.path.join(d, f"{pid}_{b}.tif")
        with open(fp, "rb") as f:
            h.update(f.read())
        with rasterio.open(fp) as src:
            x = src.read(1).astype(np.float32)
        if x.shape != (120, 120):  # 20 m (60x60) and 60 m (20x20) bands -> 10 m bilinear
            x = F.interpolate(torch.from_numpy(x)[None, None], size=(120, 120), mode="bilinear",
                              align_corners=False)[0, 0].numpy()
        arr[i] = x
    return arr, h.hexdigest()


def ndi(a, b):
    return (a - b) / np.clip(a + b, 1e-6, None)


shas = {}
for split in SPLITS:
    ids = mem.loc[mem.split == split, "patch_id"].tolist()
    X = np.lib.format.open_memmap(f"{out}/{split}_ms12.npy", "w+", np.float32, (len(ids), 12, 120, 120))
    with ProcessPoolExecutor(16) as ex:
        for k, (arr, sh) in enumerate(ex.map(load, ids, chunksize=64)):
            X[k] = arr; shas[ids[k]] = sh
    X.flush()
    json.dump(ids, open(f"{out}/{split}_ids.json", "w"))
    print(split, len(ids), flush=True)

mem["sha256"] = mem.patch_id.map(shas)
mem.to_parquet(f"{out}/splits_frozen.parquet", index=False)

# stretch on RL pool
Xtr = np.load(f"{out}/rlpool_ms12.npy", mmap_mode="r")
stretch = {}
for b in ["B02", "B03", "B04", "B08"]:
    v = np.asarray(Xtr[:, BI[b]]).ravel()
    lo, hi = np.percentile(v, [2, 98]); stretch[b] = [float(lo), float(hi)]
json.dump(stretch, open(f"{out}/stretch_rlpool_p2p98.json", "w"), indent=1)


def comp(X, bands):
    c = np.stack([(np.asarray(X[:, BI[b]]) - stretch[b][0]) / (stretch[b][1] - stretch[b][0]) for b in bands], -1)
    return (np.clip(c, 0, 1) * 255).round().astype(np.uint8)


rows = []
for split in SPLITS:
    X = np.load(f"{out}/{split}_ms12.npy", mmap_mode="r")
    np.save(f"{out}/{split}_rgb.npy", comp(X, ["B04", "B03", "B02"]))
    np.save(f"{out}/{split}_fcc.npy", comp(X, ["B08", "B04", "B03"]))
    ids = json.load(open(f"{out}/{split}_ids.json"))
    for k in range(0, len(ids), 1000):
        x = np.asarray(X[k:k + 1000])
        nd = ndi(x[:, BI["B08"]], x[:, BI["B04"]]).mean((1, 2))
        nw = ndi(x[:, BI["B03"]], x[:, BI["B08"]]).mean((1, 2))
        nm = ndi(x[:, BI["B8A"]], x[:, BI["B11"]]).mean((1, 2))
        for j in range(len(x)):
            rows.append({"patch_id": ids[k + j], "split": split, "ndvi": float(nd[j]), "ndwi": float(nw[j]), "ndmi": float(nm[j])})
desc = pd.DataFrame(rows).merge(mem[["patch_id", "labels"]], on="patch_id")
desc.to_parquet(f"{out}/descriptors.parquet", index=False)
print(desc.groupby("split")[["ndvi", "ndwi", "ndmi"]].describe().T.to_string())
