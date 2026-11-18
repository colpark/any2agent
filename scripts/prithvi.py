"""LR-2: cache Prithvi z (secondary target).
Prithvi-EO-2.0-300M encoder, num_frames=1, B02,B03,B04,B8A,B11,B12 resized to 224 (bilinear),
       model's own mean/std normalization, mean of last-layer (normed) patch tokens, CLS dropped.
Standardize per dimension with RL pool mean/std. Usage: targets.py <arrays_dir> <out_dir>
"""
import sys, os, json, glob, importlib.util
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel

A, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
MS_IDX = [BANDS.index(b) for b in ["B02", "B03", "B04", "B8A", "B11", "B12"]]
revs = json.load(open(os.path.expanduser("~/Documents/langres/model_revisions.json")))

# ---- Prithvi
pdir = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--ibm-nasa-geospatial--Prithvi-EO-2.0-300M/snapshots/" + revs["ibm-nasa-geospatial/Prithvi-EO-2.0-300M"]["revision"]))[0]
spec = importlib.util.spec_from_file_location("prithvi_mae", os.path.join(pdir, "prithvi_mae.py"))
pm = importlib.util.module_from_spec(spec); spec.loader.exec_module(pm)
cfg = json.load(open(os.path.join(pdir, "config.json")))["pretrained_cfg"]
enc = pm.PrithviViT(img_size=224, num_frames=1, patch_size=cfg["patch_size"], in_chans=6, embed_dim=cfg["embed_dim"],
                    depth=cfg["depth"], num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"],
                    coords_encoding=cfg["coords_encoding"], coords_scale_learn=cfg["coords_scale_learn"])
sd = torch.load(os.path.join(pdir, "Prithvi_EO_V2_300M.pt"), map_location="cpu", weights_only=True)
sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
sd = {k: v for k, v in sd.items() if "pos_embed" not in k}  # fixed sincos, rebuilt for num_frames=1
missing, unexpected = enc.load_state_dict(sd, strict=False)
print("prithvi missing", missing, "unexpected", unexpected, flush=True)
assert all("pos_embed" in k for k in missing) and not unexpected
enc = enc.to(dev).eval()
P_MEAN = torch.tensor(cfg["mean"], device=dev).view(1, 6, 1, 1)
P_STD = torch.tensor(cfg["std"], device=dev).view(1, 6, 1, 1)


@torch.no_grad()
def run(split):
    ms = np.load(f"{A}/{split}_ms12.npy", mmap_mode="r")
    zm = []
    for k in range(0, len(ms), 128):
        y = torch.from_numpy(np.asarray(ms[k:k + 128])[:, MS_IDX]).to(dev)
        y = F.interpolate(y, size=224, mode="bilinear", align_corners=False)
        y = ((y - P_MEAN) / P_STD).unsqueeze(2)  # B,C,T=1,H,W
        feats = enc.forward_features(y)[-1]
        zm.append(feats[:, 1:].float().mean(1).cpu())
    return torch.cat(zm).numpy()


SPL = ["calib", "rlpool", "val", "test"]
Z = {s: run(s) for s in SPL}
mu = Z["rlpool"].mean(0); sd_ = Z["rlpool"].std(0) + 1e-6
np.savez(f"{OUT}/prithvi_stats_rlpool.npz", mean=mu, std=sd_)
for s in SPL:
    np.save(f"{OUT}/{s}_prithvi_raw.npy", Z[s]); np.save(f"{OUT}/{s}_prithvi.npy", ((Z[s] - mu) / sd_).astype(np.float32))
print({s: Z[s].shape for s in SPL}, flush=True)
