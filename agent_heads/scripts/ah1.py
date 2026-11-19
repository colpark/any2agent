"""AH-1 inputs and workspaces. Usage: ah1.py captions (node A2) | prithvi | fields | embed | desc | ws
captions/prithvi/fields: the VS steps (verbal_share/scripts/vs1.py) unchanged, pointed at the 500 bank rows and ~/langres_state/ah.
embed: vs1.embed's model and encode settings on the bank (vs1.embed loops over fit/test only). desc: VS descriptors for bank and Test.
ws: the two agent workspaces and their label check."""
import json, os, re, shutil, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from ah_common import *
import vs1, vs_common
BANK = json.load(open(f"{TH}/sets.json"))["bank_rl"]
vs1.ST = ST; vs1.SPLITS = lambda: (("bank", "rlpool", BANK),)
FORMULAS = {"Brightness": "mean over pixels of (B02 + B03 + B04) / 3", "NDVI mean": "mean of (B08 - B04) / (B08 + B04)",
            "NDVI standard deviation": "standard deviation over pixels of (B08 - B04) / (B08 + B04)", "Vegetated share": "share of pixels with NDVI > 0.5",
            "NDWI mean": "mean of (B03 - B08) / (B03 + B08)", "Water share": "share of pixels with NDWI > 0", "NDMI mean": "mean of (B8A - B11) / (B8A + B11)",
            "SWIR1 mean": "mean of B11", "SWIR2 mean": "mean of B12", "NIR texture": "standard deviation over pixels of B08"}
NOTE = "Bands are raw L2A digital numbers (reflectance x 10,000). Normalized differences use a denominator clipped below at 1e-6."


def prithvi():
    try:
        vs1.prithvi()
    except FileNotFoundError:  # vs1's closing Test check needs stage1_test; check the bank token against the stage 1 cache instead
        ref = np.load(f"{H}/langres_data/prithvi/rlpool_prithvi_raw.npy")[BANK]; new = np.load(f"{ST}/stage1_bank.npy")
        print("stage-1 bank token reproduced: max abs diff", float(np.abs(ref - new).max()))


def embed():
    import torch
    from sentence_transformers import SentenceTransformer
    vs1.evict(); m = SentenceTransformer(vs_common.EMB, revision=vs_common.EREV, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}); t0 = time.time()
    R = [json.loads(l) for l in open(f"{AH}/fields_bank.jsonl")]; E = np.full((len(R), 7, m.get_sentence_embedding_dimension()), np.nan, np.float32)
    ok = [k for k, r in enumerate(R) if r["valid"]]
    for j, fl in enumerate(vs_common.FIELDS):
        E[ok, j] = m.encode([R[k]["fields"][fl] for k in ok], batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
    np.save(f"{ST}/emb_bank.npy", E); print("bank", E.shape, "valid", len(ok), round(time.time() - t0), "s")


def desc():
    S = sets(); np.save(f"{ST}/desc_bank.npy", vs_common.descriptors("rlpool", BANK)); np.save(f"{ST}/desc_test.npy", vs_common.descriptors("test", S["test"]))
    json.dump({"names": vs_common.DESC, "formulas": FORMULAS, "note": NOTE}, open(f"{ST}/descriptors.json", "w"), indent=1)


def label_check(ws):
    """No file may hold labels: expected files only, no patch ids, no label keys, no column that tracks a true label (|r| >= 0.99)."""
    S = sets(); ok = bank_ok() if os.path.exists(f"{ST}/emb_bank.npy") else np.ones(500, bool)
    Yb, Yt, Y5 = truth("rlpool", np.array(BANK)[ok]), truth("test", S["test"]), truth("rlpool", BANK[:50]); bad = []
    for f in sorted(os.listdir(ws)):
        p = f"{ws}/{f}"
        if f.endswith(".npy"):
            X = np.load(p); X = X.reshape(len(X), -1) if X.ndim > 1 else None
            if X is not None and X.shape[0] in (len(Yb), len(Yt), 50, 500):
                Y = {len(Yb): Yb, len(Yt): Yt, 50: Y5, 500: truth("rlpool", BANK)}[X.shape[0]]; X = X[:, :20000:max(1, X.shape[1] // 5000)]; Xs = (X - X.mean(0)) / (X.std(0) + 1e-12); Ys = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
                r = np.abs(Xs.T @ Ys / len(X)).max()
                if r >= .99: bad.append((f, "column tracks a label", float(r)))
        else:
            t = open(p).read()
            if re.search(r"S2[AB]_MSIL2A|\"labels\"|_ids\.json", t): bad.append((f, "id or label key", ""))
    return bad


def ws(kind):
    S = sets(); C = classes(); rep = json.load(open(f"{AH}/workspaces.json")) if os.path.exists(f"{AH}/workspaces.json") else {}
    if kind == "free": return ws_free(S, C, rep)
    w = f"{ST}/ws_code"; os.makedirs(w)  # code workspace (read-only call)
    np.save(f"{w}/bank_descriptors.npy", np.load(f"{ST}/desc_bank.npy").astype(np.float32)); shutil.copy(f"{ST}/descriptors.json", f"{w}/descriptors.json")
    X = np.load(f"{ARR}/rlpool_ms12.npy", mmap_mode="r"); np.save(f"{w}/sample_bands.npy", np.stack([np.asarray(X[i], np.float32) for i in BANK[:50]]))
    json.dump(C, open(f"{w}/classes.json", "w"), indent=1); rep["ws_code"] = {"files": sorted(os.listdir(w)), "label_check": label_check(w)}
    json.dump(rep, open(f"{AH}/workspaces.json", "w"), indent=1); print(json.dumps(rep, indent=1))


def ws_free(S, C, rep):
    ok = bank_ok(); w2 = f"{ST}/ws_free"; os.makedirs(w2); I = inputs(); bidx, tidx = np.array(BANK)[ok], S["test"]
    np.save(f"{w2}/bank_features.npy", I["F"][0].astype(np.float32)); np.save(f"{w2}/test_features.npy", I["F"][1].astype(np.float32))
    np.save(f"{w2}/bank_descriptors.npy", np.load(f"{ST}/desc_bank.npy")[ok].astype(np.float32)); np.save(f"{w2}/test_descriptors.npy", np.load(f"{ST}/desc_test.npy").astype(np.float32))
    shutil.copy(f"{ST}/descriptors.json", f"{w2}/descriptors.json"); json.dump(C, open(f"{w2}/classes.json", "w"), indent=1)
    cap_b = [json.loads(l) for l in open(f"{ST}/captions_bank.jsonl")]; cap_t = {r["split_index"]: r for r in map(json.loads, open(f"{VSS}/captions_test.jsonl"))}
    def dump(name, rows):
        with open(f"{w2}/{name}", "w") as f:
            for k, r in enumerate(rows): f.write(json.dumps({"row": k, **r}) + "\n")
    dump("bank_captions.jsonl", [{"caption": json.loads(r["raw"])} for r, o in zip(cap_b, ok) if o])
    dump("test_captions.jsonl", [{"caption": json.loads(cap_t[i]["raw"])} for i in tidx])
    def answers(fn, idx):
        A, P = fn(idx); inv = {v: k for k, v in SCORE.items()}
        return [{"answers": {str(c): ({"answer": inv[P[k, j]][0], "confidence": inv[P[k, j]][1]} if A[k, j] >= 0 else None) for j, c in enumerate(CL)}} for k in range(len(idx))]
    dump("bank_qwen.jsonl", answers(lambda i: qwen("bank", i), bidx)); dump("test_qwen.jsonl", answers(lambda i: qwen("test", i), tidx))
    dump("bank_astra.jsonl", answers(lambda i: astra("bank", i), bidx))
    json.dump([f"t{k:04d}" for k in range(len(tidx))], open(f"{w2}/test_ids.json", "w"))
    rep["ws_free"] = {"files": sorted(os.listdir(w2)), "label_check": label_check(w2), "bank_rows": int(ok.sum())}
    json.dump(rep, open(f"{AH}/workspaces.json", "w"), indent=1); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "fields": vs1.VS = AH  # fields_bank.jsonl goes to agent_heads (captions still read the VS prompt and schema)
    {"captions": vs1.captions, "prithvi": prithvi, "fields": vs1.fields, "embed": embed, "desc": desc, "ws": lambda: ws(sys.argv[2])}[sys.argv[1]]()
