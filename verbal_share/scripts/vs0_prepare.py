"""VS-0: design hash, Fit set, schema copy, checks of arrays, features, labels and the Q17 key. Writes verbal_share/vs0.json."""
import copy, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from vs_common import *

out = {"design_sha256": hashlib.sha256(open(f"{VS}/DESIGN_VS_v1.md", "rb").read()).hexdigest()}
S = sets(); fit = S["fit_rl"]; out["n_fit"], out["n_test"] = len(fit), len(S["test"])
out["fit_label_sets"] = len({"|".join(map(str, np.flatnonzero(r))) for r in labels("rlpool", fit)})
for sp, n in (("rlpool", 20000), ("test", 1000)):
    assert np.load(f"{ARR}/{sp}_ms12.npy", mmap_mode="r").shape == (n, 12, 120, 120) and len(json.load(open(f"{ARR}/{sp}_ids.json"))) == n
    for a in ("rgb", "fcc"): assert np.load(f"{ARR}/{sp}_{a}.npy", mmap_mode="r").shape[0] == n
assert np.load(f"{H}/langres_data/prithvi/test_prithvi_raw.npy").shape == (1000, 1024)
Yf, Yt = labels("rlpool", fit), labels("test", S["test"])
out["positives_fit"], out["positives_test"] = Yf.sum(0).tolist(), Yt.sum(0).tolist()
out["graded_classes"] = [c + 1 for c in range(19) if Yf[:, c].sum() >= 20 and Yt[:, c].sum() >= 20]
fb = np.load(f"{H}/langres_state/fb/q17.npz"); qf, qt = q17("rlpool", fit), q17("test", S["test"])
out["q17_match_fb_rlpool"] = float((qf == fb["y_rlpool"][fit]).mean())
t100 = [json.load(open(f"{ARR}/test_ids.json")).index(i) for i in json.load(open(f"{ARR}/test100_ids.json"))]
out["q17_match_fb_test100"] = float((qt[t100] == fb["y_test100"]).mean()); out["q17_yes_fit"], out["q17_yes_test"] = float(qf.mean()), float(qt.mean())
sc = json.load(open(f"{REPO}/frontier_share/schemas/caption.json")); s2 = copy.deepcopy(sc); del s2["properties"]["summaries"]
s2["properties"]["summary"] = {"type": "string"}; s2["required"] = [r if r != "summaries" else "summary" for r in sc["required"]]
json.dump(s2, open(f"{VS}/schemas/caption_vs.json", "w"), indent=1)
out["schema_sha256"] = hashlib.sha256(open(f"{VS}/schemas/caption_vs.json", "rb").read()).hexdigest()
json.dump(out, open(f"{VS}/vs0.json", "w"), indent=1); print(json.dumps(out))
