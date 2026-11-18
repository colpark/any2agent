"""FS-1: 16 logistic probes on cached, RL-pool-standardized Prithvi features (C=1.0, max_iter=3000).
5-fold CV accuracy on the RL pool, then fit on the full RL pool and predict the 10 selected test100 patches."""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from joblib import Parallel, delayed
H = os.path.expanduser("~"); ST = f"{H}/langres_state/fs"; FS = f"{H}/Documents/langres/frontier_share"
Xr = np.load(f"{H}/langres_data/prithvi/rlpool_prithvi.npy"); Xt = np.load(f"{H}/langres_data/prithvi/test100_prithvi.npy")
assert json.load(open(f"{ST}/keys_rlpool_ids.json")) == json.load(open(f"{H}/langres_data/arrays/rlpool_ids.json"))
assert json.load(open(f"{ST}/keys_test100_ids.json")) == json.load(open(f"{H}/langres_data/arrays/test100_ids.json"))
Kr = np.load(f"{ST}/keys_rlpool.npz")["keys"]; Kt = np.load(f"{ST}/keys_test100.npz")["keys"]
sel = json.load(open(f"{ST}/selection.json"))
Q = [q["id"] for q in json.load(open(f"{FS}/questions.json"))]

def one(k):
    y = Kr[:, k]; clf = LogisticRegression(C=1.0, max_iter=3000)
    cv = cross_val_score(clf, Xr, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="accuracy")
    clf.fit(Xr, y); p = clf.predict(Xt)
    return {"id": Q[k], "cv_acc": float(cv.mean()), "cv_folds": cv.tolist(), "majority_acc": float(max(y.mean(), 1 - y.mean())),
            "test100_acc": float((p == Kt[:, k]).mean()), "n_iter": int(clf.n_iter_[0]),
            "pred_selected": {u: bool(p[j]) for u, j in zip(sel["uid_order"], sel["test100_index"])}}

res = Parallel(n_jobs=8)(delayed(one)(k) for k in range(16))
json.dump(res, open(f"{FS}/probes.json", "w"), indent=1)
for r in res: print(r["id"], round(r["cv_acc"], 3), round(r["majority_acc"], 3), round(r["test100_acc"], 2), r["n_iter"])
