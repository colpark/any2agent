"""Shared AH helpers (DESIGN_AH_v1.md): paths, sets, classes, inputs, sources, Codex call (stage 1 wrapper), Dawid-Skene, heads, AUC
bootstrap. True labels: truth() only, called by grading code."""
import json, os, subprocess, sys, types
import numpy as np
H = os.path.expanduser("~"); REPO = f"{H}/Documents/langres"; AH = f"{REPO}/agent_heads"; ST = f"{H}/langres_state/ah"
TH, THS, VSS, ARR = f"{REPO}/teacher_heads", f"{H}/langres_state/th", f"{H}/langres_state/vs", f"{H}/langres_data/arrays"
for p in ("teacher_heads/scripts", "verbal_share/scripts"): sys.path.insert(0, f"{REPO}/{p}")
os.makedirs(ST, exist_ok=True)
CL = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 19]; LEDGER = f"{AH}/calls.jsonl"; CAP = 4
SCORE = {("yes", "high"): .9, ("yes", "medium"): .75, ("yes", "low"): .6, ("no", "low"): .4, ("no", "medium"): .25, ("no", "high"): .1}
LAYERS_F = [5, 11, 17, 23]  # blocks 6, 12, 18, 24 (as VS)


def sets():
    """bank_rl: the 500 stage 1 bank rows (RL pool). test: the 997 VS Test rows (indices into the stage 1 Test). t200: positions in test."""
    th = json.load(open(f"{TH}/sets.json")); test = np.flatnonzero(np.load(f"{VSS}/ok_test.npy")); pos = {int(t): k for k, t in enumerate(test)}
    return {"bank_rl": th["bank_rl"], "bank_ids": th["bank_ids"], "test": test.tolist(), "t200": [pos[t] for t in th["test200"] if t in pos],
            "t200_test": [t for t in th["test200"] if t in pos]}


def bank_ok():  # caption failed twice -> NaN embeddings -> patch dropped from the bank (logged)
    return ~np.isnan(np.load(f"{ST}/emb_bank.npy")).any((1, 2))


def classes():
    from th_common import CLASSES
    d = {x["id"]: x["definition"] for x in json.load(open(f"{TH}/code_teacher.json"))["definitions"]}
    return [{"id": c, "name": CLASSES[c - 1], "definition": d[c]} for c in CL]


def truth(split, idx):  # grading and the true-label reference only
    from th_common import CLASSES
    import pandas as pd
    ids = json.load(open(f"{ARR}/{split}_ids.json")); d = pd.read_parquet(f"{ARR}/descriptors.parquet").set_index("patch_id")
    return np.array([[CLASSES[c - 1] in d.loc[ids[i], "labels"] for c in CL] for i in idx], bool)


def inputs():
    """F (4,096), VC (17,920), D (10) for bank (ok rows) and Test (997), F and D standardized with bank statistics."""
    S = sets(); ok = bank_ok(); t = S["test"]
    Lb, Lt = np.load(f"{ST}/layers_bank.npy")[ok][:, LAYERS_F], np.load(f"{VSS}/layers_test.npy")[t][:, LAYERS_F]
    mu, sd = Lb.mean(0), Lb.std(0) + 1e-6; z = lambda L: ((L - mu) / sd).reshape(len(L), -1)
    Db, Dt = np.load(f"{ST}/desc_bank.npy")[ok], np.load(f"{ST}/desc_test.npy"); dm, ds = Db.mean(0), Db.std(0) + 1e-12
    Eb, Et = np.load(f"{ST}/emb_bank.npy")[ok], np.load(f"{VSS}/emb_test.npy")[t]
    return {"F": (z(Lb), z(Lt)), "VC": (Eb.reshape(len(Eb), -1), Et.reshape(len(Et), -1)), "D": ((Db - dm) / ds, (Dt - dm) / ds)}


def _rows(path, key="split_index"):
    return {r[key]: r for r in map(json.loads, open(path))}


def qwen(tag, idx):
    """Stage 1 Qwen3.5-9B answers: (ans int8 1/0/-1, score float) on the 15 classes for stage 1 split rows idx."""
    R = _rows(f"{THS}/qwen_{tag}.jsonl"); A = np.full((len(idx), len(CL)), -1, np.int8); P = np.full(A.shape, .5)
    for k, i in enumerate(idx):
        r = R.get(int(i)) or {}
        for j, c in enumerate(CL):
            a, cf = (r.get("answers") or {}).get(str(c)), (r.get("confidence") or {}).get(str(c))
            if a in ("yes", "no"): A[k, j] = a == "yes"; P[k, j] = SCORE.get((a, cf), .5)
    return A, P


def astra(tag, idx):
    """Stage 1 Astra answers with confidence, read from the raw call outputs (ok calls in teacher_heads/calls.jsonl)."""
    got = {}
    for r in map(json.loads, open(f"{TH}/calls.jsonl")):
        if r.get("set") != tag or r["status"] != "ok": continue
        obj = json.load(open(f"{TH}/{r['raw_dir']}/last_message.json"))
        for i, p in zip(r["patch_index"], sorted(obj["patches"], key=lambda p: p["pid"])): got[int(i)] = {a["id"]: (a["answer"], a["confidence"]) for a in p["answers"]}
    A = np.full((len(idx), len(CL)), -1, np.int8); P = np.full(A.shape, .5)
    for k, i in enumerate(idx):
        for j, c in enumerate(CL):
            if int(i) in got and c in got[int(i)]: a = got[int(i)][c]; A[k, j] = a[0] == "yes"; P[k, j] = SCORE[a]
    return A, P


def codex(name, prompt, ws, schema, sandbox="read-only", timeout=2700):
    """The stage 1 wrapper th_common.codex_call (agent settings, venv first on PATH, TH-E2 audit) with raw output under agent_heads/raw.
    sandbox="workspace-write" swaps only the -s value and adds network off. A file_change item counts as a flag unless every path lies in ws."""
    import th_common as thc
    thc.TH = AH; real = subprocess.run; used_cmd = []

    def run(cmd, **kw):
        if isinstance(cmd, list) and cmd[:1] == ["timeout"] and sandbox != "read-only":
            i = cmd.index("-s"); cmd = cmd[:i] + ["-s", sandbox, "-c", "sandbox_workspace_write.network_access=false"] + cmd[i + 2:]
        used_cmd.append(cmd); return real(cmd, **kw)
    thc.subprocess = types.SimpleNamespace(run=run)
    try:
        rec, out = thc.codex_call(name, prompt, ws, schema, timeout=timeout)
    finally:
        thc.subprocess = subprocess
    rec["cmd"] = " ".join(used_cmd[0]); rec["sandbox"] = sandbox
    ev = [json.loads(l) for l in open(f"{out}/events.jsonl") if l.startswith("{")]
    fc = [ch.get("path", "") for e in ev if e.get("item", {}).get("type") == "file_change" for ch in e["item"].get("changes", [])]
    rec["file_change_paths"] = sorted(set(fc))
    if sandbox != "read-only" and fc and all(os.path.realpath(p if os.path.isabs(p) else f"{ws}/{p}").startswith(os.path.realpath(ws) + "/") for p in fc):
        rec["audit_flags"] = [f for f in rec["audit_flags"] if tuple(f) != ("tool", "file_change", "")]
    return rec, out


def used():
    return sum(1 for _ in open(LEDGER)) if os.path.exists(LEDGER) else 0


def dawid_skene(L, iters=200, tol=1e-6):
    """L: n x S in {1, 0, -1 abstain}. Binary DS with Beta(2,2)-smoothed sensitivity/specificity per source and a class prior, EM from the
    majority vote. Returns P(yes), params, n_iter, fallback flag (prior outside [0.01, 0.99] -> majority vote)."""
    v, y = L >= 0, L == 1; nv = v.sum(1); mv = np.where(nv > 0, y.sum(1) / np.maximum(nv, 1), .5)
    mv = np.where(mv > .5, 1., np.where(mv < .5, 0., .5)); q = mv.copy(); old = None
    for it in range(1, iters + 1):
        pi = q.mean(); se = ((q[:, None] * (y & v)).sum(0) + 1) / ((q[:, None] * v).sum(0) + 2)
        sp = (((1 - q)[:, None] * (~y & v)).sum(0) + 1) / (((1 - q)[:, None] * v).sum(0) + 2)
        par = np.r_[pi, se, sp]
        if old is not None and np.abs(par - old).max() <= tol: break
        old = par; pi_c = np.clip(pi, 1e-9, 1 - 1e-9)
        lo = np.log(pi_c / (1 - pi_c)) + ((y & v) * np.log(se / (1 - sp)) + (~y & v) * np.log((1 - se) / sp)).sum(1); q = 1 / (1 + np.exp(-lo))
    fb = not (.01 <= pi <= .99)
    return (mv if fb else q), {"prior": float(pi), "sens": se.tolist(), "spec": sp.tolist()}, it, fb


def head(Xb, y, Xt, w=None, cv=True):
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler().fit(Xb)
    m = LogisticRegressionCV(Cs=[0.01, 0.1, 1], cv=5, scoring="neg_log_loss", max_iter=3000) if cv else LogisticRegression(C=0.1, max_iter=3000)
    return m.fit(s.transform(Xb), y, sample_weight=w).predict_proba(s.transform(Xt))[:, 1]


def auc(y, p):  # rank AUC with ties; nan if one class
    from scipy.stats import rankdata
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else float((rankdata(p)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_idx(n, B=1000, seed=0):
    rng = np.random.default_rng(seed); return [rng.integers(0, n, n) for _ in range(B)]
