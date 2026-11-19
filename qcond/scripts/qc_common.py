"""Shared QC helpers (DESIGN_QC_v1.md): paths, sets, features, question keys, Astra calls (stage 1 wrapper, image-QA settings), fits.
True labels: truth23() only, called by grading code (qc3 grade / qc4)."""
import json, os, sys, threading
import numpy as np
H = os.path.expanduser("~"); REPO = f"{H}/Documents/langres"; QC = f"{REPO}/qcond"; ST = f"{H}/langres_state/qc"
sys.path.insert(0, f"{REPO}/agent_heads/scripts")
import ah_common as ah
from ah_common import CL, LAYERS_F, auc, boot_idx, VSS, ARR, TH
import th_common as thc
os.makedirs(ST, exist_ok=True)
LEDGER, CAP = f"{QC}/calls.jsonl", 140
SOFT = {("yes", "high"): .97, ("yes", "medium"): .80, ("yes", "low"): .57, ("no", "low"): .28, ("no", "medium"): .10, ("no", "high"): .015}
DIRECT = ah.SCORE  # yes high 0.9 ... no high 0.1
GQ = [("G1", "Is any forest present?"), ("G2", "Is any farmland present?"), ("G3", "Is any built-up land present?"),
      ("G4", "Is any open water present?"), ("G5", "Is any wetland present?"),
      ("G6", "Is any natural open land present, such as grassland, heath, shrub or sand?"),
      ("G7", "Does the patch mix farmland with forest, wetland or natural open land?"), ("G8", "Does the patch contain more than three land cover types?")]
GM = {"G1": (9, 10, 11), "G2": (3, 4, 5, 6, 7, 8), "G3": (1, 2), "G4": (18, 19), "G5": (16, 17), "G6": (12, 13, 14, 15)}
SEM = [f"C{c}" for c in CL] + [g for g, _ in GQ]  # the 23 held-out semantic question ids


def practice():
    """Positions (into the 497 AH bank rows, file order) of the 300 Practice patches: seed 0 draw, sorted."""
    return np.sort(np.random.default_rng(0).choice(int(ah.bank_ok().sum()), 300, replace=False))


def rows():
    """Stage 1 split rows: practice (RL pool), test (997 stage 1 Test rows), t200 (positions in test)."""
    S = ah.sets(); bank = np.array(S["bank_rl"])[ah.bank_ok()]
    return {"practice": bank[practice()], "test": np.array(S["test"]), "t200": np.array(S["t200"]), "t200_test": np.array(S["t200_test"])}


def feats():
    """F on Practice and Test: blocks 6, 12, 18, 24 mean-pooled, concatenated, standardized with Practice statistics."""
    Lb = np.load(f"{ah.ST}/layers_bank.npy")[ah.bank_ok()][practice()][:, LAYERS_F]; Lt = np.load(f"{VSS}/layers_test.npy")[ah.sets()["test"]][:, LAYERS_F]
    mu, sd = Lb.mean(0), Lb.std(0) + 1e-6; z = lambda L: ((L - mu) / sd).reshape(len(L), -1)
    return z(Lb), z(Lt)


def derive(P):
    """P: n x 19 scores (or 0/1) over classes 1..19 -> n x 23 (15 classes, G1..G8) by the stage 1 key rules. On scores: OR = max,
    AND = min, 'more than three' = 4th largest; on 0/1 input this is exactly the stage 1 rule."""
    g = lambda *c: P[:, [x - 1 for x in c]].max(1); G = {k: g(*v) for k, v in GM.items()}
    G7 = np.minimum(G["G2"], np.maximum.reduce([G["G1"], G["G5"], G["G6"]])); G8 = -np.sort(-P, 1)[:, 3]
    return np.column_stack([P[:, [c - 1 for c in CL]], G["G1"], G["G2"], G["G3"], G["G4"], G["G5"], G["G6"], G7, G8])


def truth23(split, idx):  # grading and true-label heads only
    ids = json.load(open(f"{ARR}/{split}_ids.json")); return derive(thc.true_labels([ids[i] for i in idx]).astype(float)) > .5


def astra19(tag, idx, sc):
    """Stage 1 Astra answers (19 classes) for split rows idx from the ok raw calls, scored by sc[(answer, confidence)]; NaN if missing."""
    got = {}
    for r in map(json.loads, open(f"{TH}/calls.jsonl")):
        if r.get("set") != tag or r["status"] != "ok": continue
        obj = json.load(open(f"{TH}/{r['raw_dir']}/last_message.json"))
        for i, p in zip(r["patch_index"], sorted(obj["patches"], key=lambda p: p["pid"])): got[int(i)] = {a["id"]: sc[(a["answer"], a["confidence"])] for a in p["answers"]}
    return np.array([[got.get(int(i), {}).get(c, np.nan) for c in range(1, 20)] for i in idx])


def qwen19(idx, sc):
    R = {r["split_index"]: r for r in map(json.loads, open(f"{ah.THS}/qwen_test.jsonl"))}
    return np.array([[sc.get((R[int(i)]["answers"].get(str(c)), R[int(i)]["confidence"].get(str(c))), .5) for c in range(1, 20)] for i in idx])


LOCK, PENDING, STOP = threading.Lock(), set(), threading.Event()


def used():
    return sum(1 for _ in open(LEDGER)) if os.path.exists(LEDGER) else 0


def call(tag, prompt, schema_path, images, validate, extra=()):
    """One Astra call with the stage 1 image-QA settings (agent=False: any tool voids it) in a fresh empty folder. Ledger + cap 140."""
    import tempfile
    with LOCK:
        n = used() + len(PENDING)
        if n >= CAP or STOP.is_set(): return None, None
        cid = f"q{n + 1:03d}"; PENDING.add(cid)
    ws = tempfile.mkdtemp(dir=ST, prefix=f"ws_{cid}_"); thc.TH = QC
    if extra:  # QC-X1: extra codex flags after "exec" (question call only)
        import subprocess, types
        thc.subprocess = types.SimpleNamespace(run=lambda cmd, **kw: subprocess.run(cmd[:3] + list(extra) + cmd[3:], **kw))
    try:
        rec, out = thc.codex_call(f"{cid}_{tag}", prompt, ws, schema_path, images=images, agent=False)
    finally:
        import subprocess; thc.subprocess = subprocess
    rec["extra_flags"] = list(extra)
    try:
        obj = json.load(open(f"{out}/last_message.json")); errs = validate(obj)
    except Exception as e:
        obj, errs = None, [f"parse: {e}"]
    rec.update({"call_id": cid, "tag": tag, "errors": errs[:5], "workdir_empty_after": os.listdir(ws) == []})
    rec["status"] = "usage_limit" if rec["usage_limit"] else "voided_tool" if rec["audit_flags"] else "invalid" if errs else "ok"
    if rec["usage_limit"]: STOP.set()
    with LOCK:
        open(LEDGER, "a").write(json.dumps(rec) + "\n"); PENDING.discard(cid)
    print(json.dumps({x: rec[x] for x in ("call_id", "tag", "status", "latency_s", "errors")}), flush=True)
    return rec, obj


def questions():
    return json.load(open(f"{QC}/questions.json"))


def logit_fit(X, y, w, C):
    """Logistic answerer on F (already standardized): returns (coef 4096, intercept)."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=5000).fit(X, y, sample_weight=w); return m.coef_[0], m.intercept_[0]


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))
