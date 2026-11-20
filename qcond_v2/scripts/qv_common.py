"""Shared QV helpers (DESIGN_QV_v1.md): v1 reuse (qcond/scripts), rewrites, embeddings, column AUC. True labels: grading code only."""
import json, os, sys
import numpy as np
H = os.path.expanduser("~"); REPO = f"{H}/Documents/langres"; QV = f"{REPO}/qcond_v2"; SQV = f"{H}/langres_state/qv"
sys.path.insert(0, f"{REPO}/qcond/scripts")
from qc_common import *  # noqa: F401,F403  (QC, ST = ~/langres_state/qc, rows, feats, derive, truth23, SEM, sig, auc, boot_idx, ...)
import qc3 as v1fit, qc4 as v1meas  # noqa: E402
os.makedirs(SQV, exist_ok=True)
ALPHAS = np.logspace(-3, 3, 13)


def v1():
    """The v1 objects reused as stored: practice ids (139, answerer order), theta (139 x 4097), C, held-out ids, comparison scores."""
    J = json.load(open(f"{QC}/qc3.json")); Z = np.load(f"{ST}/qc3.npz"); Q = {q["id"]: q for q in questions()}
    return {"pids": J["practice_ids"], "hids": J["held_ids"], "theta": Z["theta"], "C": float(Z["C"]), "Q": Q, "Z": Z, "J": J}


def practice_targets(pids, Q):
    """Binary label, weight (v1 practice_labels) and soft target (Astra soft P or code key) per practice question, 300 x n."""
    allp = [q for q in questions() if q["split"] == "practice"]; Ya, Wa = v1fit.practice_labels(allp)  # QV-E1: v1 indexes soft columns within the list given; pass all 141 as v1 did
    pos = [[q["id"] for q in allp].index(i) for i in pids]; Y, W = Ya[:, pos], Wa[:, pos]
    P = np.load(f"{ST}/soft_practice.npy"); sem = [q["id"] for q in questions() if q["split"] == "practice" and q["kind"] == "semantic"]
    S = np.column_stack([P[:, sem.index(i)] if Q[i]["kind"] == "semantic" else Y[:, k].astype(float) for k, i in enumerate(pids)])
    return Y, W, S


def rewrites():
    return {r["id"]: r for r in json.load(open(f"{QV}/rewrites.json"))}


def emb():
    """Original (v1), positive-form and negation embeddings by question id."""
    E = np.load(f"{ST}/emb_all.npy"); Q = {q["id"]: q for q in questions()}; R = np.load(f"{SQV}/emb_rewrites.npz"); ids = list(R["ids"])
    return ({i: E[q["emb_row"]] for i, q in Q.items()}, {i: R["pos"][k] for k, i in enumerate(ids)}, {i: R["neg"][k] for k, i in enumerate(ids)})


def auc_cols(Y, S):
    """AUC per column (ties averaged); nan where a column has one class."""
    from scipy.stats import rankdata
    Y = np.asarray(Y, bool); r = rankdata(S, axis=0); n1 = Y.sum(0); n0 = len(Y) - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        a = ((r * Y).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return np.where((n1 > 0) & (n0 > 0), a, np.nan)
