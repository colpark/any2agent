"""QV-2/QV-3 fitting. Usage: qv2.py fit SIZE [SIZE ...]  (host A GPU, one process)
For each size (practice questions, nested seed-0 subsets stratified semantic/physical) and each condition A{1,2} x B{1,2}:
base ridge(s) with alpha by leave-one-question-out AUC (negation pairs leave with their source), ZeroMLP (A2; seeds 0-2; 5 grouped folds, seed 0,
patience 5, then retrain on all), temperature by grouped-CV log loss. Output: ~/langres_state/qv/fit_{size}.npz, qcond_v2/fit_{size}.json.
Practice labels only; the replication check (size 139, A1B1) is the one grading call here."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import minimize_scalar
from qv_common import *

V = v1(); Q = V["Q"]; PIDS, HIDS = V["pids"], V["hids"]; TH_ = dict(zip(PIDS, V["theta"])); Fp, Ft = feats()
Yall, Wall, Sall = practice_targets(PIDS, Q); E0, EP, EN = emb(); RW = rewrites(); SIGN = {i: RW[i]["sign"] for i in RW}


def subset(n):
    rng = np.random.default_rng(0); sem = [i for i in PIDS if Q[i]["kind"] == "semantic"]; phy = [i for i in PIDS if Q[i]["kind"] == "physical"]
    ps, pp = rng.permutation(len(sem)), rng.permutation(len(phy)); k = int(round(n * len(sem) / len(PIDS)))
    return sorted([sem[j] for j in ps[:k]] + [phy[j] for j in pp[:n - k]], key=PIDS.index)


def ridge(X, T, a):
    return v1fit.ridge(X, T, a)


class Base:
    """B1: ridge e -> (w, b). B2: ridge e+ -> s*w, ridge [e; e_neg] -> [b; -b]. theta(q) for a question, theta_neg(q) for its negation."""
    def __init__(s, B, ids, a):
        s.B = B; T = np.array([TH_[i] for i in ids])
        if B == 1: s.M, s.c = ridge(np.array([E0[i] for i in ids]), T, a)
        else:
            sg = np.array([SIGN[i] for i in ids], float)[:, None]; s.Mw, s.cw = ridge(np.array([EP[i] for i in ids]), sg * T[:, :4096], a)
            s.Mb, s.cb = ridge(np.array([E0[i] for i in ids] + [EN[i] for i in ids]), np.r_[T[:, 4096], -T[:, 4096]][:, None], a)
    def theta(s, i, neg=False):
        if s.B == 1: return E0[i] @ s.M + s.c
        sg = SIGN[i] * (-1 if neg else 1); return np.r_[sg * (EP[i] @ s.Mw + s.cw), ((EN[i] if neg else E0[i]) @ s.Mb + s.cb)[0]]


def logits(th, F):
    return F @ th[:4096] + th[4096]


def select_alpha(B, ids):
    """LOQO over the subset: mean AUC of the predicted base on Practice vs the practice labels. Returns alpha, LOQO logits (300 x n), table."""
    col = [PIDS.index(i) for i in ids]; tab, Zs = {}, {}
    for a in ALPHAS:
        Z = np.column_stack([logits(Base(B, ids[:k] + ids[k + 1:], a).theta(i), Fp) for k, i in enumerate(ids)])
        Zs[a] = Z; tab[float(a)] = float(np.nanmean(auc_cols(Yall[:, col], Z)))
    a = max(tab, key=tab.get); return a, Zs[a], tab


def fit_T(Z, col):
    Y, W = Yall[:, col], Wall[:, col]
    return float(np.exp(minimize_scalar(lambda lt: v1fit.wll(Y.ravel(), sig((Z / np.exp(lt)).ravel()), W.ravel()), bounds=(-6, 6), method="bounded").x))


# ---------------- ZeroMLP ----------------
import torch, torch.nn as nn
DEV = "cuda"


def pca_init(X, k, seed):
    """Top-k PCA directions of X (rows); when X has fewer than k informative directions, complete with random orthonormal directions."""
    mu = X.mean(0); _, sv, Vt = np.linalg.svd(X - mu, full_matrices=False); r = int((sv > 1e-6 * sv[0]).sum()); D = Vt[:min(r, k)]
    if len(D) < k:
        R = np.random.default_rng(seed).standard_normal((k - len(D), X.shape[1])); R -= R @ D.T @ D; D = np.r_[D, np.linalg.qr(R.T)[0].T]
    return D.astype(np.float32), mu.astype(np.float32), len(D) if len(D) < k else k, r


class ZMLP(nn.Module):
    def __init__(s, Pd, Pm, Qd, Qm):
        super().__init__(); s.P, s.Q = nn.Linear(4096, 256), nn.Linear(Qd.shape[1], 256)
        with torch.no_grad():
            s.P.weight.copy_(torch.tensor(Pd)); s.P.bias.copy_(torch.tensor(-Pd @ Pm)); s.Q.weight.copy_(torch.tensor(Qd)); s.Q.bias.copy_(torch.tensor(-Qd @ Qm))
        s.mlp = nn.Sequential(nn.Linear(769, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 1))
        nn.init.zeros_(s.mlp[-1].weight); nn.init.zeros_(s.mlp[-1].bias)
    def forward(s, f, u, sg):
        a, b = s.P(f), s.Q(u); return s.mlp(torch.cat([a, b, a * b, sg[:, None]], 1))[:, 0]


def table(B, base, ids):
    """Question rows for training: (u, s, base logits on Practice, soft target column, source id). B2 adds the negation rows (u = e+, s = -s, 1 - P)."""
    rows_ = []
    for i in ids:
        c = PIDS.index(i)
        if B == 1: rows_.append((E0[i], 1., logits(base.theta(i), Fp), Sall[:, c], i))
        else:
            rows_.append((EP[i], float(SIGN[i]), logits(base.theta(i), Fp), Sall[:, c], i)); rows_.append((EP[i], -float(SIGN[i]), logits(base.theta(i, True), Fp), 1 - Sall[:, c], i))
    return rows_


def tens(rows_):
    U = torch.tensor(np.array([r[0] for r in rows_]), dtype=torch.float32, device=DEV); S = torch.tensor([r[1] for r in rows_], dtype=torch.float32, device=DEV)
    BL = torch.tensor(np.array([r[2] for r in rows_]).T, dtype=torch.float32, device=DEV); TG = torch.tensor(np.array([r[3] for r in rows_]).T, dtype=torch.float32, device=DEV)
    return U, S, BL, TG  # BL, TG: 300 x nrows


FPt = None


def run_epoch(m, opt, data, train, gen):
    U, S, BL, TG = data; n, q = BL.shape; idx = torch.arange(n * q, device=DEV)
    if train: idx = idx[torch.randperm(n * q, generator=gen, device="cpu").to(DEV)]
    m.train(train); tot = 0.
    for k in range(0, n * q, 1024):
        b = idx[k:k + 1024]; pi, qi = b // q, b % q
        with torch.set_grad_enabled(train):
            z = BL[pi, qi] + m(FPt[pi], U[qi], S[qi]); loss = nn.functional.binary_cross_entropy_with_logits(z, TG[pi, qi], reduction="sum")
        if train: opt.zero_grad(); (loss / len(b)).backward(); opt.step()
        tot += float(loss)
    return tot / (n * q)


def predict(m, F, U, S, BLt):
    m.eval()
    with torch.no_grad():
        Ft_ = torch.tensor(F, dtype=torch.float32, device=DEV); return np.column_stack([(BLt[:, j] + m(Ft_, U[j].expand(len(F), -1), S[j].expand(len(F))).cpu().numpy()) for j in range(len(U))])


def make(B, ids, seed):
    torch.manual_seed(seed); Pd, Pm, _, _ = pca_init(Fp, 256, seed); us = np.array([(E0 if B == 1 else EP)[i] for i in ids]); Qd, Qm, _, r = pca_init(us, 256, seed)
    m = ZMLP(Pd, Pm, Qd, Qm).to(DEV); return m, torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2), r


def zero_mlp(B, ids, a, base_all):
    """Grouped 5-fold early stopping (folds by question, seed 0), then retrain on all ids for the best epoch count; seeds 0-2.
    Returns OOF logits (300 x n original questions, seed mean), per-seed epochs, the final models."""
    global FPt
    FPt = torch.tensor(Fp, dtype=torch.float32, device=DEV); fold = np.random.default_rng(0).permutation(len(ids)) % 5; oof = np.zeros((3, 300, len(ids))); eps, finals = [], []
    bases = [Base(B, [i for k, i in enumerate(ids) if fold[k] != f], a) for f in range(5)]
    for seed in (0, 1, 2):
        gen = torch.Generator().manual_seed(seed); runs = []
        for f in range(5):
            tr = [i for k, i in enumerate(ids) if fold[k] != f]; va = [i for k, i in enumerate(ids) if fold[k] == f]
            m, opt, _ = make(B, tr, seed); runs.append((m, opt, tens(table(B, bases[f], tr)), tens(table(B, bases[f], va)), va))
        best, best_ep, wait, hist, snap = np.inf, 0, 0, [], {}
        for ep in range(1, 51):
            vl = []
            for m, opt, dtr, dva, va in runs: run_epoch(m, opt, dtr, True, gen); vl.append(run_epoch(m, None, dva, False, gen))
            hist.append(float(np.mean(vl)))
            if hist[-1] < best - 1e-9:
                best, best_ep, wait = hist[-1], ep, 0
                for f, (m, _, _, dva, va) in enumerate(runs):  # out-of-fold logits on original questions at this epoch
                    U, S, BL, _ = dva; keep = slice(None) if B == 1 else slice(0, None, 2); m.eval()
                    with torch.no_grad(): snap[f] = predict(m, Fp, U[keep], S[keep], BL[:, keep].cpu().numpy())
            else:
                wait += 1
                if wait >= 5: break
        for f, (_, _, _, _, va) in enumerate(runs): oof[seed][:, [ids.index(i) for i in va]] = snap[f]
        m, opt, r = make(B, ids, seed); d = tens(table(B, base_all, ids))
        for _ in range(best_ep): run_epoch(m, opt, d, True, gen)
        eps.append({"seed": seed, "best_epoch": best_ep, "val_curve": hist, "q_rank": r}); finals.append(m)
    return oof.mean(0), eps, finals


def heldout_base(base, F):
    return np.column_stack([logits(base.theta(h), F) for h in HIDS])


def fit_size(n):
    ids = PIDS if n == len(PIDS) else subset(n); col = [PIDS.index(i) for i in ids]; out, rep = {}, {"size": n, "ids": ids}
    for B in (1, 2):
        t0 = time.time(); a, Zloqo, tab = select_alpha(B, ids); base = Base(B, ids, a); bt = heldout_base(base, Ft)
        T1 = fit_T(Zloqo, col); out[f"A1B{B}"] = sig(bt / T1)
        rep[f"B{B}"] = {"alpha": a, "loqo_auc_mean": tab[a], "loqo_table": tab, "loqo_auc": dict(zip(ids, auc_cols(Yall[:, col], Zloqo).tolist())), "sec": round(time.time() - t0)}
        rep[f"A1B{B}"] = {"T": T1, "T_method": "leave-one-question-out (as v1)"}
        t0 = time.time(); oof, eps, finals = zero_mlp(B, ids, a, base); T2 = fit_T(oof, col)
        U = torch.tensor(np.array([(E0 if B == 1 else EP)[h] for h in HIDS]), dtype=torch.float32, device=DEV)
        S = torch.tensor([1. if B == 1 else float(SIGN[h]) for h in HIDS], dtype=torch.float32, device=DEV)
        zt = np.mean([predict(m, Ft, U, S, bt) for m in finals], 0); out[f"A2B{B}"] = sig(zt / T2)
        rep[f"A2B{B}"] = {"T": T2, "T_method": "grouped 5-fold OOF (seed mean)", "seeds": eps, "oof_auc_mean": float(np.nanmean(auc_cols(Yall[:, col], oof))), "sec": round(time.time() - t0)}
        out[f"base_B{B}"], out[f"logit_A2B{B}"] = bt, zt
        print(n, B, json.dumps({k: (v if k != "seeds" else [(e["best_epoch"], e["q_rank"]) for e in v]) for k, v in rep[f"A2B{B}"].items()}), "alpha", a, "T1", round(T1, 3), flush=True)
    np.savez(f"{SQV}/fit_{n}.npz", **out); json.dump(rep, open(f"{QV}/fit_{n}.json", "w"), indent=1)
    if n == len(PIDS):  # replication check (grading)
        Y = truth23("test", rows()["test"]); m = float(np.nanmean(auc_cols(Y, out["A1B1"][:, :len(SEM)]))); v = float(np.nanmean(auc_cols(Y, V["Z"]["qc"][:, :len(SEM)])))
        chk = {"A1B1_mean_sem_auc": m, "v1_stored": v, "abs_diff": abs(m - v), "pass": abs(m - v) <= .005, "max_abs_prob_diff_vs_v1": float(np.abs(out["A1B1"] - V["Z"]["qc"]).max())}
        json.dump(chk, open(f"{QV}/replication_check.json", "w"), indent=1); print("REPLICATION", chk, flush=True)
        if not chk["pass"]: sys.exit("replication failed: log a QV-E entry and stop")


if __name__ == "__main__":
    assert sys.argv[1] == "fit"
    for n in map(int, sys.argv[2:]): fit_size(n)
