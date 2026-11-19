"""VS-2 (host A, CPU, no labels): maps from text (V, VC) to Prithvi F; verbal share, geometry check, layer curve, field table,
learning curve, shuffle sanity check, residuals. Outputs verbal_share/vs2.json, ~/langres_state/vs/{fhat,resid}_{fit,test}.npy."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from vs_common import *
KF = KFold(5, shuffle=True, random_state=0)


def sqd(A, B):  # squared Euclidean distances through BLAS (the scipy routine is single-threaded)
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T, 0)


def load():
    e = os.environ.get("VS_EMB", "emb"); Ef, Et = np.load(f"{ST}/{e}_fit.npy"), np.load(f"{ST}/{e}_test.npy")  # VS_EMB: dry-run inputs only
    ok_f, ok_t = ~np.isnan(Ef).any((1, 2)), ~np.isnan(Et).any((1, 2))
    Lf, Lt = np.load(f"{ST}/layers_fit.npy")[ok_f], np.load(f"{ST}/layers_test.npy")[ok_t]
    mu, sd = Lf.mean(0), Lf.std(0) + 1e-6; Lf, Lt = (Lf - mu) / sd, (Lt - mu) / sd  # per layer, Fit statistics
    return Ef[ok_f], Et[ok_t], Lf, Lt, ok_f, ok_t


def ridge(): return RidgeCV(alphas=np.logspace(-1, 5, 13), alpha_per_target=True)


def krr_cv(X, Y):
    """gamma in {.25,.5,1,2,4}/median sq. distance, alpha in logspace(-3,1,5); one pair for all targets by 5-fold CV R2 (eigen shortcut,
    identical to KernelRidge's closed form)."""
    m = np.median(sqd(X, X)[np.triu_indices(len(X), 1)]); best = (-np.inf, None)
    G, A = [g / m for g in (0.25, 0.5, 1, 2, 4)], np.logspace(-3, 1, 5); sc = np.zeros((5, 5))
    for tr, va in KF.split(X):
        mu = Y[tr].mean(0); Dtt, Dvt = sqd(X[tr], X[tr]), sqd(X[va], X[tr])
        for i, g in enumerate(G):
            w, U = np.linalg.eigh(np.exp(-g * Dtt)); Kv = np.exp(-g * Dvt); UtY = U.T @ Y[tr]
            for j, a in enumerate(A): sc[i, j] += r2w(Y[va], Kv @ (U @ (UtY / (w + a)[:, None])), mu) / 5
    i, j = np.unravel_index(sc.argmax(), sc.shape); return KernelRidge(kernel="rbf", gamma=G[i], alpha=A[j]), float(sc.max())


def knn_cv(X, Y):
    sc = {}
    for k in (5, 10, 20, 50):
        sc[k] = np.mean([r2w(Y[va], KNeighborsRegressor(k, metric="cosine", weights="distance").fit(X[tr], Y[tr]).predict(X[va]), Y[tr].mean(0)) for tr, va in KF.split(X)])
    k = max(sc, key=sc.get); return KNeighborsRegressor(k, metric="cosine", weights="distance"), float(sc[k])


def cv_r2(model, X, Y): return float(np.mean([r2w(Y[va], model.fit(X[tr], Y[tr]).predict(X[va]), Y[tr].mean(0)) for tr, va in KF.split(X)]))


def fit_family(fam, X, Y):
    if fam == "linear": m = ridge(); return m, cv_r2(m, X, Y)
    return krr_cv(X, Y) if fam == "kernel" else knn_cv(X, Y)


def share(fam, Xf, Yf, Xt, Yt, ci=False):
    m, cv = fit_family(fam, Xf, Yf); P = m.fit(Xf, Yf).predict(Xt); mu = Yf.mean(0)
    out = {"test_r2": float(r2w(Yt, P, mu)), "cv_r2_fit": cv, "params": {k: v for k, v in m.get_params().items() if k in ("gamma", "alpha", "n_neighbors")}}
    if ci: out["ci"] = boot_r2(Yt, P, mu)
    return out, m, P


def cka(K, L):
    n = len(K); Hm = np.eye(n) - 1 / n; Kc, Lc = Hm @ K @ Hm, Hm @ L @ Hm
    return float((Kc * Lc).sum() / np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum()))


def rbf_gram(X):
    D = sqd(X, X); return np.exp(-D / (2 * np.median(D[np.triu_indices(len(X), 1)])))


def mknn(X, Y, k=10):
    a = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X).kneighbors(X, return_distance=False)[:, 1:]
    b = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    return float(np.mean([len(set(p) & set(q)) / k for p, q in zip(a, b)]))


def main():
    t0 = time.time(); Ef, Et, Lf, Lt, ok_f, ok_t = load(); out = {"n_fit": int(ok_f.sum()), "n_test": int(ok_t.sum())}
    T = lambda E, idx: E[:, idx].reshape(len(E), -1); V, VC = list(range(6)), list(range(7))
    F_f, F_t = Lf[:, LAYERS_F].reshape(len(Lf), -1), Lt[:, LAYERS_F].reshape(len(Lt), -1)
    # H1: all three maps, V and VC
    out["maps"] = {}
    for rep, idx in (("V", V), ("VC", VC)):
        out["maps"][rep] = {fam: share(fam, T(Ef, idx), F_f, T(Et, idx), F_t, ci=True)[0] for fam in ("linear", "kernel", "knn")}
        print(rep, {f: (round(v["cv_r2_fit"], 3), round(v["test_r2"], 3)) for f, v in out["maps"][rep].items()}, round(time.time() - t0), "s", flush=True)
    fam = max(out["maps"]["VC"], key=lambda f: out["maps"]["VC"][f]["cv_r2_fit"]); out["selected"] = fam
    out["gap_best_nonlinear_minus_linear"] = {r: max(out["maps"][r]["kernel"]["test_r2"], out["maps"][r]["knn"]["test_r2"]) - out["maps"][r]["linear"]["test_r2"] for r in ("V", "VC")}
    # sanity: VC rows shuffled across Fit patches
    perm = np.random.default_rng(0).permutation(len(Ef))
    out["shuffle_sanity"] = {f: share(f, T(Ef, VC)[perm], F_f, T(Et, VC), F_t)[0]["test_r2"] for f in sorted({fam, "linear"})}
    # residuals from the selected VC map: cross-fitted on Fit, full-Fit map on Test
    m, _ = fit_family(fam, T(Ef, VC), F_f); Fh_f = np.zeros_like(F_f)
    for tr, va in KF.split(F_f): Fh_f[va] = m.fit(T(Ef, VC)[tr], F_f[tr]).predict(T(Ef, VC)[va])
    Fh_t = m.fit(T(Ef, VC), F_f).predict(T(Et, VC))
    for n_, a in (("fhat_fit", Fh_f), ("fhat_test", Fh_t), ("resid_fit", F_f - Fh_f), ("resid_test", F_t - Fh_t)): np.save(f"{ST}/{n_}.npy", a.astype(np.float32))
    np.save(f"{ST}/ok_fit.npy", ok_f); np.save(f"{ST}/ok_test.npy", ok_t)
    out["crossfit_r2_fit"] = float(r2w(F_f, Fh_f, F_f.mean(0)))
    # geometry check on Test: VC space vs each layer and F
    Xvc = T(Et, VC); sh = Xvc[np.random.default_rng(0).permutation(len(Xvc))]; Kl, Kr = Xvc @ Xvc.T, rbf_gram(Xvc); Ksl, Ksr = sh @ sh.T, rbf_gram(sh)
    out["geometry"] = {}
    for name, Y in [(f"L{l + 1}", Lt[:, l]) for l in range(24)] + [("F", F_t)]:
        Gl, Gr = Y @ Y.T, rbf_gram(Y)
        out["geometry"][name] = {"mknn10": mknn(Xvc, Y), "mknn10_shuf": mknn(sh, Y), "cka_lin": cka(Kl, Gl), "cka_lin_shuf": cka(Ksl, Gl),
                                 "cka_rbf": cka(Kr, Gr), "cka_rbf_shuf": cka(Ksr, Gr)}
    print("geometry F", out["geometry"]["F"], round(time.time() - t0), "s", flush=True)
    fams = sorted({fam, "linear"})
    out["layer_curve"] = {rep: {f: [share(f, T(Ef, idx), Lf[:, l], T(Et, idx), Lt[:, l])[0]["test_r2"] for l in range(24)] for f in fams} for rep, idx in (("V", V), ("VC", VC))}
    print("layer curve done", round(time.time() - t0), "s", flush=True)
    out["field_table"] = {FIELDS[j]: {f: share(f, Ef[:, j], F_f, Et[:, j], F_t, ci=True)[0] for f in fams} for j in range(7)}
    sub = np.random.default_rng(0).permutation(len(Ef))  # nested subsets: first n of one permutation
    out["learning_curve"] = {str(n): {f: share(f, T(Ef, VC)[sub[:min(n, len(Ef))]], F_f[sub[:min(n, len(Ef))]], T(Et, VC), F_t)[0]["test_r2"] for f in fams} for n in (250, 500, 1000, 2000)}
    out["sec"] = round(time.time() - t0); json.dump(out, open(f"{VS}/vs2.json", "w"), indent=1); print("DONE", out["sec"], "s")


if __name__ == "__main__":
    main()
