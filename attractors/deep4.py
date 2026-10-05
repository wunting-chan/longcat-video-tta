import numpy as np, json
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
import deep as Dp, analyze as AZ
D, names = Dp.load(); FPS = 16; out = {}
R = json.load(open("results_long/relax.json"))
# 1) local stability proxy: same prompt, different noise (from t=0); growth of distance over time
print("1) seed-pair divergence (same prompt, different noise)")
for m, runs in D.items():
    P = sorted({k[0] for k in runs}); curves = []
    for p in P:
        a = Dp.per_second(runs[(p, 0)]["cls"]); b = Dp.per_second(runs[(p, 1)]["cls"])
        a /= np.linalg.norm(a, axis=1, keepdims=True); b /= np.linalg.norm(b, axis=1, keepdims=True)
        curves.append(1 - (a * b).sum(1))
    C = np.array([c[:min(map(len, curves))] for c in curves])
    early = np.polyfit(np.arange(5, 40), np.log(C[:, 5:40].mean(0)), 1)[0]
    out.setdefault("seed_pair", {})[m] = {"d_5s": float(C[:, 5].mean()), "d_60s": float(C[:, 60].mean()), "d_end": float(C[:, -10:].mean()),
                                          "log_growth_per_s_5_40": float(early)}
    print(f"   {m}: d(5s) {C[:,5].mean():.3f}  d(60s) {C[:,60].mean():.3f}  d(end) {C[:,-10:].mean():.3f}  log-growth 5-40 s {early:+.4f}/s")
# 2) is relaxation time a property of the prompt? (SF seed pairs)
tau = {k: t for k, t in zip(sorted(D["sf"]), R["sf"]["tau_s"])}
pairs = np.array([(tau[(p, 0)], tau[(p, 1)]) for p in sorted({k[0] for k in tau})])
r, p = spearmanr(pairs[:, 0], pairs[:, 1])
rng = np.random.default_rng(0); within = np.mean(np.abs(np.log(pairs[:, 0] / pairs[:, 1])))
perm = [np.mean(np.abs(np.log(pairs[:, 0] / pairs[rng.permutation(len(pairs)), 1]))) for _ in range(5000)]
out["tau_seed_pairs"] = {"pairs": pairs.tolist(), "spearman": float(r), "p": float(p), "perm_p": float(np.mean(np.array(perm) <= within))}
print(f"2) SF relaxation time seed pairs {pairs.round(0).tolist()} spearman {r:.2f} (p={p:.3f}), permutation p={np.mean(np.array(perm) <= within):.3f}")
# 3) Rolling Forcing per-video change vectors (start 5-20 s vs end 160-180 s)
print("3) per-video appearance change, start -> end (low-level)")
keys = ["brightness", "contrast", "saturation", "sharpness", "spectral_slope", "flow_mag"]
for m in ("rf", "ll", "sf"):
    rows = []
    for k, r in sorted(D[m].items()):
        L = Dp.per_second(np.nan_to_num(r["low"]))
        st, en = L[5:20].mean(0), L[-20:].mean(0)
        rows.append({n: float((en[names.index(n)] - st[names.index(n)]) / (abs(st[names.index(n)]) + 1e-6)) for n in keys})
    out.setdefault("change", {})[m] = rows
    M = np.array([[r_[n] for n in keys] for r_ in rows])
    print(f"   {m}: median relative change", dict(zip(keys, np.round(np.median(M, 0), 2))), "| frac brightness up", round(float(np.mean(M[:, 0] > 0.05)), 2), "frac sharpness up", round(float(np.mean(M[:, 3] > 0.1)), 2))
# 4) recurrence with smoothness-preserving surrogates (1 Hz, PCA-10, phase-randomized per component)
def pca10(X):
    X = X - X.mean(0); U, S, Vt = np.linalg.svd(X, full_matrices=False); return X @ Vt[:10].T
def phase_rand(Y, rng):
    F = np.fft.rfft(Y, axis=0); ph = np.exp(2j * np.pi * rng.random(F.shape)); ph[0] = 1
    return np.fft.irfft(F * ph, n=len(Y), axis=0)
def rqa(Y, rr=0.1, lmin=3):
    Dm = np.linalg.norm(Y[:, None] - Y[None], axis=-1); n = len(Dm); eps = np.quantile(Dm[np.triu_indices(n, 1)], rr)
    R_ = Dm <= eps; np.fill_diagonal(R_, False); rec = R_.sum(); det = lam = 0
    for k in range(1, n):
        dl = np.diagonal(R_, k).astype(int); runs = np.diff(np.concatenate([[0], dl, [0]])); L = np.where(runs == -1)[0] - np.where(runs == 1)[0]; det += L[L >= lmin].sum()
    for j in range(n):
        col = R_[:, j].astype(int); runs = np.diff(np.concatenate([[0], col, [0]])); L = np.where(runs == -1)[0] - np.where(runs == 1)[0]; lam += L[L >= lmin].sum()
    return 2 * det / rec, lam / rec
print("4) recurrence vs phase-randomized surrogates (DET, LAM)")
rng = np.random.default_rng(0)
for m, runs in D.items():
    v, s = [], []
    for k, r in runs.items():
        Y = pca10(Dp.per_second(r["cls"]))
        v.append(rqa(Y)); s.append(np.mean([rqa(phase_rand(Y, rng)) for _ in range(5)], 0))
    v, s = np.array(v), np.array(s)
    out.setdefault("rqa", {})[m] = {"DET": float(v[:, 0].mean()), "DET_sur": float(s[:, 0].mean()), "LAM": float(v[:, 1].mean()), "LAM_sur": float(s[:, 1].mean()),
                                    "frac_DET_above_sur": float(np.mean(v[:, 0] > s[:, 0]))}
    print(f"   {m}: DET {v[:,0].mean():.3f} vs surrogate {s[:,0].mean():.3f} (above in {np.mean(v[:,0]>s[:,0]):.2f} of runs); LAM {v[:,1].mean():.3f} vs {s[:,1].mean():.3f}")
# 5) prompt half-life (SF): fit D_prompt(t) excess over D_noise-matched floor
ph = json.load(open("results_long/phase0.json"))["sync"]["sf"]
dp = np.array(ph["D_prompt"]); t = np.arange(len(dp))
f = lambda t, a, tau, c: c + a * np.exp(-t / tau)
pp, _ = curve_fit(f, t[5:], dp[5:], p0=[0.4, 100, 0.5], maxfev=20000)
out["prompt_distance_fit"] = {"a": float(pp[0]), "tau_s": float(pp[1]), "asymptote": float(pp[2]), "half_life_s": float(pp[1] * np.log(2))}
print(f"5) SF different-prompt distance decays with tau {pp[1]:.0f} s (half-life {pp[1]*np.log(2):.0f} s) toward {pp[2]:.2f}")
json.dump(out, open("results_long/deep4.json", "w"), indent=1)
