"""Relaxation time per rollout: distance in DINOv2 space from frame t to the rollout's OWN final state
(mean of last 20 s), per second; fit d(t) = d_inf + A*exp(-t/tau) on t in [5, T-20]. Also prompt half-life (SF)."""
import numpy as np, json
from scipy.optimize import curve_fit
import deep as Dp
D, names = Dp.load(); FPS = 16; out = {}
def fit(y):
    t = np.arange(len(y), dtype=float)
    f = lambda t, a, tau, c: c + a * np.exp(-t / tau)
    try:
        p, _ = curve_fit(f, t, y, p0=[y[0] - y[-1], 40, y[-1]], bounds=([0, 1, -1], [2, 2000, 2]), maxfev=20000)
        r2 = 1 - np.sum((y - f(t, *p)) ** 2) / np.sum((y - y.mean()) ** 2)
        return float(p[1]), float(r2)
    except Exception:
        return None, None
for m, runs in D.items():
    taus, r2s = [], []
    for k, r in sorted(runs.items()):
        ps = Dp.per_second(r["cls"]); ps /= np.linalg.norm(ps, axis=1, keepdims=True)
        end = ps[-20:].mean(0); end /= np.linalg.norm(end)
        d = 1 - ps @ end
        tau, r2 = fit(d[5:-20])
        taus.append(tau); r2s.append(r2)
    t = np.array([x for x in taus if x is not None]); r = np.array([x for x in r2s if x is not None])
    out[m] = {"tau_s": taus, "r2": r2s, "median_tau": float(np.median(t)), "iqr": np.percentile(t, [25, 75]).tolist(), "median_r2": float(np.median(r))}
    print(f"{m}: relaxation time median {np.median(t):.0f} s (IQR {np.percentile(t,25):.0f}-{np.percentile(t,75):.0f}), fit R2 median {np.median(r):.2f}; capped at 2000: {(t>=1999).sum()}")
# distance to own end state at 10 s vs 170 s, to see how far each model actually travels
for m, runs in D.items():
    trav = []
    for k, r in runs.items():
        ps = Dp.per_second(r["cls"]); ps /= np.linalg.norm(ps, axis=1, keepdims=True)
        st = ps[2:8].mean(0); st /= np.linalg.norm(st); end = ps[-20:].mean(0); end /= np.linalg.norm(end)
        trav.append(1 - st @ end)
    out[m]["start_to_end_distance_median"] = float(np.median(trav))
    print(f"   {m}: start-to-end distance median {np.median(trav):.3f}")
json.dump(out, open("results_long/relax.json", "w"), indent=1)
