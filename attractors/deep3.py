import numpy as np, json, glob
from scipy.stats import spearmanr, wilcoxon
FPS, PRE = 16, 33
def load(m):
    fs = sorted(glob.glob(f"obs/{m}/*.npz")); C, L, S = [], [], []
    for f in fs:
        d = np.load(f); c = d["cls"].astype(np.float32)[:537]; c /= np.linalg.norm(c, axis=1, keepdims=True)
        C.append(c); L.append(d["low"][:537]); S.append(str(d["stem"]))
    return np.stack(C), np.stack(L), S, [str(x) for x in np.load(fs[0])["low_names"]]
M = {m: load(m) for m in ["notta", "rolling_notta", "sf_always_search", "sf_pseudo", "real"]}
names = M["notta"][3]
res = {}
def drift_to_attractor(C):
    # leave-one-out centroid of everyone's last 2 s; per clip: proximity gain from opening (0.5-2 s) to last 2 s
    ends = C[:, -2 * FPS:].mean(1); n = len(C); out = np.zeros(n)
    for i in range(n):
        cen = np.delete(ends, i, 0).mean(0); cen /= np.linalg.norm(cen)
        a = C[i, 8:PRE].mean(0); a /= np.linalg.norm(a); b = C[i, -2 * FPS:].mean(0); b /= np.linalg.norm(b)
        out[i] = b @ cen - a @ cen
    return out
for m in M: M[m] = M[m] + (drift_to_attractor(M[m][0]),)
for m in M:
    print(f"{m:18s} per-clip drift toward shared end state: median {np.median(M[m][4]):+.3f}, frac>0 {np.mean(M[m][4] > 0):.2f}")
# predictors of SF per-clip drift from the REAL opening (frames 8..32 are the same real pixels in every method)
C, L, S, _, dr = M["notta"]
op = {n: L[:, 8:PRE, i].mean(1) for i, n in enumerate(names) if n != "diff_energy"}
op["opening_flow"] = L[:, 9:PRE, names.index("flow_mag")].mean(1)
# content proximity of the opening to SF's end state (leave-one-out)
ends = C[:, -2 * FPS:].mean(1)
op["opening_near_endstate"] = np.array([ (C[i, 8:PRE].mean(0) / np.linalg.norm(C[i, 8:PRE].mean(0))) @ (lambda c: c / np.linalg.norm(c))(np.delete(ends, i, 0).mean(0)) for i in range(len(C))])
print("\npredictors of Self Forcing per-clip drift (n=128), Spearman:")
P = {}
for n, x in op.items():
    r, p = spearmanr(x, dr); P[n] = (float(r), float(p)); print(f"   {n:24s} {r:+.2f}  p={p:.4f}")
res["sf_drift_predictors"] = P
# is a clip's drift a property of the clip? correlation across methods (same opening)
for a, b in [("notta", "rolling_notta"), ("notta", "sf_always_search"), ("notta", "real"), ("rolling_notta", "real")]:
    r, p = spearmanr(M[a][4], M[b][4]); print(f"drift correlation {a} vs {b}: {r:+.2f} (p={p:.3g})"); res[f"corr_{a}_{b}"] = (float(r), float(p))
# paired: does search slow the drift for the same clip?
for b in ["sf_always_search", "sf_pseudo"]:
    d = M[b][4] - M["notta"][4]; p = wilcoxon(d).pvalue
    print(f"{b} minus SF per-clip drift: median {np.median(d):+.4f}, frac less drift {np.mean(d < 0):.2f}, Wilcoxon p={p:.3g}")
    res[f"paired_{b}"] = (float(np.median(d)), float(np.mean(d < 0)), float(p))
json.dump(res, open("results/deep3.json", "w"), indent=1)
