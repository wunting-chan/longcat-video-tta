import numpy as np, json
from scipy.stats import wilcoxon, kendalltau
import deep as Dp
D, names = Dp.load(); sf = D["sf"]; FPS = 16
res = {}
# ---- transition shape: abrupt or gradual? (10%->90% of the proximity rise, per video)
widths, curves = [], {}
for k in sorted(sf):
    prox, _ = Dp.attractor_proximity(sf, k)
    sm = np.convolve(prox, np.ones(5) / 5, "same")
    lo, hi = np.median(sm[2:12]), np.median(sm[-20:])
    f = (sm - lo) / (hi - lo + 1e-9)
    t10 = next((t for t in range(len(f)) if f[t] > 0.1 and (f[t:t+5] > 0.1).all()), None)
    t90 = next((t for t in range(len(f)) if f[t] > 0.9 and (f[t:t+5] > 0.9).all()), None)
    widths.append((t10, t90, None if t10 is None or t90 is None else t90 - t10))
    curves[f"p{k[0]}s{k[1]}"] = f.tolist()
w = np.array([x[2] for x in widths if x[2] is not None], float)
res["transition_10_90_s"] = {"per_video": widths, "median_width_s": float(np.median(w)), "iqr": [float(np.percentile(w, 25)), float(np.percentile(w, 75))]}
print("transition width 10-90 pct (s): median", np.median(w), "IQR", np.percentile(w, [25, 75]).round(0), "per video", [x[2] for x in widths])
# ---- within-video early warning: indicator levels in [Tc-30, Tc] vs [Tc-90, Tc-60] (same video)
A = {k: Dp.collapse_time(Dp.attractor_proximity(sf, k)[0]) for k in sf}
def stats_win(x):
    x = x - np.convolve(x, np.ones(5) / 5, "same"); x = x[2:-2]
    v = x.var(); return v, (np.corrcoef(x[1:], x[:-1])[0, 1] if v > 0 else np.nan)
fi = {"flow": names.index("flow_mag"), "sharpness": names.index("sharpness"), "saturation": names.index("saturation")}
for nm in ["attractor_proximity", "dino_speed", "flow", "sharpness", "saturation"]:
    near, far = [], []
    for k, tc in A.items():
        if tc is None or tc < 90: continue
        if nm == "attractor_proximity": s = Dp.attractor_proximity(sf, k)[0]
        elif nm == "dino_speed":
            c = sf[k]["cls"]; s = Dp.per_second(1 - (c[8:] * c[:-8]).sum(1))
        else: s = Dp.per_second(np.nan_to_num(sf[k]["low"][:, fi[nm]]))
        near.append(stats_win(s[tc - 30:tc])); far.append(stats_win(s[tc - 90:tc - 60]))
    near, far = np.array(near), np.array(far)
    if len(near) >= 5:
        pv = wilcoxon(near[:, 0], far[:, 0]).pvalue; pa = wilcoxon(np.nan_to_num(near[:, 1]), np.nan_to_num(far[:, 1])).pvalue
        res[f"within_{nm}"] = {"n": len(near), "var_ratio_med": float(np.median(near[:, 0] / far[:, 0])), "var_p": float(pv),
                               "ac1_near": float(np.nanmean(near[:, 1])), "ac1_far": float(np.nanmean(far[:, 1])), "ac1_p": float(pa)}
        print(f"within-video {nm:20s} n={len(near)} var near/far median ratio {np.median(near[:,0]/far[:,0]):.2f} (p={pv:.3f}); AC1 far {np.nanmean(far[:,1]):.2f} -> near {np.nanmean(near[:,1]):.2f} (p={pa:.3f})")
json.dump(res, open("results_long/deep2.json", "w"), indent=1, default=float)
