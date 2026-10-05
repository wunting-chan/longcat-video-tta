#!/usr/bin/env python3
"""Zero-GPU go/no-go tests on existing rollouts.

A  per-video collapse time (Self Forcing, 180 s): leave-one-out attractor proximity
B  early-warning signals before collapse (variance, lag-1 autocorrelation, Kendall tau)
   vs. matched windows in rollouts that do not collapse (Rolling Forcing, LongLive)
C  do the first 30 s predict when a video collapses?
D  same prompt, different seed: is fate (collapse time) shared?
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy.stats import kendalltau, spearmanr

FPS = 16
OBS = "lobs/main"


def load():
    D = {}
    for f in sorted(glob.glob(os.path.join(OBS, "*", "*.npz"))):
        if "_twin" in os.path.basename(f):
            continue
        d = np.load(f)
        c = d["cls"].astype(np.float32)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        D.setdefault(str(d["model"]), {})[(int(d["prompt_index"]), int(d["seed"]))] = {"cls": c, "low": d["low"]}
    names = [str(x) for x in np.load(f)["low_names"]]
    return D, names


def per_second(x):
    n = len(x) // FPS
    return x[: n * FPS].reshape(n, FPS, *x.shape[1:]).mean(1)


def attractor_proximity(runs, key, end_s=20):
    """cos(frame_t, centroid of the LAST end_s seconds of all OTHER prompts' rollouts), per second,
    minus cos(frame_t, own opening (first 2 s)). Positive = closer to the shared end state than to own start."""
    others = [v["cls"][-end_s * FPS:].mean(0) for k, v in runs.items() if k[0] != key[0]]
    cen = np.mean(others, 0); cen /= np.linalg.norm(cen)
    c = runs[key]["cls"]
    own = c[: 2 * FPS].mean(0); own /= np.linalg.norm(own)
    ps = per_second(c)
    ps /= np.linalg.norm(ps, axis=1, keepdims=True)
    return ps @ cen - ps @ own, ps @ cen


def collapse_time(prox, sustain=10):
    """first second after which proximity stays > 0 (closer to attractor than to own opening) for `sustain` s"""
    pos = prox > 0
    for t in range(len(pos) - sustain):
        if pos[t:t + sustain].all():
            return t
    return None


def ews(series, end, win=20, look=60):
    """rolling variance and lag-1 autocorrelation (window `win` s at 1 Hz after detrending by a
    rolling mean) over [end-look, end]; returns Kendall tau of each indicator against time."""
    x = np.asarray(series, float)
    lo = max(win, end - look)
    if end - lo < 15:
        return None
    var, ac = [], []
    for t in range(lo, end):
        w = x[t - win:t]
        w = w - np.convolve(w, np.ones(5) / 5, "same")      # remove slow trend
        w = w[2:-2]
        v = w.var()
        var.append(v)
        ac.append(np.corrcoef(w[1:], w[:-1])[0, 1] if v > 0 else np.nan)
    tt = np.arange(len(var))
    return kendalltau(tt, var)[0], kendalltau(tt, np.nan_to_num(ac))[0]


def main():
    D, names = load()
    out = {}
    sf = D["sf"]
    # ---------------- A: collapse times
    A = {}
    for k in sorted(sf):
        prox, near = attractor_proximity(sf, k)
        A[k] = {"tc": collapse_time(prox), "prox": prox}
    tcs = {k: v["tc"] for k, v in A.items()}
    out["A_collapse_times_s"] = {f"p{k[0]}s{k[1]}": v for k, v in tcs.items()}
    print("A  SF collapse times (s):", {f"p{k[0]}s{k[1]}": v for k, v in tcs.items()})
    for m in ("rf", "ll"):
        cc = [collapse_time(attractor_proximity(D[m], k)[0]) for k in sorted(D[m])]
        out[f"A_{m}_collapse_times"] = cc
        print(f"   {m} collapse times (control):", cc)

    # ---------------- B: early warning before collapse vs controls
    obs_list = {"attractor_proximity": None, "dino_speed": None, "saturation": names.index("saturation"),
                "sharpness": names.index("sharpness"), "flow": names.index("flow_mag")}

    def series(run, k, model_runs, name):
        if name == "attractor_proximity":
            return attractor_proximity(model_runs, k)[0]
        if name == "dino_speed":
            c = run["cls"]
            sp = 1 - (c[8:] * c[:-8]).sum(1)
            return per_second(sp)
        return per_second(np.nan_to_num(run["low"][:, obs_list[name]]))

    B = {}
    for name in obs_list:
        pre, ctrl = [], []
        for k, a in A.items():
            if a["tc"] is None:
                continue
            r = ews(series(sf[k], k, sf, name), a["tc"])
            if r:
                pre.append(r)
        # controls: same window positions in non-collapsing models (median SF collapse time)
        tmed = int(np.median([v for v in tcs.values() if v is not None]))
        for m in ("rf", "ll"):
            for k in D[m]:
                r = ews(series(D[m][k], k, D[m], name), tmed)
                if r:
                    ctrl.append(r)
        pre, ctrl = np.array(pre), np.array(ctrl)
        B[name] = {"n_pre": len(pre), "n_ctrl": len(ctrl),
                   "tau_var_pre": float(np.nanmean(pre[:, 0])) if len(pre) else None,
                   "tau_ac1_pre": float(np.nanmean(pre[:, 1])) if len(pre) else None,
                   "tau_var_ctrl": float(np.nanmean(ctrl[:, 0])) if len(ctrl) else None,
                   "tau_ac1_ctrl": float(np.nanmean(ctrl[:, 1])) if len(ctrl) else None,
                   "frac_pre_ac1_rising": float(np.mean(pre[:, 1] > 0)) if len(pre) else None,
                   "frac_ctrl_ac1_rising": float(np.mean(ctrl[:, 1] > 0)) if len(ctrl) else None,
                   "frac_pre_var_rising": float(np.mean(pre[:, 0] > 0)) if len(pre) else None,
                   "frac_ctrl_var_rising": float(np.mean(ctrl[:, 0] > 0)) if len(ctrl) else None}
        print(f"B  {name:20s}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in B[name].items()})
    out["B_early_warning"] = B

    # ---------------- C: does the first 30 s predict collapse time?
    keys = [k for k in A if A[k]["tc"] is not None]
    feats = {}
    for k in keys:
        prox = A[k]["prox"]
        lowp = per_second(np.nan_to_num(sf[k]["low"]))
        c = sf[k]["cls"]
        feats[k] = {
            "prox_at_10s": prox[10], "prox_slope_0_30": np.polyfit(np.arange(30), prox[:30], 1)[0],
            "sat_slope_0_30": np.polyfit(np.arange(30), lowp[:30, names.index("saturation")], 1)[0],
            "bright_0_5": lowp[:5, names.index("brightness")].mean(),
            "flow_0_10": lowp[:10, names.index("flow_mag")].mean(),
            "speed_0_30": float(np.mean(1 - (c[8:30 * FPS] * c[:30 * FPS - 8]).sum(1))),
        }
    tc = np.array([A[k]["tc"] for k in keys])
    C = {}
    for f in feats[keys[0]]:
        x = np.array([feats[k][f] for k in keys])
        r, p = spearmanr(x, tc)
        C[f] = {"spearman": float(r), "p": float(p)}
    out["C_early_predictors"] = C
    print("C  predictors of SF collapse time (n=%d):" % len(keys), {f: (round(v['spearman'], 2), round(v['p'], 3)) for f, v in C.items()})

    # ---------------- D: same prompt, two seeds: shared fate?
    pairs = [(A[(p, 0)]["tc"], A[(p, 1)]["tc"]) for p in sorted({k[0] for k in A}) if (p, 1) in A]
    pa = np.array([x for x in pairs if None not in x], float)
    if len(pa) >= 4:
        r, p = spearmanr(pa[:, 0], pa[:, 1])
        within = np.mean(np.abs(pa[:, 0] - pa[:, 1]))
        perm = []
        rng = np.random.default_rng(0)
        for _ in range(5000):
            q = rng.permutation(len(pa))
            perm.append(np.mean(np.abs(pa[:, 0] - pa[q, 1])))
        out["D_shared_fate"] = {"pairs": pairs, "spearman": float(r), "p": float(p),
                                "mean_abs_diff_same_prompt": float(within),
                                "mean_abs_diff_shuffled": float(np.mean(perm)),
                                "perm_p": float(np.mean(np.array(perm) <= within))}
        print("D  seed-pair collapse times:", pairs, "| |diff| same prompt %.1f s vs shuffled %.1f s (perm p=%.3f), spearman %.2f"
              % (within, np.mean(perm), np.mean(np.array(perm) <= within), r))
    json.dump(out, open("results_long/deep.json", "w"), default=lambda o: None if o is None else float(o), indent=1)


if __name__ == "__main__":
    main()
