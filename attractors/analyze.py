#!/usr/bin/env python3
"""Phase -1 attractor probes on existing 30 s rollouts (cite-128, Wan2.1-1.3B students).

Methods: notta (Self Forcing), rolling_notta (Rolling Forcing), sf_always_search, sf_pseudo,
and real (the true Panda continuation, as a control). Frames 0..32 are the real opening.
Writes probes.json and figures into --out.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FPS, PRE, CHUNK = 16, 33, 84
LABEL = {"real": "Real video", "notta": "Self Forcing", "rolling_notta": "Rolling Forcing",
         "sf_always_search": "SF + always search", "sf_pseudo": "SF + pseudo search"}
COLOR = {"real": (0, 0, 0), "notta": (0, 0, 1), "rolling_notta": (1, 0, 0),
         "sf_always_search": (0, 0.5, 0), "sf_pseudo": (0.75, 0, 0.75)}


def style():
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 6.0, "axes.linewidth": 0.42, "axes.grid": True,
        "grid.color": "#B0B0B0", "grid.linewidth": 0.42, "lines.linewidth": 0.8, "lines.markersize": 2.4,
        "legend.fontsize": 5.4, "axes.titlesize": 6.4, "axes.labelsize": 6.2, "xtick.labelsize": 5.6,
        "ytick.labelsize": 5.6, "xtick.major.width": 0.42, "ytick.major.width": 0.42, "pdf.fonttype": 42})


def load(obs_dir, methods):
    data = {}
    for m in methods:
        files = sorted(glob.glob(os.path.join(obs_dir, m, "*.npz")))
        if not files:
            continue
        cls, low, chosen, stems = [], [], [], []
        for f in files:
            d = np.load(f)
            c = d["cls"].astype(np.float32)
            cls.append(c / np.linalg.norm(c, axis=1, keepdims=True))
            low.append(d["low"])
            chosen.append(d["chosen"])
            stems.append(str(d["stem"]))
        T = min(len(c) for c in cls)
        data[m] = {"cls": np.stack([c[:T] for c in cls]), "low": np.stack([l[:T] for l in low]),
                   "chosen": chosen, "stems": stems, "names": [str(x) for x in np.load(files[0])["low_names"]]}
    # align clip order by stem across methods
    ref = data[methods[0]]["stems"] if methods[0] in data else None
    for m in data:
        order = [data[m]["stems"].index(s) for s in ref if s in data[m]["stems"]]
        for k in ("cls", "low"):
            data[m][k] = data[m][k][order]
        data[m]["chosen"] = [data[m]["chosen"][i] for i in order]
        data[m]["stems"] = [data[m]["stems"][i] for i in order]
    return data


def secs(t):
    return (np.asarray(t) - PRE) / FPS


# ------------------------------------------------------------------ probe 1: memory half-life
def memory(data):
    """Top-1 retrieval of a clip's own opening (mean [CLS] of real frames 8..32) from frame t."""
    out = {}
    for m, d in data.items():
        cls = d["cls"]
        ref = cls[:, 8:PRE].mean(1)
        ref /= np.linalg.norm(ref, axis=1, keepdims=True)
        n, T, _ = cls.shape
        acc = np.zeros(T)
        own = np.zeros(T)
        for t in range(T):
            s = cls[:, t] @ ref.T                           # [clip, opening]
            acc[t] = (s.argmax(1) == np.arange(n)).mean()
            own[t] = (np.diag(s) - (s.sum(1) - np.diag(s)) / (n - 1)).mean()   # own-vs-others margin
        # half-life of the own-opening margin over the generated part (exponential fit on 1 s means)
        tt = np.arange(PRE, T)
        y = own[PRE:]
        sec = np.arange(len(y)) // FPS
        ym = np.array([y[sec == k].mean() for k in range(sec.max() + 1)])
        xs = np.arange(len(ym)) + 0.5
        pos = ym > 1e-4
        hl = None
        if pos.sum() >= 4:
            b = np.polyfit(xs[pos][: min(12, pos.sum())], np.log(ym[pos][: min(12, pos.sum())]), 1)[0]
            hl = float(np.log(2) / -b) if b < 0 else None
        out[m] = {"acc": acc.tolist(), "margin": own.tolist(), "half_life_s": hl,
                  "acc_at_s": {str(s): float(acc[min(T - 1, PRE + s * FPS)]) for s in (1, 5, 10, 20, 30)}}
    return out


# ------------------------------------------------------------------ probe 2: ensemble convergence
def convergence(data, rng):
    """Mean pairwise cosine distance among the 128 clips at time t (ensemble spread),
    and distance of each clip to the time-t ensemble mean direction."""
    out = {}
    for m, d in data.items():
        cls = d["cls"]
        n, T, _ = cls.shape
        spread = np.zeros(T)
        for t in range(T):
            g = cls[:, t] @ cls[:, t].T
            spread[t] = 1 - (g.sum() - n) / (n * (n - 1))
        out[m] = {"spread": spread.tolist(),
                  "spread_open": float(spread[8:PRE].mean()), "spread_last2s": float(spread[-2 * FPS:].mean())}
    return out


# ------------------------------------------------------------------ probe 3: fixed points / freezing
def freezing(data, names):
    """Embedding speed (1 - cos between frames 8 apart) and flow; a clip is 'frozen' in a 1 s window
    when both are below thresholds set from the real-video distribution (5th percentile)."""
    fi = names.index("flow_mag")
    real = data.get("real")
    out = {}
    lag = 8
    speeds = {}
    for m, d in data.items():
        cls = d["cls"]
        sp = 1 - np.einsum("ntd,ntd->nt", cls[:, lag:], cls[:, :-lag])
        speeds[m] = sp
    if real is not None:
        th_sp = np.nanpercentile(speeds["real"][:, PRE:], 5)
        th_fl = np.nanpercentile(real["low"][:, PRE:, fi], 5)
    else:
        th_sp, th_fl = 0.01, 0.05
    for m, d in data.items():
        sp = speeds[m]
        fl = d["low"][:, :, fi]
        n, T = fl.shape
        W = FPS
        nwin = (T - PRE) // W
        frozen = np.zeros((n, nwin), bool)
        for k in range(nwin):
            a, b = PRE + k * W, PRE + (k + 1) * W
            frozen[:, k] = (np.nanmedian(sp[:, a - lag:b - lag], 1) < th_sp) & (np.nanmedian(fl[:, a:b], 1) < th_fl)
        # absorbing test: P(frozen at k+1 | frozen at k) vs P(frozen at k+1 | living at k)
        f0, f1 = frozen[:, :-1].ravel(), frozen[:, 1:].ravel()
        p_stay = float(f1[f0].mean()) if f0.any() else None
        p_enter = float(f1[~f0].mean()) if (~f0).any() else None
        out[m] = {"frozen_frac": frozen.mean(0).tolist(), "p_stay_frozen": p_stay, "p_enter_frozen": p_enter,
                  "frac_ever_frozen": float(frozen.any(1).mean()), "frac_frozen_last5s": float(frozen[:, -5:].all(1).mean()),
                  "speed_mean": np.nanmean(sp, 0).tolist(), "flow_mean": np.nanmean(fl, 0).tolist()}
    out["_thresholds"] = {"speed": float(th_sp), "flow": float(th_fl)}
    return out


# ------------------------------------------------------------------ probe 4: empirical vector field
def vector_field(data, names, xa="contrast", ya="flow_mag", dt=FPS, nb=12):
    xi, yi = names.index(xa), names.index(ya)
    allx = np.concatenate([d["low"][:, PRE:, xi].ravel() for d in data.values()])
    ally = np.concatenate([d["low"][:, PRE:, yi].ravel() for d in data.values()])
    xe = np.nanpercentile(allx, np.linspace(1, 99, nb + 1))
    ye = np.nanpercentile(ally, np.linspace(1, 99, nb + 1))
    out = {"x": xa, "y": ya, "xe": xe.tolist(), "ye": ye.tolist(), "fields": {}}
    for m, d in data.items():
        L = d["low"]
        # 1 s smoothed observables to remove frame jitter
        k = np.ones(FPS) / FPS
        X = np.apply_along_axis(lambda v: np.convolve(np.nan_to_num(v), k, "same"), 1, L[:, PRE:, xi])
        Y = np.apply_along_axis(lambda v: np.convolve(np.nan_to_num(v), k, "same"), 1, L[:, PRE:, yi])
        x0, y0 = X[:, FPS:-dt - FPS].ravel(), Y[:, FPS:-dt - FPS].ravel()
        dx = (X[:, FPS + dt:-FPS] - X[:, FPS:-dt - FPS]).ravel()
        dy = (Y[:, FPS + dt:-FPS] - Y[:, FPS:-dt - FPS]).ravel()
        U = np.full((nb, nb), np.nan); V = np.full((nb, nb), np.nan); C = np.zeros((nb, nb))
        bx = np.clip(np.searchsorted(xe, x0) - 1, 0, nb - 1)
        by = np.clip(np.searchsorted(ye, y0) - 1, 0, nb - 1)
        for i in range(nb):
            for j in range(nb):
                s = (bx == i) & (by == j)
                C[j, i] = s.sum()
                if s.sum() >= 30:
                    U[j, i] = dx[s].mean(); V[j, i] = dy[s].mean()
        out["fields"][m] = {"U": U.tolist(), "V": V.tolist(), "C": C.tolist(),
                            "traj_x": X.mean(0)[::8].tolist(), "traj_y": Y.mean(0)[::8].tolist(),
                            "end_x": X[:, -FPS:].mean(1).tolist(), "end_y": Y[:, -FPS:].mean(1).tolist()}
    return out


# ------------------------------------------------------------------ probe 5: recurrence quantification
def rqa_one(E, rr=0.1, lmin=2):
    D = 1 - E @ E.T
    n = len(D)
    eps = np.quantile(D[np.triu_indices(n, 1)], rr)
    R = D <= eps
    np.fill_diagonal(R, False)
    rec = R.sum()
    if rec == 0:
        return 0, 0, 0
    # diagonal lines (upper triangle)
    det_pts = 0
    for k in range(1, n):
        dline = np.diagonal(R, k).astype(int)
        runs = np.diff(np.concatenate([[0], dline, [0]]))
        st, en = np.where(runs == 1)[0], np.where(runs == -1)[0]
        L = en - st
        det_pts += L[L >= lmin].sum()
    det = 2 * det_pts / rec
    lam_pts, tt = 0, []
    for j in range(n):
        col = R[:, j].astype(int)
        runs = np.diff(np.concatenate([[0], col, [0]]))
        st, en = np.where(runs == 1)[0], np.where(runs == -1)[0]
        L = en - st
        lam_pts += L[L >= lmin].sum(); tt += list(L[L >= lmin])
    return float(det), float(lam_pts / rec), float(np.mean(tt) if tt else 0)


def rqa(data, rng, step=4):
    out = {}
    for m, d in data.items():
        vals, sur = [], []
        for c in d["cls"][:, PRE::step]:
            vals.append(rqa_one(c))
            sur.append(rqa_one(c[rng.permutation(len(c))]))
        v, s = np.array(vals), np.array(sur)
        out[m] = {"DET": float(v[:, 0].mean()), "LAM": float(v[:, 1].mean()), "TT_steps": float(v[:, 2].mean()),
                  "DET_surrogate": float(s[:, 0].mean()), "LAM_surrogate": float(s[:, 1].mean()),
                  "per_clip": v.tolist()}
    return out


# ------------------------------------------------------------------ probe 6: twin divergence
def twins(data, base="notta", arms=("sf_always_search", "sf_pseudo")):
    """Same opening, same host, trajectories split at the first chunk where the search arm did not
    keep candidate 0. Distance after the split, aligned at the split, vs the unrelated-clip level."""
    out = {}
    if base not in data:
        return out
    B = data[base]["cls"]
    n, T, _ = B.shape
    unrelated = float(np.mean([1 - (B[i, PRE:] * B[(i + 1) % n, PRE:]).sum(-1).mean() for i in range(n)]))
    for a in arms:
        if a not in data:
            continue
        A = data[a]["cls"]
        curves, pre_split = [], []
        for i in range(n):
            ch = data[a]["chosen"][i]
            nz = np.where(ch != 0)[0]
            if len(nz) == 0:
                continue
            t0 = PRE + int(nz[0]) * CHUNK
            dist = 1 - (A[i] * B[i]).sum(-1)
            pre_split.append(float(dist[PRE:t0].mean()) if t0 > PRE else np.nan)
            seg = dist[t0:t0 + 3 * CHUNK]
            if len(seg) == 3 * CHUNK:
                curves.append(seg)
        curves = np.array(curves)
        out[a] = {"n": int(len(curves)), "mean_curve": curves.mean(0).tolist() if len(curves) else [],
                  "pre_split_dist": float(np.nanmean(pre_split)) if pre_split else None,
                  "unrelated_level": unrelated}
    return out


# ------------------------------------------------------------------ probe 7: early warning before freezing
def early_warning(data, names, frz, horizon=4 * FPS, win=2 * FPS):
    """For clips that go from living to frozen (first frozen 1 s window preceded by >=3 living windows),
    compute lag-1 autocorrelation and variance of the 1-D embedding velocity in the window just before
    the event vs. the window 'horizon' earlier. Critical slowing down predicts both rise."""
    out = {}
    lag = 8
    for m, d in data.items():
        if m.startswith("_"):
            continue
        cls = d["cls"]
        sp = 1 - np.einsum("ntd,ntd->nt", cls[:, 1:], cls[:, :-1])          # frame-to-frame speed
        ff = np.array(frz[m]["frozen_frac"])
        events = []
        # recompute per-clip frozen windows with the same thresholds
        th = frz["_thresholds"]
        fi = names.index("flow_mag")
        fl = d["low"][:, :, fi]
        spl = 1 - np.einsum("ntd,ntd->nt", cls[:, lag:], cls[:, :-lag])
        nwin = (cls.shape[1] - PRE) // FPS
        for i in range(len(cls)):
            fr = [(np.nanmedian(spl[i, PRE + k * FPS - lag:PRE + (k + 1) * FPS - lag]) < th["speed"]) and
                  (np.nanmedian(fl[i, PRE + k * FPS:PRE + (k + 1) * FPS]) < th["flow"]) for k in range(nwin)]
            for k in range(3, nwin):
                if fr[k] and not any(fr[k - 3:k]):
                    events.append((i, PRE + k * FPS)); break

        def stats(x):
            x = x - x.mean()
            v = x.var()
            ac = (x[1:] * x[:-1]).mean() / v if v > 0 else np.nan
            return ac, v
        near, far = [], []
        for i, te in events:
            a = te - win
            b = te - horizon - win
            if b < PRE + 1:
                continue
            near.append(stats(sp[i, a:te])); far.append(stats(sp[i, b:b + win]))
        near, far = np.array(near), np.array(far)
        if len(near):
            out[m] = {"n_events": len(events), "n_used": len(near),
                      "ac1_far": float(np.nanmean(far[:, 0])), "ac1_near": float(np.nanmean(near[:, 0])),
                      "var_far": float(np.nanmean(far[:, 1])), "var_near": float(np.nanmean(near[:, 1])),
                      "frac_ac1_rises": float(np.nanmean(near[:, 0] > far[:, 0])),
                      "frac_var_rises": float(np.nanmean(near[:, 1] > far[:, 1]))}
        else:
            out[m] = {"n_events": len(events), "n_used": 0}
    return out


# ------------------------------------------------------------------ figures
def figures(P, data, outdir):
    style()
    ms = [m for m in ["real", "notta", "rolling_notta", "sf_always_search", "sf_pseudo"] if m in data]
    T = data[ms[0]]["cls"].shape[1]
    t = secs(np.arange(T))
    fig, ax = plt.subplots(2, 3, figsize=(6.3, 3.6))
    a = ax[0, 0]
    for m in ms:
        a.plot(t, P["memory"][m]["acc"], color=COLOR[m], label=LABEL[m], lw=0.7)
    a.axvline(0, color="#777777", lw=0.4, ls=(0, (2, 1.5)))
    a.set_title("Which opening produced this frame? (top-1, n=128)", pad=3)
    a.set_xlabel("Time after opening (s)"); a.set_ylabel("Retrieval accuracy")
    a.legend(loc="upper right", fontsize=4.6)
    a = ax[0, 1]
    for m in ms:
        a.plot(t, P["convergence"][m]["spread"], color=COLOR[m], lw=0.7)
    a.axvline(0, color="#777777", lw=0.4, ls=(0, (2, 1.5)))
    a.set_title("Ensemble spread (mean pairwise distance)", pad=3)
    a.set_xlabel("Time after opening (s)"); a.set_ylabel("1 − cos")
    a = ax[0, 2]
    nwin = len(P["freezing"][ms[0]]["frozen_frac"])
    for m in ms:
        a.plot(np.arange(nwin) + 0.5, P["freezing"][m]["frozen_frac"], color=COLOR[m], lw=0.7, marker="o", ms=1.5)
    a.set_title("Share of clips frozen (1 s windows)", pad=3)
    a.set_xlabel("Time after opening (s)"); a.set_ylabel("Fraction frozen")
    a = ax[1, 0]
    for m in ms:
        a.plot(t[:len(P["freezing"][m]["flow_mean"])], np.convolve(np.nan_to_num(P["freezing"][m]["flow_mean"]), np.ones(FPS) / FPS, "same"),
               color=COLOR[m], lw=0.7)
    a.set_title("Mean optical-flow magnitude", pad=3)
    a.set_xlabel("Time after opening (s)"); a.set_ylabel("px / frame (208×120)")
    a = ax[1, 1]
    for a_name, c in P["twins"].items():
        if not c["mean_curve"]:
            continue
        tt = np.arange(len(c["mean_curve"])) / FPS
        a.plot(tt, c["mean_curve"], color=COLOR[a_name], lw=0.7, label=f"{LABEL[a_name]} vs SF (n={c['n']})")
        a.axhline(c["unrelated_level"], color="#777777", lw=0.5, ls=(0, (3, 1.5)))
    a.text(0.98, 0.9, "dashed: unrelated clips", transform=a.transAxes, ha="right", fontsize=4.6, color="#555555")
    a.set_title("Twin divergence after a split", pad=3)
    a.set_xlabel("Time after split (s)"); a.set_ylabel("1 − cos (same frame index)")
    a.legend(loc="lower right", fontsize=4.4)
    a = ax[1, 2]
    names = ["DET", "LAM"]
    x = np.arange(len(ms))
    for k, nm in enumerate(names):
        a.bar(x + (k - 0.5) * 0.36, [P["rqa"][m][nm] for m in ms], 0.34,
              color=[(0.3, 0.42, 1.0) if k == 0 else (0.67, 0.76, 1.0)] * len(ms), label=nm)
        a.scatter(x + (k - 0.5) * 0.36, [P["rqa"][m][nm + "_surrogate"] for m in ms], s=4, color="black", zorder=3,
                  label="shuffled-time surrogate" if k == 0 else None)
    a.set_xticks(x); a.set_xticklabels([LABEL[m].replace(" + ", "+\n") for m in ms], fontsize=4.4)
    a.set_title("Recurrence: determinism / laminarity", pad=3); a.legend(fontsize=4.4, loc="lower right")
    fig.tight_layout(w_pad=1.0, h_pad=1.2)
    fig.savefig(os.path.join(outdir, "phase_m1_probes.png"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "phase_m1_probes.pdf"), bbox_inches="tight")
    plt.close(fig)
    # vector field
    vf = P["vector_field"]
    fig, axes = plt.subplots(1, len(ms), figsize=(6.3, 1.6), sharex=True, sharey=True)
    xe, ye = np.array(vf["xe"]), np.array(vf["ye"])
    xc, yc = (xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2
    XX, YY = np.meshgrid(xc, yc)
    for a, m in zip(np.atleast_1d(axes), ms):
        f = vf["fields"][m]
        U, V = np.array(f["U"]), np.array(f["V"])
        a.quiver(XX, YY, U, V, color=COLOR[m], angles="xy", scale_units="xy", scale=1, width=0.006)
        a.scatter(f["end_x"], f["end_y"], s=1.2, color=COLOR[m], alpha=0.35, lw=0)
        a.plot(f["traj_x"], f["traj_y"], color="black", lw=0.6)
        a.set_title(LABEL[m], fontsize=5.6, pad=2)
        a.set_xlabel(vf["x"], fontsize=5.0)
    np.atleast_1d(axes)[0].set_ylabel(vf["y"], fontsize=5.0)
    fig.tight_layout(w_pad=0.4)
    fig.savefig(os.path.join(outdir, "phase_m1_vector_field.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    methods = ["notta", "rolling_notta", "sf_always_search", "sf_pseudo", "real"]
    data = load(a.obs, methods)
    names = data["notta"]["names"]
    P = {"methods": list(data), "n_clips": {m: int(d["cls"].shape[0]) for m, d in data.items()}}
    P["memory"] = memory(data)
    P["convergence"] = convergence(data, rng)
    P["freezing"] = freezing(data, names)
    P["vector_field"] = vector_field(data, names)
    P["rqa"] = rqa(data, rng)
    P["twins"] = twins(data)
    P["early_warning"] = early_warning(data, names, P["freezing"])
    json.dump(P, open(os.path.join(a.out, "probes.json"), "w"))
    figures(P, data, a.out)
    # compact console summary
    for m in data:
        print(f"{m:18s} half-life {P['memory'][m]['half_life_s']}  acc@1/5/10/30s "
              f"{[round(P['memory'][m]['acc_at_s'][k], 3) for k in ('1', '5', '10', '30')]}  "
              f"spread open/last {P['convergence'][m]['spread_open']:.3f}/{P['convergence'][m]['spread_last2s']:.3f}  "
              f"frozen ever/last5 {P['freezing'][m]['frac_ever_frozen']:.2f}/{P['freezing'][m]['frac_frozen_last5s']:.2f}  "
              f"P(stay)={P['freezing'][m]['p_stay_frozen']} P(enter)={P['freezing'][m]['p_enter_frozen']}  "
              f"DET {P['rqa'][m]['DET']:.3f} (sur {P['rqa'][m]['DET_surrogate']:.3f}) LAM {P['rqa'][m]['LAM']:.3f} (sur {P['rqa'][m]['LAM_surrogate']:.3f})")
    print("twins", {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'mean_curve'} for k, v in P["twins"].items()})
    print("early warning", P["early_warning"])


if __name__ == "__main__":
    main()
