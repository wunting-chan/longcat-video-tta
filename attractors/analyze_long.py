#!/usr/bin/env python3
"""Phase 0 analysis: multi-minute rollouts of Self Forcing (sf), Rolling Forcing (rf), LongLive (ll).

Probes
  drift     per-observable median trajectory; settle vs run-away (drift speed early vs late)
  freeze    frozen 1 s windows (Phase -1 real-video thresholds); absorbing test P(stay frozen)
  sync      over time: D_noise = same prompt / different seed, D_prompt = different prompt /
            same seed (identical noise), D_both = different prompt and seed
  memory    which prompt produced frame t? reference = the other seed's first 2 s
  ends      end-frame gallery per model
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import style

FPS = 16
NAME = {"sf": "Self Forcing", "rf": "Rolling Forcing", "ll": "LongLive"}
COL = {"sf": (0, 0, 1), "rf": (1, 0, 0), "ll": (0, 0.5, 0)}


def load(obs):
    D = {}
    for f in sorted(glob.glob(os.path.join(obs, "*", "*.npz"))):
        if "_twin" in os.path.basename(f):
            continue
        d = np.load(f)
        m = str(d["model"])
        c = d["cls"].astype(np.float32)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        D.setdefault(m, {})[(int(d["prompt_index"]), int(d["seed"]))] = {
            "cls": c, "low": d["low"], "thumbs": d["thumbs"], "tidx": d["thumb_idx"]}
    names = [str(x) for x in np.load(f)["low_names"]]
    return D, names


def smooth(v, w=FPS):
    return np.convolve(np.nan_to_num(v), np.ones(w) / w, "valid")


def drift(D, names):
    out = {}
    for m, runs in D.items():
        T = min(r["low"].shape[0] for r in runs.values())
        L = np.stack([r["low"][:T] for r in runs.values()])           # [run, T, K]
        med = np.nanmedian(L, 0)
        sec = T // FPS
        per_s = np.array([np.nanmedian(L[:, s * FPS:(s + 1) * FPS], axis=(0, 1)) for s in range(sec)])
        early = np.abs(per_s[20] - per_s[5]) / 15
        late = np.abs(per_s[-1] - per_s[-31]) / 30
        out[m] = {"T": int(T), "per_second_median": per_s.tolist(),
                  "early_rate": dict(zip(names, early.tolist())), "late_rate": dict(zip(names, late.tolist())),
                  "late_over_early": dict(zip(names, (late / np.maximum(early, 1e-9)).tolist()))}
    return out


def freeze(D, names, th):
    fi = names.index("flow_mag")
    out = {}
    lag = 8
    for m, runs in D.items():
        rows = []
        for r in runs.values():
            c, fl = r["cls"], r["low"][:, fi]
            sp = 1 - (c[lag:] * c[:-lag]).sum(1)
            n = (len(c) - lag) // FPS
            rows.append([(np.nanmedian(sp[k * FPS:(k + 1) * FPS]) < th["speed"]) and
                         (np.nanmedian(fl[k * FPS + lag:(k + 1) * FPS + lag]) < th["flow"]) for k in range(n - 1)])
        n = min(len(x) for x in rows)
        F = np.array([x[:n] for x in rows])
        f0, f1 = F[:, :-1].ravel(), F[:, 1:].ravel()
        out[m] = {"frozen_frac": F.mean(0).tolist(), "frac_ever": float(F.any(1).mean()),
                  "frac_last30s_all": float(F[:, -30:].all(1).mean()),
                  "p_stay": float(f1[f0].mean()) if f0.any() else None,
                  "p_enter": float(f1[~f0].mean()) if (~f0).any() else None,
                  "first_frozen_s": [int(np.argmax(x)) if x.any() else None for x in F]}
    return out


def sync(D):
    out = {}
    for m, runs in D.items():
        T = min(r["cls"].shape[0] for r in runs.values())
        keys = sorted(runs)
        P = sorted({k[0] for k in keys}); S = sorted({k[1] for k in keys})
        def dist(a, b):
            return 1 - (runs[a]["cls"][:T] * runs[b]["cls"][:T]).sum(1)
        dn = [dist((p, S[0]), (p, S[1])) for p in P if len(S) > 1 and (p, S[1]) in runs]
        dp = [dist((p, s), (q, s)) for s in S for p, q in itertools.combinations(P, 2) if (p, s) in runs and (q, s) in runs]
        db = [dist((p, S[0]), (q, S[1])) for p in P for q in P if p != q and len(S) > 1 and (q, S[1]) in runs]
        sec = lambda x: np.array([np.mean(np.stack(x)[:, s * FPS:(s + 1) * FPS]) for s in range(T // FPS)]) if x else None
        out[m] = {k: (v.tolist() if v is not None else None) for k, v in
                  {"D_noise": sec(dn), "D_prompt": sec(dp), "D_both": sec(db)}.items()}
    return out


def memory(D):
    out = {}
    for m, runs in D.items():
        T = min(r["cls"].shape[0] for r in runs.values())
        P = sorted({k[0] for k in runs}); S = sorted({k[1] for k in runs})
        if len(S) < 2:
            continue
        acc = np.zeros(T // FPS)
        for s in range(T // FPS):
            hits = []
            for (p, sd) in runs:
                other = [x for x in S if x != sd][0]
                refs = np.stack([runs[(q, other)]["cls"][: 2 * FPS].mean(0) for q in P])
                refs /= np.linalg.norm(refs, axis=1, keepdims=True)
                f = runs[(p, sd)]["cls"][s * FPS:(s + 1) * FPS].mean(0)
                hits.append(P[int(np.argmax(refs @ f))] == p)
            acc[s] = np.mean(hits)
        out[m] = {"acc_per_s": acc.tolist(), "chance": 1 / len(P)}
    return out


def figures(R, D, names, out):
    style()
    ms = [m for m in ("sf", "rf", "ll") if m in D]
    fig, ax = plt.subplots(2, 3, figsize=(6.3, 3.7))
    for a, obs in zip(ax[0], ["saturation", "sharpness", "flow_mag"]):
        i = names.index(obs)
        for m in ms:
            y = np.array(R["drift"][m]["per_second_median"])[:, i]
            a.plot(np.arange(len(y)), y, color=COL[m], lw=0.8, label=NAME[m])
        a.set_title(f"Median {obs.replace('_', ' ')}", pad=3); a.set_xlabel("Time (s)")
    ax[0, 0].legend(fontsize=4.8)
    a = ax[1, 0]
    for m in ms:
        a.plot(np.arange(len(R["freeze"][m]["frozen_frac"])), R["freeze"][m]["frozen_frac"], color=COL[m], lw=0.8)
    a.set_title("Share of rollouts frozen (1 s windows)", pad=3); a.set_xlabel("Time (s)")
    a = ax[1, 1]
    for m in ms:
        s = R["sync"][m]
        if s["D_prompt"] is None:
            continue
        a.plot(s["D_prompt"], color=COL[m], lw=0.8, label=f"{NAME[m]}: diff. prompt, same noise")
        if s["D_noise"] is not None:
            a.plot(s["D_noise"], color=COL[m], lw=0.8, ls=(0, (3, 1.3)), label="same prompt, diff. noise")
            a.plot(s["D_both"], color=COL[m], lw=0.5, ls=(0, (1, 1.2)), label="diff. prompt and noise")
    a.set_title("Who decides the future: prompt or noise?", pad=3); a.set_xlabel("Time (s)"); a.set_ylabel("1 − cos")
    a.legend(fontsize=3.8, loc="center right")
    a = ax[1, 2]
    for m in ms:
        if m in R["memory"]:
            a.plot(R["memory"][m]["acc_per_s"], color=COL[m], lw=0.8, label=NAME[m])
            a.axhline(R["memory"][m]["chance"], color="#777777", lw=0.5, ls=(0, (3, 1.5)))
    a.set_title("Which prompt produced this second?", pad=3); a.set_xlabel("Time (s)"); a.set_ylabel("Retrieval accuracy")
    fig.tight_layout(w_pad=0.8, h_pad=1.0)
    fig.savefig(os.path.join(out, "phase0_probes.png"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(out, "phase0_probes.pdf"), bbox_inches="tight")
    plt.close(fig)
    # end-state galleries: rows = prompts (seed 0), columns = thumbnails over time, one figure per model
    for m in ms:
        runs = D[m]
        P = sorted({k[0] for k in runs})
        keys = [(p, 0) for p in P if (p, 0) in runs]
        nt = len(runs[keys[0]]["tidx"])
        fig, axs = plt.subplots(len(keys), nt, figsize=(6.3, 6.3 * len(keys) / nt * 120 / 208 * 1.05))
        for r, k in enumerate(keys):
            for c in range(nt):
                ax_ = axs[r, c]
                ax_.imshow(runs[k]["thumbs"][c]); ax_.set_xticks([]); ax_.set_yticks([])
                if r == 0:
                    ax_.set_title(f"{runs[k]['tidx'][c] / FPS:.0f} s", fontsize=5.0, pad=2)
            axs[r, 0].set_ylabel(f"p{k[0]}", fontsize=5)
        fig.suptitle(NAME[m], fontsize=6.4, y=1.0)
        fig.subplots_adjust(wspace=0.03, hspace=0.03)
        fig.savefig(os.path.join(out, f"phase0_strip_{m}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--thresholds", required=True, help="Phase -1 probes.json (real-video freeze thresholds)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D, names = load(a.obs)
    th = json.load(open(a.thresholds))["freezing"]["_thresholds"]
    R = {"drift": drift(D, names), "freeze": freeze(D, names, th), "sync": sync(D), "memory": memory(D),
         "n_runs": {m: len(v) for m, v in D.items()}, "thresholds": th}
    json.dump(R, open(os.path.join(a.out, "phase0.json"), "w"))
    figures(R, D, names, a.out)
    for m in R["drift"]:
        dr = R["drift"][m]
        print(m, "runs", R["n_runs"][m], "T", dr["T"],
              "late/early drift", {k: round(v, 2) for k, v in dr["late_over_early"].items() if k in ("saturation", "sharpness", "contrast", "flow_mag", "spectral_slope")})
        fz = R["freeze"][m]
        print("   freeze ever", round(fz["frac_ever"], 2), "all-frozen last 30 s", round(fz["frac_last30s_all"], 2),
              "P(stay)", fz["p_stay"], "P(enter)", fz["p_enter"])
        s = R["sync"][m]
        for k in ("D_noise", "D_prompt", "D_both"):
            if s[k]:
                print(f"   {k}: 5s {s[k][5]:.3f} 30s {s[k][30]:.3f} 60s {s[k][min(60, len(s[k]) - 1)]:.3f} end {s[k][-1]:.3f}")
        if m in R["memory"]:
            acc = R["memory"][m]["acc_per_s"]
            print("   prompt retrieval 5/30/60/end", [round(acc[i], 2) for i in (5, 30, min(60, len(acc) - 1), len(acc) - 1)])


if __name__ == "__main__":
    main()
