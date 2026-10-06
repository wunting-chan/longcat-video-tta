#!/usr/bin/env python3
"""Figures for the video-attractors research-update slides.

DeepSeek technical-report style (as measured for the AI progress report, see report/figs.py):
line plots in DejaVu Sans with pure primaries, small markers, thin black axes and a #B0B0B0
grid; bar charts in DejaVu Serif with the #4D6BFE hatched lead series and pastel companions;
image grids as tight mosaics with small serif labels. Sizes are printed sizes for a 16:9
beamer page (6.30 x 3.54 in); type is set ~1.25x the report's so it reads on a projector.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "..", "attractors")
FR = os.path.join(HERE, "frames")
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, A)

FPS = 16
DS_BLUE, DS_LBLUE, DS_GRAY, DS_LGRAY = "#4D6BFE", "#AAC1FF", "#BDBDBD", "#D4D4D4"
DS_TAN, DS_CREAM, DS_TEXT, DS_SPINE = "#E8D2A0", "#F5EBD2", "#262626", "#CCCCCC"
MODELS = ["sf", "rf", "ll"]
MLABEL = {"sf": "Self Forcing", "rf": "Rolling Forcing", "ll": "LongLive"}
MCOL = {"sf": (1, 0, 0), "rf": (0, 0, 1), "ll": (0, 0.5, 0)}
MMARK = {"sf": "o", "rf": "s", "ll": "^"}
PROMPTS = ["Tokyo street", "Woolly mammoths", "Space-man trailer", "Big Sur waves",
           "Fluffy monster", "Paper coral reef", "Crowned pigeon", "Pirate ships in coffee"]
W = 5.9  # usable figure width on a slide, inches


def line_style():
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.0,
        "axes.linewidth": 0.5, "axes.edgecolor": "black",
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "axes.grid": True, "grid.color": "#B0B0B0", "grid.linewidth": 0.45,
        "lines.linewidth": 0.95, "lines.markersize": 3.0,
        "legend.fontsize": 6.6, "legend.framealpha": 0.8, "legend.fancybox": True,
        "axes.titlesize": 7.6, "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def bar_style():
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 7.0,
        "text.color": DS_TEXT, "axes.labelcolor": DS_TEXT,
        "xtick.color": DS_TEXT, "ytick.color": DS_TEXT,
        "axes.edgecolor": DS_SPINE, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 0, "ytick.major.size": 2.4,
        "hatch.color": "white", "hatch.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def bar_axes(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, ls=(0, (4, 3)), color="#E8E8E8", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# =========================================================================== data
def load_long(sub="main"):
    D = {}
    for f in sorted(glob.glob(os.path.join(A, "lobs", sub, "*", "*.npz"))):
        if "_twin" in os.path.basename(f):
            continue
        d = np.load(f)
        c = d["cls"].astype(np.float32)
        n = len(c) // FPS
        ps = c[: n * FPS].reshape(n, FPS, -1).mean(1)
        ps /= np.linalg.norm(ps, axis=1, keepdims=True)
        low = np.nan_to_num(d["low"][: n * FPS]).reshape(n, FPS, -1).mean(1)
        D.setdefault(str(d["model"]), {})[(int(d["prompt_index"]), int(d["seed"]))] = {"cls": ps, "low": low}
    names = [str(x) for x in np.load(f)["low_names"]]
    return D, names


def prompt_metrics(runs):
    """D_prompt(t): mean cosine distance between different-prompt runs of the same seed.
    D_noise(t): mean distance between the two seeds of the same prompt.
    retrieval(t): share of runs whose nearest run of the other seed (at time t) has the same prompt."""
    keys = sorted(runs)
    T = min(len(v["cls"]) for v in runs.values())
    P = sorted({k[0] for k in keys})
    X = {k: runs[k]["cls"][:T] for k in keys}
    dp, dn, acc = [], [], []
    for t in range(T):
        dps = [1 - X[(p, s)][t] @ X[(q, s)][t] for s in (0, 1) for p in P for q in P if p < q]
        dns = [1 - X[(p, 0)][t] @ X[(p, 1)][t] for p in P]
        ok = 0
        for s in (0, 1):
            o = 1 - s
            for p in P:
                sims = [X[(p, s)][t] @ X[(q, o)][t] for q in P]
                ok += int(P[int(np.argmax(sims))] == p)
        dp.append(np.mean(dps)); dn.append(np.mean(dns)); acc.append(ok / (2 * len(P)))
    return np.array(dp), np.array(dn), np.array(acc)


def frame(tag, t):
    p = os.path.join(FR, f"{tag}__t{t}.jpg")
    if os.path.exists(p):
        return Image.open(p).convert("RGB")
    return None


# =========================================================================== image grids
def image_grid(rows, times, row_labels, name, title_times=True, wfig=W, row_label_w=0.95,
               highlight=None):
    """rows: list of frame tags. DeepSeek-style mosaic: thin gaps, serif labels."""
    bar_style()
    ims = [[frame(r, t) for t in times] for r in rows]
    ar = 9 / 16
    for row in ims:
        for im in row:
            if im is not None:
                ar = im.size[1] / im.size[0]
                break
    nr, nc = len(rows), len(times)
    cw = (wfig - row_label_w) / nc
    fig = plt.figure(figsize=(wfig, nr * cw * ar + (0.18 if title_times else 0.02)))
    gs = fig.add_gridspec(nr, nc, left=row_label_w / wfig, right=1, top=1 - (0.16 / fig.get_figheight() if title_times else 0),
                          bottom=0, wspace=0.03, hspace=0.05)
    for i in range(nr):
        for j in range(nc):
            ax = fig.add_subplot(gs[i, j])
            im = ims[i][j]
            if im is not None:
                ax.imshow(im)
            else:
                ax.set_facecolor("#EEEEEE")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if highlight and (i, j) in highlight:
                for s in ax.spines.values():
                    s.set_visible(True); s.set_color(DS_BLUE); s.set_linewidth(1.6)
            if i == 0 and title_times:
                ax.set_title(f"{float(times[j]):g} s" if float(times[j]) >= 1 else "0 s", fontsize=7.2, pad=2.5,
                             color=DS_TEXT)
            if j == 0:
                ax.text(-0.06, 0.5, row_labels[i], transform=ax.transAxes, ha="right", va="center",
                        fontsize=6.9, color=DS_TEXT)
    save(fig, name)


def fig_photo_grids():
    T5 = ["0.5", "30", "60", "120", "178"]
    image_grid([f"main__sf__sf_p0{p}_s0" for p in (0, 1, 3, 4, 6, 7)], T5,
               [PROMPTS[p] for p in (0, 1, 3, 4, 6, 7)], "photo_sf_converge")
    T6 = ["0.5", "20", "60", "90", "120", "178"]
    for p in (0, 3):
        image_grid([f"main__{m}__{m}_p0{p}_s0" for m in MODELS], T6,
                   [MLABEL[m] for m in MODELS], f"photo_three_fates_p{p}")
    # bifurcation: one prompt, four settings
    for p in (0, 5):
        tags = [f"main__sf__sf_p0{p}_s0", f"bif_w21_s3__sf__sf_p0{p}_s0",
                f"bif_w12_s0__sf__sf_p0{p}_s0", f"bif_w12_s3__sf__sf_p0{p}_s0"]
        image_grid(tags, ["0.5", "30", "60", "120", "178"],
                   ["native (window 21, no sink)", "+ sink 3", "window 12", "window 12 + sink 3"],
                   f"photo_bifurcation_p{p}", row_label_w=1.45)
    # twins: same run, one noise block perturbed at 40 s
    for m in MODELS:
        image_grid([f"main__{m}__{m}_p00_s0", f"main__{m}__{m}_p00_s0_twin40e0.05"],
                   ["20", "30", "45", "60", "120", "178"], ["original", "twin (nudged at 29 s)"],
                   f"photo_twins_{m}", row_label_w=1.25)


def fig_end_states():
    """8 prompts x 3 models, the frame at 0.5 s and at 178 s: where each video ends up."""
    bar_style()
    ims = {(m, p, t): frame(f"main__{m}__{m}_p0{p}_s0", t) for m in MODELS for p in range(8) for t in ("0.5", "178")}
    im0 = next(v for v in ims.values() if v is not None)
    ar = im0.size[1] / im0.size[0]
    lw = 0.95
    cw = (W - lw) / 8
    fig = plt.figure(figsize=(W, 4 * cw * ar + 0.28))
    rows = [("start", "sf", "0.5"), ("sf", "sf", "178"), ("rf", "rf", "178"), ("ll", "ll", "178")]
    labels = ["Opening (SF)", "Self Forcing, 3 min", "Rolling Forcing, 3 min", "LongLive, 3 min"]
    gs = fig.add_gridspec(4, 8, left=lw / W, right=1, top=1 - 0.26 / fig.get_figheight(), bottom=0,
                          wspace=0.03, hspace=0.06)
    for i, (_, m, t) in enumerate(rows):
        for p in range(8):
            ax = fig.add_subplot(gs[i, p])
            im = ims[(m, p, t)]
            if im is not None:
                ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(PROMPTS[p].replace(" ", "\n", 1) if len(PROMPTS[p]) > 12 else PROMPTS[p],
                             fontsize=6.0, pad=2)
            if p == 0:
                ax.text(-0.06, 0.5, labels[i], transform=ax.transAxes, ha="right", va="center", fontsize=6.6)
    save(fig, "photo_end_states")


def fig_openings():
    bar_style()
    ims = [frame(f"main__sf__sf_p0{p}_s0", "0.5") for p in range(8)]
    ar = ims[0].size[1] / ims[0].size[0]
    cw = 3.6 / 4
    fig = plt.figure(figsize=(3.6, 2 * (cw * ar + 0.16)))
    gs = fig.add_gridspec(2, 4, left=0, right=1, top=0.93, bottom=0, wspace=0.04, hspace=0.22)
    for p in range(8):
        ax = fig.add_subplot(gs[p // 4, p % 4]); ax.imshow(ims[p]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(PROMPTS[p], fontsize=6.0, pad=2)
    save(fig, "photo_end_states_top")


# =========================================================================== line plots
def fig_convergence(D):
    line_style()
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.15))
    res = {}
    for m in MODELS:
        dp, dn, acc = prompt_metrics(D[m])
        res[m] = (dp, dn, acc)
        t = np.arange(len(dp))
        axs[0].plot(t, dp, color=MCOL[m], label=MLABEL[m], marker=MMARK[m], markevery=15)
        axs[0].plot(t, dn, color=MCOL[m], ls=(0, (3, 2)), lw=0.7)
        axs[1].plot(t, acc * 100, color=MCOL[m], label=MLABEL[m], marker=MMARK[m], markevery=15)
    axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Distance between videos (1 − cos)")
    axs[0].set_title("(a) Different prompts become the same video")
    axs[0].plot([], [], color="k", ls=(0, (3, 2)), lw=0.7, label="same prompt, other seed")
    axs[0].legend(loc="lower left")
    axs[1].axhline(100 / 8, color="#777777", lw=0.7, ls=":")
    axs[1].text(178, 100 / 8 + 2, "chance", ha="right", fontsize=6.2, color="#555555")
    axs[1].set_xlabel("Time (s)"); axs[1].set_ylabel("Prompt identifiable from frame (%)")
    axs[1].set_title("(b) The prompt is forgotten")
    axs[1].set_ylim(0, 105)
    axs[1].legend(loc="lower left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "line_convergence")
    return res


def fig_halflife(res):
    line_style()
    dp, dn, _ = res["sf"]
    t = np.arange(len(dp), dtype=float)
    f = lambda t, a, tau, c: c + a * np.exp(-t / tau)
    (a, tau, c), _ = curve_fit(f, t, dp, p0=(0.4, 80, 0.6))
    hl = tau * np.log(2)
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.1))
    ax.plot(t, dp, color=MCOL["sf"], marker="o", markevery=15, label="Self Forcing, different prompts")
    ax.plot(t, f(t, a, tau, c), color="k", lw=0.8, ls="--", label=f"exponential fit, half-life {hl:.0f} s")
    ax.plot(t, dn, color="#777777", lw=0.8, label="noise floor: same prompt, other seed")
    ax.axhline(c, color="#999999", lw=0.6, ls=":")
    ax.text(2, c - 0.035, f"asymptote {c:.2f}", fontsize=6.2, color="#555555")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance between videos (1 − cos)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "line_halflife")
    return {"a": a, "tau": tau, "c": c, "half_life": hl, "noise_floor_end": float(np.mean(dn[-20:]))}


def fig_observables(D, names):
    line_style()
    feats = [("saturation", "Saturation"), ("brightness", "Brightness"), ("sharpness", "Sharpness"),
             ("flow_mag", "Motion (optical flow)")]
    fig, axs = plt.subplots(1, 4, figsize=(W, 1.75))
    for ax, (k, lab) in zip(axs, feats):
        j = names.index(k)
        for m in MODELS:
            X = np.array([v["low"][:178, j] for v in D[m].values()])
            if k == "flow_mag":
                X = np.array([np.convolve(x, np.ones(5) / 5, "same") for x in X])
            med = np.median(X, 0)
            lo, hi = np.percentile(X, 25, 0), np.percentile(X, 75, 0)
            t = np.arange(len(med))
            ax.plot(t, med, color=MCOL[m], label=MLABEL[m], marker=MMARK[m], markevery=30)
            ax.fill_between(t, lo, hi, color=MCOL[m], alpha=0.10, lw=0)
        ax.set_title(lab); ax.set_xlabel("Time (s)")
        ax.set_xticks([0, 60, 120, 180])
    axs[0].legend(loc="upper left", fontsize=5.8)
    fig.tight_layout(w_pad=0.8)
    save(fig, "line_observables")


def fig_phase_m1():
    """30 s V2V on 128 Panda clips: generators contract, real video does not; search delays it."""
    line_style()
    P = json.load(open(os.path.join(A, "results", "probes.json")))
    meth = [("real", "Real video (ground truth)", (0, 0, 0), "-"),
            ("notta", "Self Forcing", MCOL["sf"], "-"),
            ("rolling_notta", "Rolling Forcing", MCOL["rf"], "-"),
            ("sf_always_search", "Self Forcing + best-of-4 search", (0.75, 0, 0.75), "--")]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.0))
    for k, lab, col, ls in meth:
        sp = np.array(P["convergence"][k]["spread"])
        axs[0].plot(np.arange(len(sp)) / (len(sp) / 30), sp, color=col, ls=ls, label=lab)
    axs[0].set_title("(a) Ensemble spread across 128 clips")
    axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Mean pairwise distance")
    axs[0].legend(loc="lower left", fontsize=5.8)
    # saturation drift from per-clip observables
    for k, lab, col, ls in meth:
        fs = sorted(glob.glob(os.path.join(A, "obs", k, "*.npz")))
        rows = []
        for f in fs:
            d = np.load(f)
            names = [str(x) for x in d["low_names"]]
            s = np.nan_to_num(d["low"][:, names.index("saturation")])
            fps = float(d["fps"]) if "fps" in d.files else 16
            n = int(len(s) // fps)
            rows.append(s[: int(n * fps)].reshape(n, -1).mean(1)[:30] if n >= 30 else None)
        rows = np.array([r for r in rows if r is not None])
        axs[1].plot(np.arange(rows.shape[1]), np.median(rows, 0), color=col, ls=ls, label=lab)
    axs[1].set_title("(b) Colour saturation (median clip)")
    axs[1].set_xlabel("Time (s)"); axs[1].set_ylabel("Saturation")
    fig.tight_layout(w_pad=1.5)
    save(fig, "line_phase_m1")


def fig_twins():
    line_style()
    T = json.load(open(os.path.join(A, "results_long", "twins_dino.json")))
    J = json.load(open(os.path.join(A, "results_long", "twins.json")))
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.05), gridspec_kw={"width_ratios": [1.5, 1]})
    for m in MODELS:
        for e, ls in (("0.005", (0, (3, 2))), ("0.05", "-")):
            y = np.array(T[f"{m}_{e}"])
            t = np.arange(len(y))
            axs[0].plot(t, np.clip(y, 1e-4, None), color=MCOL[m], ls=ls,
                        label=f"{MLABEL[m]}, ε={e}" if True else None)
    axs[0].axvline(29, color="#777777", lw=0.6, ls=":")
    axs[0].text(30, 1.2e-4, "one noise block\nperturbed (29 s)", fontsize=6.0, color="#555555", va="bottom")
    axs[0].set_yscale("log"); axs[0].set_ylim(1e-4, 1.5)
    axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Twin distance (1 − cos, DINOv2)")
    axs[0].set_title("(a) A tiny nudge grows, then saturates")
    axs[0].legend(loc="lower right", fontsize=5.4, ncol=1)
    bar_style()
    ax = axs[1]
    bar_axes(ax)
    lam = [J[m]["0.05"]["lambda_per_s_first_7.5s"] for m in MODELS]
    end = [float(np.mean(T[f"{m}_0.05"][-10:])) for m in MODELS]
    x = np.arange(3)
    ax.bar(x - 0.19, lam, 0.36, color=DS_BLUE, hatch="////", edgecolor="white", lw=0, label="growth rate in latent space (1/s)")
    ax.bar(x + 0.19, end, 0.36, color=DS_TAN, edgecolor="white", lw=0, label="final semantic distance (DINOv2)")
    for i in range(3):
        ax.text(x[i] - 0.19, lam[i] + 0.01, f"{lam[i]:.2f}", ha="center", fontsize=6.0)
        ax.text(x[i] + 0.19, end[i] + 0.01, f"{end[i]:.2f}", ha="center", fontsize=6.0)
    ax.set_xticks(x); ax.set_xticklabels([MLABEL[m].replace(" ", "\n") for m in MODELS])
    ax.set_title("(b) Sensitive ≠ unstable", fontsize=7.6)
    ax.legend(loc="upper left", fontsize=5.8, frameon=False)
    ax.set_ylim(0, 0.62)
    fig.tight_layout(w_pad=1.5)
    save(fig, "line_twins")


def fig_bifurcation():
    line_style()
    B = json.load(open(os.path.join(A, "results_long", "bif.json")))
    lab = {"w21 s0 (native)": ("native (window 21, no sink)", (1, 0, 0), "o"),
           "w21 s3": ("+ sink 3", (0, 0, 1), "s"),
           "w12 s0": ("window 12", (0.75, 0, 0.75), "D"),
           "w12 s3": ("window 12 + sink 3", (0, 0.5, 0), "^")}
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.0))
    for k, (l, c, mk) in lab.items():
        axs[0].plot(B[k]["D_prompt"], color=c, marker=mk, markevery=20, label=l)
        axs[1].plot(B[k]["sim_to_own_opening"], color=c, marker=mk, markevery=20, label=l)
    axs[0].set_title("(a) Do different prompts merge?")
    axs[0].set_ylabel("Distance between prompts"); axs[0].set_xlabel("Time (s)")
    axs[1].set_title("(b) Does the video remember its opening?")
    axs[1].set_ylabel("Similarity to own opening"); axs[1].set_xlabel("Time (s)")
    axs[0].legend(loc="lower left", fontsize=5.8)
    fig.tight_layout(w_pad=1.5)
    save(fig, "line_bifurcation")


def fig_phase_portrait(D):
    """PCA of per-second DINOv2 states: each line is one 3-minute video."""
    line_style()
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.0))
    cmap = plt.get_cmap("tab10")
    allX = np.concatenate([v["cls"][:178] for m in MODELS for v in D[m].values()])
    mu = allX.mean(0)
    U, S, Vt = np.linalg.svd(allX - mu, full_matrices=False)
    V = Vt[:2].T
    lim = None
    for ax, m in zip(axs, MODELS):
        for (p, s), v in sorted(D[m].items()):
            if s != 0:
                continue
            Y = (v["cls"][:178] - mu) @ V
            Ys = np.array([np.convolve(Y[:, i], np.ones(9) / 9, "valid") for i in range(2)]).T
            ax.plot(Ys[:, 0], Ys[:, 1], color=cmap(p), lw=0.8, alpha=0.9)
            ax.plot(*Ys[0], "o", color=cmap(p), ms=3.2, mec="white", mew=0.4)
            ax.plot(*Ys[-1], "X", color=cmap(p), ms=4.2, mec="black", mew=0.4)
        ax.set_title(MLABEL[m])
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2" if m == "sf" else "")
    xs = [a.get_xlim() for a in axs]; ys = [a.get_ylim() for a in axs]
    for a in axs:
        a.set_xlim(min(x[0] for x in xs), max(x[1] for x in xs)); a.set_ylim(min(y[0] for y in ys), max(y[1] for y in ys))
    axs[0].plot([], [], "o", color="gray", ms=3.2, label="start"); axs[0].plot([], [], "X", color="gray", ms=4.2, mec="k", label="3 min")
    axs[0].legend(loc="lower left", fontsize=5.8)
    fig.tight_layout(w_pad=0.6)
    save(fig, "line_phase_portrait")


def fig_ews(D, names):
    import deep
    line_style()
    sf = D["sf"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.0), gridspec_kw={"width_ratios": [1.4, 1]})
    # (a) attractor proximity aligned at collapse time
    os.chdir(A)
    Dd, _ = deep.load()
    for k in sorted(Dd["sf"]):
        prox, _ = deep.attractor_proximity(Dd["sf"], k)
        tc = deep.collapse_time(prox)
        if tc is None:
            continue
        tt = np.arange(len(prox)) - tc
        axs[0].plot(tt, prox, color=(1, 0, 0), alpha=0.35, lw=0.7)
    axs[0].axvline(0, color="k", lw=0.6, ls=":")
    axs[0].axhline(0, color="#777777", lw=0.5)
    axs[0].set_xlim(-90, 90)
    axs[0].set_xlabel("Time relative to collapse (s)")
    axs[0].set_ylabel("Pull toward end state")
    axs[0].set_title("(a) Collapse is a slow ramp (16 SF videos)")
    bar_style()
    ax = axs[1]; bar_axes(ax)
    B = json.load(open(os.path.join(A, "results_long", "deep.json")))["B_early_warning"]
    sig = [("attractor_proximity", "proximity"), ("saturation", "saturation"), ("flow", "motion")]
    x = np.arange(len(sig))
    pre = [B[k]["frac_pre_ac1_rising"] * 100 for k, _ in sig]
    ctl = [B[k]["frac_ctrl_ac1_rising"] * 100 for k, _ in sig]
    ax.bar(x - 0.19, pre, 0.36, color=DS_BLUE, hatch="////", edgecolor="white", lw=0, label="before SF collapse")
    ax.bar(x + 0.19, ctl, 0.36, color=DS_GRAY, edgecolor="white", lw=0, label="matched window, RF/LL")
    ax.axhline(50, color="#777777", lw=0.6, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in sig])
    ax.set_ylabel("Rising autocorrelation (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(b) No early warning", fontsize=7.6)
    ax.legend(loc="upper right", fontsize=5.6, frameon=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "line_ews")


def fig_lol():
    """Synchronized snap-back test (LoL 2601.16914) with circular-shift surrogates."""
    line_style()
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.6), sharey=True)
    for ax, m in zip(axs, MODELS):
        R = []
        for f in sorted(glob.glob(os.path.join(A, "lobs", "main", m, "*.npz"))):
            if "_twin" in f:
                continue
            c = np.load(f)["cls"].astype(np.float32); c /= np.linalg.norm(c, axis=1, keepdims=True)
            o = c[:FPS].mean(0); o /= np.linalg.norm(o)
            s = c @ o
            R.append(np.clip(np.r_[np.zeros(8), s[8:] - s[:-8]], 0, None))
        L = min(map(len, R)); R = np.array([r[:L] for r in R]); R[:, :FPS * 10] = 0
        stat = lambda X: np.convolve(X.mean(0), np.ones(8) / 8, "same")[FPS * 10:].max()
        obs = stat(R)
        sur = np.array([stat(np.array([np.roll(r, rng.integers(L)) for r in R])) for _ in range(500)])
        ax.hist(sur, bins=30, color=DS_LBLUE, edgecolor="white", lw=0.3)
        ax.axvline(obs, color=(1, 0, 0), lw=1.0)
        ax.set_title(f"{MLABEL[m]}  (p = {np.mean(sur >= obs):.2f})")
        ax.set_xlabel("Peak synchronized snap-back")
    axs[0].set_ylabel("Surrogates")
    fig.tight_layout(w_pad=0.6)
    save(fig, "line_lol")


# =========================================================================== bars
def fig_rqa():
    bar_style()
    R = json.load(open(os.path.join(A, "results_long", "deep4.json")))["rqa"]
    fig, ax = plt.subplots(figsize=(W * 0.5, 1.9))
    bar_axes(ax)
    x = np.arange(3)
    a = [R[m]["DET"] for m in MODELS]; b = [R[m]["DET_sur"] for m in MODELS]
    ax.bar(x - 0.19, a, 0.36, color=DS_BLUE, hatch="////", edgecolor="white", lw=0, label="generated")
    ax.bar(x + 0.19, b, 0.36, color=DS_GRAY, edgecolor="white", lw=0, label="phase-randomized surrogate")
    for i in range(3):
        ax.text(x[i] - 0.19, a[i] + 0.01, f"{a[i]:.2f}", ha="center", fontsize=6.0)
        ax.text(x[i] + 0.19, b[i] + 0.01, f"{b[i]:.2f}", ha="center", fontsize=6.0)
    ax.set_xticks(x); ax.set_xticklabels([MLABEL[m] for m in MODELS])
    ax.set_ylabel("Determinism (RQA)"); ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.8, frameon=False)
    save(fig, "bar_rqa")


def fig_tta():
    bar_style()
    meth = ["No\nadaptation", "AdaSteer", "LoRA\nr8", "TinyLoRA\nbare", "TinyLoRA\ntied"]
    panda = [154.7, 153.4, 157.9, 154.2, 161.1]
    ucf = [85.7, 88.3, 88.6, 85.6, 86.7]
    cols = [DS_GRAY, DS_BLUE, DS_LBLUE, DS_TAN, DS_CREAM]
    hatches = [None, "////", None, None, None]
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.85))
    for ax, vals, title in ((axs[0], panda, "Panda-70M, 999 videos"), (axs[1], ucf, "UCF-101, 932 videos")):
        bar_axes(ax)
        x = np.arange(len(meth))
        for i in range(len(meth)):
            ax.bar(x[i], vals[i], 0.7, color=cols[i], hatch=hatches[i], edgecolor="white", lw=0)
            ax.text(x[i], vals[i] + 0.4, f"{vals[i]:.1f}", ha="center", fontsize=6.2)
        ax.set_ylim(min(vals) - 12, max(vals) + 4)
        ax.set_xticks(x); ax.set_xticklabels(meth, fontsize=6.0)
        ax.set_title(title, fontsize=7.4)
    axs[0].set_ylabel("FVD (lower is better)")
    fig.tight_layout(w_pad=1.5)
    save(fig, "bar_tta_null")


def fig_atlas():
    """ΔIQ vs the do-nothing host for every Wan family with a valid number (atlas rows 12–31,
    excluding n=2 smokes). Edits spread to catastrophic losses; selection stays near zero."""
    bar_style()
    edit = [("AdaSteer on Wan (fixed)", -27.95), ("AdaSteer on Wan (stream)", -19.14), ("AdaSteer on Wan (resid)", -52.87),
            ("intra redraw", -2.43), ("lastmix", -0.99), ("nudge", -0.96), ("next-seed", -0.51), ("wiggle", -2.51),
            ("latent-motion pick", -0.87), ("SF sampler mix", -0.11), ("RF sampler mix", 1.60),
            ("schedule: linger-high", -0.36), ("schedule: dump-early", 1.44), ("leftover ρ lo", 1.39),
            ("leftover ρ hi", -2.26), ("leftover adapt", 0.77), ("leftover look", 2.81), ("KV context noise (RF)", 0.9),
            ("KV context noise (SF)", -1.4), ("FIFO lookahead", 1.53), ("FIFO sick-only", 0.05), ("extra sink", 0.33),
            ("noise warp (always)", -21.44), ("noise warp (live)", -16.20), ("pred-slide", -3.81),
            ("pred-slide persist", -36.66), ("fast-weight write", -16.8), ("KV window", -0.49)]
    sel = [("always-search", 0.12), ("pseudo-future search", 0.31), ("rewind", -0.65), ("sick-search", -0.52),
           ("prefix-match", -0.77), ("freeze-score RF", -0.80), ("freeze-score SF", 0.02), ("CachedSearch", -0.79),
           ("re-gate", -0.85), ("I2V best-of-4 always", 0.04), ("I2V best-of-4 gated", -0.05)]
    fig, ax = plt.subplots(figsize=(W, 1.75))
    rng = np.random.default_rng(3)
    for y, data, col, mk in ((1, edit, DS_BLUE, "o"), (0, sel, DS_TAN, "D")):
        v = np.array([d for _, d in data])
        jit = rng.uniform(-0.17, 0.17, len(v))
        ax.scatter(v, y + jit, s=16, color=col, edgecolor="#555555", lw=0.3, marker=mk, zorder=3)
    ax.annotate("", xy=(-55, 1.28), xytext=(-15, 1.28), arrowprops=dict(arrowstyle="-", lw=0.6, color="#888888"))
    ax.text(-29, 1.33, "broke the video: AdaSteer on Wan, noise warp,\npred-slide persist, fast-weight write",
            ha="center", va="bottom", fontsize=5.8, color=DS_TEXT)
    ax.text(0, -0.42, "|ΔIQ| < 1", ha="center", fontsize=5.8, color="#666666")
    ax.axvline(0, color="#555555", lw=0.6)
    ax.axvspan(-1, 1, color="#EEEEEE", zorder=0)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([-50, -20, -10, -5, -2, -1, 0, 1, 2]); ax.set_xticklabels(["−50", "−20", "−10", "−5", "−2", "−1", "0", "+1", "+2"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Select among\nown outputs", "Edit the\ntrajectory"])
    ax.set_ylim(-0.5, 1.75)
    ax.set_xlim(-70, 4)
    ax.set_xlabel("Δ imaging quality vs. doing nothing (VBench, 0–100)")
    ax.xaxis.grid(True, ls=(0, (4, 3)), color="#E8E8E8", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "dot_atlas")


def main():
    D, names = load_long()
    res = fig_convergence(D)
    hl = fig_halflife(res)
    print("half-life fit", hl)
    fig_observables(D, names)
    fig_phase_portrait(D)
    fig_twins()
    fig_bifurcation()
    fig_rqa()
    fig_tta()
    fig_atlas()
    fig_lol()
    fig_phase_m1()
    fig_ews(D, names)
    if os.path.isdir(FR) and os.listdir(FR):
        fig_photo_grids()
        fig_end_states()
        fig_openings()
    json.dump({"halflife": hl, "retrieval_end": {m: float(res[m][2][-10:].mean()) for m in MODELS},
               "dprompt_start_end": {m: [float(res[m][0][:3].mean()), float(res[m][0][-5:].mean())] for m in MODELS},
               "dnoise_end": {m: float(res[m][1][-5:].mean()) for m in MODELS}},
              open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
