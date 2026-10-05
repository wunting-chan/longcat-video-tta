#!/usr/bin/env python3
"""Sponsor-safe charts for the summer 2026 progress note.

Styled after the DeepSeek technical reports: headline row in a saturated
blue with white hatching, baselines in slate greys, value labels on bars,
light dashed y-grid, legend in a single row above the axes.

Best-of-N seed search is a known sampling practice. The gated variant
(`sf_pseudo`) is plotted by outcome only, as "Gated seed search"; do not
put internal arm names or the gate's mechanism in labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Patch

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "sweep_experiment/reports/briefing_charts_raw/2026-09-09_slim"
CITE128 = RAW / "wan/cite128_per_video.csv"
TTA1000 = RAW / "longcat_1000v"
AR = RAW / "longcat_ar"
GATE = RAW / "longcat_gate"
OUT = ROOT / "sweep_experiment/reports/paper_tables/sponsor_summer_2026_figures"

BLUE = "#4D6BFE"
BLUE_DK = "#2B3FB8"
SLATE = "#8A94A8"
SLATE_LT = "#C5CCD9"
INK = "#1F2329"
MUTE = "#6B7280"
GRID = "#E3E6EC"
RED = "#E0685A"

SYSTEMS = [
    ("notta", "Published few-step baseline", SLATE_LT, None),
    ("rolling_notta", "Published streaming baseline", SLATE, None),
    ("sf_always_search", "Best-of-N seed search", BLUE_DK, None),
    ("sf_pseudo", "Gated seed search", BLUE, "///"),
]
HERO = {"sf_always_search", "sf_pseudo"}
SHORT = {
    "notta": "Few-step",
    "rolling_notta": "Streaming",
    "sf_always_search": "Best-of-N",
    "sf_pseudo": "Gated",
}
WALL_S = {"notta": 108, "rolling_notta": 47, "sf_always_search": 354, "sf_pseudo": 294}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "hatch.color": "white",
            "hatch.linewidth": 1.1,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / name}.png/.pdf")


def _bar_labels(ax, bars, fmt: str, bold_max: bool = True, pad: float = 0.6) -> None:
    heights = [b.get_height() for b in bars]
    top = max(heights)
    for b, h in zip(bars, heights):
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + pad,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold" if (bold_max and h == top) else "normal",
        )


def _cite128() -> pd.DataFrame:
    d = pd.read_csv(CITE128)
    return d[d.method.isin([s[0] for s in SYSTEMS])].copy()


# ---------------------------------------------------------------- overview


def fig_overview(d: pd.DataFrame) -> None:
    dims = [
        ("subject_consistency", "Subject\nconsistency", 100),
        ("background_consistency", "Background\nconsistency", 100),
        ("aesthetic_quality", "Aesthetic\nquality", 100),
        ("imaging_quality", "Imaging\nquality", 1),
        ("motion_smoothness", "Motion\nsmoothness", 100),
        ("temporal_flickering", "Temporal\nflickering", 100),
        ("dynamic_degree", "Dynamic degree\n(% dynamic clips)", 100),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 4.3))
    width = 0.2
    x = np.arange(len(dims))
    for k, (m, label, color, hatch) in enumerate(SYSTEMS):
        sub = d[d.method == m]
        vals = []
        for col, _, scale in dims:
            v = sub[col].mean() if col == "dynamic_degree" else sub[col].median()
            vals.append(v * scale)
        bars = ax.bar(
            x + (k - (len(SYSTEMS) - 1) / 2) * width,
            vals,
            width,
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.6,
            label=label,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.8,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.3,
                color=BLUE_DK if m in HERO else INK,
                fontweight="bold" if m in HERO else "normal",
            )
    ax.set_xticks(x, [d_[1] for d_ in dims])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Score (%)")
    ax.legend(ncol=len(SYSTEMS), loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.axvspan(5.5, 6.5, color=BLUE, alpha=0.05, zorder=0)
    _save(fig, "fig01_overview_vbench128")


# ------------------------------------------------------- parameter space


def _psnr_1000(method: str) -> pd.Series:
    d = pd.read_csv(TTA1000 / method / "vbench_per_video.csv")
    d = d[d.video_id.str.contains("PSNR-")]
    idx = d.video_id.str.extract(r"^(\d+)_")[0].astype(int)
    psnr = d.video_id.str.extract(r"PSNR-([\d.]+)")[0].astype(float)
    return pd.Series(psnr.values, index=idx.values).groupby(level=0).first()


def fig_param_tta() -> None:
    base = _psnr_1000("NOTTA")
    ada = _psnr_1000("ADA")
    j = pd.concat([base, ada], axis=1, keys=["b", "a"]).dropna()
    delta = (j.a - j.b).values

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(11.0, 3.9), gridspec_kw={"width_ratios": [1, 1.6]}
    )

    labels = ["Do nothing", "Always-on\nparameter TTA", "Hindsight\nskip (oracle)"]
    vals = [17.930, 17.938, 18.123]
    colors = [SLATE_LT, BLUE, "white"]
    bars = ax0.bar(labels, vals, color=colors, width=0.6, edgecolor=[SLATE_LT, "white", BLUE_DK])
    bars[1].set_hatch("///")
    bars[2].set_hatch("..")
    bars[2].set_linewidth(1.2)
    for b, v in zip(bars, vals):
        ax0.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", fontsize=9)
    ax0.set_ylim(17.8, 18.2)
    ax0.set_ylabel("Mean PSNR (dB)")
    ax0.set_title("(a) Mean over 1,000 videos", loc="left")
    ax0.annotate(
        "+0.19 dB headroom\nneeds a skip rule",
        xy=(2, 18.123),
        xytext=(1.0, 18.16),
        fontsize=8.5,
        color=BLUE_DK,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=BLUE_DK, lw=0.9),
    )

    bins = np.linspace(-1.5, 1.5, 61)
    clipped = np.clip(delta, -1.49, 1.49)
    n, edges, patches = ax1.hist(clipped, bins=bins, color=SLATE_LT, edgecolor="white", lw=0.4)
    for p, left in zip(patches, edges[:-1]):
        if left >= 0.1:
            p.set_facecolor(BLUE)
        elif left < -0.1:
            p.set_facecolor(RED)
    ax1.axvline(0, color=INK, lw=0.8)
    win = (delta > 0.1).mean() * 100
    lose = (delta < -0.1).mean() * 100
    ymax = n.max()
    ax1.text(0.75, ymax * 0.8, f"{win:.0f}% gain\n> 0.1 dB", color=BLUE_DK, ha="center", fontsize=9)
    ax1.text(-0.75, ymax * 0.8, f"{lose:.0f}% lose\n> 0.1 dB", color=RED, ha="center", fontsize=9)
    ax1.set_xlabel("Per-video ΔPSNR vs. do nothing (dB, clipped at ±1.5)")
    ax1.set_ylabel("Videos")
    ax1.set_title(f"(b) Per-video change, N={len(delta)}  ·  mean {delta.mean():+.3f} dB", loc="left")
    fig.tight_layout(w_pad=3)
    _save(fig, "fig02_param_tta_mean_and_spread")


def fig_surprise() -> None:
    q = ["Q1\nleast surprising", "Q2", "Q3", "Q4", "Q5\nmost surprising"]
    d = [0.112, 0.069, -0.001, -0.012, -0.130]
    fig, (ax, ax1) = plt.subplots(
        1, 2, figsize=(11.0, 3.9), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    colors = [BLUE if v > 0.02 else (RED if v < -0.02 else SLATE_LT) for v in d]
    bars = ax.bar(q, d, color=colors, width=0.62, edgecolor="white")
    for b in bars:
        if b.get_facecolor()[:3] == plt.matplotlib.colors.to_rgb(BLUE):
            b.set_hatch("///")
    ax.axhline(0, color=INK, lw=0.8)
    for i, v in enumerate(d):
        ax.text(i, v + (0.008 if v >= 0 else -0.02), f"{v:+.3f}", ha="center", fontsize=9)
    ax.set_ylim(-0.17, 0.15)
    ax.set_ylabel("Mean ΔPSNR (dB)")
    ax.set_xlabel("Denoising loss on the context frames (quintile, ≈200 videos each)")
    ax.set_title("(a) Gain by model surprise", loc="left")
    ax.text(
        4.35, 0.11, "Videos with the highest loss\ndegrade the most",
        ha="right", fontsize=8.5, color=MUTE,
    )

    routers = []
    for exp, label in [
        ("video_caption_only", "Pixel & caption statistics (9)"),
        ("diffusion_ood_only", "Denoising loss (20)"),
        ("video_caption_ood", "Pixel, caption & loss (29)"),
        ("vae_inference_embedding", "Compressed-video latents (130)"),
        ("video_caption_ood_vae", "All features (159)"),
    ]:
        row = json.loads((GATE / "deploy_strict_router" / f"{exp}.json").read_text())["row"]
        routers.append((label, row["captured_pct"], "VBench"))
    psnr = json.loads((GATE / "deploy_psnr_router" / "results.json").read_text())
    routers.append(
        ("Pixel & caption statistics (9)", 100 * psnr["psnr_policy"]["fraction_oracle_captured"], "PSNR")
    )
    y = np.arange(len(routers))
    vals = [r[1] for r in routers]
    cols = [BLUE if v > 0 else RED for v in vals]
    hb = ax1.barh(y, vals, color=cols, height=0.6, edgecolor="white")
    for b, v in zip(hb, vals):
        if v > 0:
            b.set_hatch("///")
        ax1.text(v + (0.3 if v >= 0 else -0.3), b.get_y() + b.get_height() / 2, f"{v:+.1f}%",
                 va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax1.set_yticks(y, [f"{r[0]}\n{r[2]} objective" for r in routers], fontsize=8)
    ax1.invert_yaxis()
    ax1.axvline(0, color=INK, lw=0.8)
    ax1.set_xlim(-11, 7)
    ax1.grid(axis="y", visible=False)
    ax1.grid(axis="x", color=GRID, ls="--", lw=0.7)
    ax1.set_xlabel("Oracle gain recovered (%; 0 = fixed setting, 100 = oracle)")
    ax1.set_title("(b) Learned per-video routers", loc="left")
    fig.tight_layout(w_pad=2.5)
    _save(fig, "fig03_param_tta_surprise")


# ------------------------------------------------------------ long horizon


def fig_drift() -> None:
    pv = pd.read_csv(AR / "longhorizon_sweep_delta_stream_clean_native_12ch/per_video/per_video_per_chunk.csv")
    signals = [
        ("sharpness", "Sharpness"),
        ("temporal_motion", "Temporal motion"),
        ("contrast", "Contrast"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharex=True)
    for ax, (sig, title) in zip(axes, signals):
        s = pv[pv.signal == sig].pivot(index="video", columns="chunk", values="NOTTA")
        s = s.sort_index(axis=1)
        # One shared reference (the chunk-1 mean over videos) so the bold line is
        # exactly the pooled drift quoted in the report.
        pct = 100.0 * (s / s[s.columns[0]].mean() - 1.0)
        chunks = pct.columns.values
        for _, row in pct.iterrows():
            ax.plot(chunks, row.values, color=SLATE_LT, lw=0.9, alpha=0.9)
        q25, mean, q75 = pct.quantile(0.25), pct.mean(), pct.quantile(0.75)
        ax.fill_between(chunks, q25, q75, color=BLUE, alpha=0.15, lw=0)
        ax.plot(chunks, mean, color=BLUE, lw=2.2, marker="o", ms=3.5)
        ax.axhline(0, color=INK, lw=0.8)
        ups = int((s[s.columns[-1]] > s[s.columns[0]]).sum())
        verb = "rose" if mean.values[-1] > 0 else "fell"
        ax.set_title(title, loc="left")
        ax.text(0.02, 0.97, f"{verb} in {ups if verb == 'rose' else len(s) - ups} of {len(s)} videos",
                transform=ax.transAxes, fontsize=8.5, color=MUTE, va="top")
        ax.set_xticks([1, 3, 6, 9, 12])
        ax.set_xlabel("Chunk (~5 s each)")
        last = mean.values[-1]
        ax.text(chunks[-1] + 0.25, last, f"{last:+.0f}%", color=BLUE_DK, fontsize=9.5,
                fontweight="bold", va="center")
        ax.set_xlim(0.6, 13.8)
        lo, hi = np.nanpercentile(pct.values, [3, 97])
        pad = 0.15 * (hi - lo)
        ax.set_ylim(min(lo, -5) - pad, hi + pad)
    axes[0].set_ylabel("Change vs. first-chunk mean (%)")
    handles = [
        plt.Line2D([], [], color=SLATE_LT, lw=1, label="individual video (N=8)"),
        plt.Line2D([], [], color=BLUE, lw=2.2, marker="o", ms=3.5, label="mean over videos"),
        Patch(color=BLUE, alpha=0.15, label="inter-quartile range"),
    ]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=2)
    _save(fig, "fig05_long_horizon_drift")


def fig_param_long_null() -> None:
    variants = [
        ("longhorizon_sweep_delta_stream_native_12ch", "Update variant 1", SLATE),
        ("longhorizon_sweep_delta_stream_clean_native_12ch", "Update variant 2", BLUE),
    ]
    names = {
        "sharpness": "Sharpness",
        "temporal_motion": "Temporal motion",
        "colorfulness": "Colorfulness",
        "contrast": "Contrast",
        "psnr": "PSNR",
        "ssim": "SSIM",
        "lpips": "LPIPS",
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, ls="--", lw=0.7)
    order = list(names)
    pmin = 1.0
    for k, (series, label, color) in enumerate(variants):
        rows = {r["signal"]: r for r in json.loads((AR / series / "paired/paired_stats.json").read_text())["rows"]}
        for i, sig in enumerate(order):
            r = rows[sig]
            base = r["mean_abs_drift_A"]
            mid = 100 * r["mean_reduction_A_minus_B"] / base
            lo, hi = (100 * c / base for c in r["ci95"])
            y = i + (k - 0.5) * 0.28
            ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="round")
            ax.plot(mid, y, "o", color=color, ms=6, mec="white", mew=0.8)
            pmin = min(pmin, r["signflip_p"])
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(range(len(order)), [names[s] for s in order])
    ax.invert_yaxis()
    ax.set_xlabel("Drift reduction vs. no update (% of baseline drift, 95% CI)")
    ax.text(0.99, 0.02, f"all paired sign-flip p ≥ {pmin:.2f}", transform=ax.transAxes,
            ha="right", fontsize=8.5, color=MUTE)
    ax.text(0.02, 1.01, "← update makes drift worse", transform=ax.transAxes, fontsize=8, color=MUTE)
    ax.text(0.98, 1.01, "update reduces drift →", transform=ax.transAxes, fontsize=8, color=MUTE, ha="right")
    ax.legend(
        handles=[plt.Line2D([], [], color=c, lw=2.2, marker="o", label=l) for _, l, c in variants],
        loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2,
    )
    _save(fig, "fig04_param_update_long_horizon_null")


# ------------------------------------------------------------ 128-clip table


def fig_frontier(d: pd.DataFrame) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
    for m, label, color, hatch in SYSTEMS:
        sub = d[d.method == m]
        dyn = 100 * sub.dynamic_degree.mean()
        iq = sub.imaging_quality.median()
        marker = "*" if m == "sf_pseudo" else ("D" if m == "sf_always_search" else "o")
        ax0.scatter(WALL_S[m], dyn, s={"*": 420, "D": 110, "o": 170}[marker], marker=marker,
                    color=color, edgecolor=BLUE_DK if m in HERO else SLATE, lw=1, zorder=3)
        offset, ha, va = {"notta": ((10, -4), "left", "top"),
                          "rolling_notta": ((10, -4), "left", "top"),
                          "sf_always_search": ((8, -10), "left", "top"),
                          "sf_pseudo": ((-18, 4), "right", "bottom")}[m]
        ax0.annotate(f"{SHORT[m]}\n{dyn:.1f}% dynamic · IQ {iq:.1f}", (WALL_S[m], dyn),
                     textcoords="offset points", xytext=offset, ha=ha, va=va, fontsize=8.5,
                     fontweight="bold" if m == "sf_pseudo" else "normal")
    ax0.add_patch(FancyArrowPatch((114, 33.6), (278, 47.0), arrowstyle="-|>", mutation_scale=12,
                                  color=BLUE, lw=1.2, ls="--", connectionstyle="arc3,rad=-0.25"))
    ax0.text(175, 36.5, "+15 pts dynamic\nhighest imaging quality", color=BLUE_DK, fontsize=8.5, ha="center")
    ax0.set_xscale("log")
    ax0.set_xticks([40, 60, 100, 200, 400], ["40", "60", "100", "200", "400"])
    ax0.set_xlim(35, 480)
    ax0.set_ylim(22, 56)
    ax0.set_xlabel("Generation time per 30 s clip (s, log scale)")
    ax0.set_ylabel("Dynamic clips (% of 128)")
    ax0.set_title("(a) Motion versus cost", loc="left")
    ax0.grid(axis="x", color=GRID, ls="--", lw=0.7)

    x = np.arange(len(SYSTEMS))
    for k, (m, label, color, hatch) in enumerate(SYSTEMS):
        b = ax1.bar(k, WALL_S[m], color=color, hatch=hatch, edgecolor="white", width=0.6)
        ax1.text(k, WALL_S[m] + 6, f"{WALL_S[m]} s", ha="center", fontsize=9,
                 fontweight="bold" if m == "sf_pseudo" else "normal")
    ax1.set_xticks(x, [SHORT[s[0]] for s in SYSTEMS])
    ax1.set_ylabel("Seconds per clip")
    ax1.set_ylim(0, 400)
    ax1.set_title("(b) Generation time", loc="left")
    fig.tight_layout(w_pad=3)
    _save(fig, "fig06_quality_cost_frontier")


def fig_transitions(d: pd.DataFrame) -> None:
    p = d.pivot(index="clip", columns="method", values="dynamic_degree")
    b, s = p["notta"].astype(int), p["sf_always_search"].astype(int)
    mat = np.array([[((b == i) & (s == j)).sum() for j in (0, 1)] for i in (0, 1)])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.grid(False)
    cell = [["#EEF0F4", BLUE], ["#F7D9D4", SLATE_LT]]
    rgb = np.array([[plt.matplotlib.colors.to_rgb(c) for c in row] for row in cell])
    ax.imshow(rgb)
    for x in (0.5,):
        ax.axvline(x, color="white", lw=3)
        ax.axhline(x, color="white", lw=3)
    names = [["Stayed static", "Became dynamic"], ["Became static", "Stayed dynamic"]]
    for i in range(2):
        for j in range(2):
            v = mat[i, j]
            c = "white" if (i, j) == (0, 1) else INK
            ax.text(j, i - 0.08, f"{v}", ha="center", va="center", fontsize=20, fontweight="bold", color=c)
            ax.text(j, i + 0.22, names[i][j], ha="center", va="center", fontsize=9, color=c)
    ax.set_xticks([0, 1], ["static", "dynamic"])
    ax.set_yticks([0, 1], ["static", "dynamic"])
    ax.set_xlabel("Best-of-N seed search")
    ax.set_ylabel("Published few-step baseline")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Per-clip Dynamic Degree, N={len(p)}  ·  net +{mat[0,1]-mat[1,0]}", loc="left")
    _save(fig, "fig07_clip_transition_matrix")


def fig_distributions(d: pd.DataFrame) -> None:
    metrics = [
        ("imaging_quality", "Imaging quality"),
        ("subject_consistency", "Subject consistency"),
        ("temporal_flickering", "Temporal flickering (higher = steadier)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    for ax, (col, title) in zip(axes, metrics):
        data = [d[d.method == m][col].values for m, *_ in SYSTEMS]
        if col == "temporal_flickering":
            lo = min(np.percentile(x, 2) for x in data)
            data = [np.clip(x, lo, None) for x in data]
        parts = ax.violinplot(data, showextrema=False, widths=0.8)
        for body, (m, _, color, _) in zip(parts["bodies"], SYSTEMS):
            body.set_facecolor(color)
            body.set_edgecolor(BLUE_DK if m in HERO else SLATE)
            body.set_alpha(0.85)
        for k, x in enumerate(data, start=1):
            q1, med, q3 = np.percentile(x, [25, 50, 75])
            ax.plot([k, k], [q1, q3], color=INK, lw=3, solid_capstyle="butt")
            ax.plot(k, med, "o", color="white", mec=INK, ms=5, zorder=4)
            ax.annotate(f"{med:.1f}" if col == "imaging_quality" else f"{med:.3f}", (k, q3),
                        textcoords="offset points", xytext=(0, 3), ha="center", va="bottom", fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=5)
        ax.set_xticks(range(1, len(SYSTEMS) + 1), [SHORT[m] for m, *_ in SYSTEMS])
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Per-clip score (N=128)")
    fig.tight_layout(w_pad=2)
    _save(fig, "fig08_per_clip_distributions")


def fig_paired(d: pd.DataFrame) -> None:
    p = d.pivot(index="clip", columns="method")
    iq_b, iq_s = p["imaging_quality"]["notta"], p["imaging_quality"]["sf_always_search"]
    fl_b, fl_s = p["temporal_flickering"]["notta"], p["temporal_flickering"]["sf_always_search"]
    dy_b, dy_s = p["dynamic_degree"]["notta"], p["dynamic_degree"]["sf_always_search"]
    woke = (dy_b == 0) & (dy_s == 1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    lim = (40, 82)
    ax0.plot(lim, lim, color=MUTE, lw=0.8, ls="--")
    ax0.scatter(iq_b[~woke], iq_s[~woke], s=16, color=SLATE_LT, edgecolor=SLATE, lw=0.4, label="other clips")
    ax0.scatter(iq_b[woke], iq_s[woke], s=34, color=BLUE, edgecolor="white", lw=0.6, label="became dynamic (25)")
    ax0.set_xlim(*lim)
    ax0.set_ylim(*lim)
    ax0.set_aspect("equal")
    ax0.grid(axis="x", color=GRID, ls="--", lw=0.7)
    ax0.set_xlabel("Imaging quality, few-step baseline")
    ax0.set_ylabel("Imaging quality, Best-of-N seed search")
    ax0.set_title("(a) Picture quality, clip by clip", loc="left")
    ax0.legend(loc="upper left")

    dfl = (fl_s - fl_b).values
    diq = (iq_s - iq_b).values
    ax1.axhline(0, color=INK, lw=0.8)
    ax1.axvline(0, color=INK, lw=0.8)
    ax1.axvspan(-0.30, -0.02, color=RED, alpha=0.06, lw=0)
    ax1.scatter(dfl[~woke.values], diq[~woke.values], s=16, color=SLATE_LT, edgecolor=SLATE, lw=0.4)
    ax1.scatter(dfl[woke.values], diq[woke.values], s=34, color=BLUE, edgecolor="white", lw=0.6)
    twitch = int(((fl_s - fl_b) < -0.02)[woke].sum())
    ax1.text(-0.235, 9, f"flicker region\n{twitch} of 25 newly dynamic\nclips fall here", color=RED, fontsize=8.5)
    ax1.set_xlim(-0.25, 0.03)
    ax1.set_xlabel("Δ temporal flickering (negative = more flicker)")
    ax1.set_ylabel("Δ imaging quality")
    ax1.set_title("(b) Dynamic labels attributable to flicker", loc="left")
    ax1.grid(axis="x", color=GRID, ls="--", lw=0.7)
    fig.tight_layout(w_pad=3)
    _save(fig, "fig09_paired_quality_and_flicker")


# ------------------------------------------------------- late September veto


def fig_store_veto() -> None:
    arms = [
        ("Same student,\nno extra store", SLATE_LT, None, dict(iq=72.96, subj=0.897, flick=0.981, dyn=2)),
        ("Evict old frames,\nno store write", SLATE, None, dict(iq=72.46, subj=0.887, flick=0.981, dyn=1)),
        ("Test-time\nstore write", BLUE, "///", dict(iq=56.14, subj=0.562, flick=0.903, dyn=8)),
    ]
    ranges = dict(iq=(56.04, 56.29), subj=(0.558, 0.565), flick=(0.897, 0.908))
    panels = [
        ("iq", "Imaging quality", "{:.1f}", (40, 80)),
        ("subj", "Subject consistency", "{:.2f}", (0.4, 1.0)),
        ("flick", "Temporal flickering", "{:.3f}", (0.85, 1.0)),
        ("dyn", "Clips labelled dynamic (of 8)", "{:.0f}", (0, 9)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.4))
    for ax, (key, title, fmt, ylim) in zip(axes, panels):
        for k, (label, color, hatch, v) in enumerate(arms):
            ax.bar(k, v[key], color=color, hatch=hatch, edgecolor="white", width=0.62)
            if key in ranges and hatch:
                lo, hi = ranges[key]
                ax.errorbar(k, v[key], yerr=[[v[key] - lo], [hi - v[key]]], color=INK, capsize=3, lw=1)
            ax.text(k, v[key] + (ylim[1] - ylim[0]) * 0.02, fmt.format(v[key]), ha="center", fontsize=8.5)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(3), ["no store", "evict only", "store write"], fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10)
    axes[3].text(2, 4, "flicker,\nnot motion", ha="center", va="center", fontsize=8.5,
                 color="white", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc=RED, ec="none"))
    fig.legend(
        handles=[Patch(facecolor=c, hatch=h, edgecolor="white", label=l.replace("\n", " ")) for l, c, h, _ in arms],
        ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.1),
    )
    fig.text(0.5, -0.04, "Text-to-video, 8 clips, 30 s, full-clip VBench. Whisker = range over the write rules tried.",
             ha="center", fontsize=8.5, color=MUTE)
    fig.tight_layout(w_pad=2)
    _save(fig, "fig10_session_store_veto")


def main() -> None:
    _style()
    d = _cite128()
    fig_overview(d)
    fig_param_tta()
    fig_surprise()
    fig_param_long_null()
    fig_drift()
    fig_frontier(d)
    fig_transitions(d)
    fig_distributions(d)
    fig_paired(d)
    fig_store_veto()
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
