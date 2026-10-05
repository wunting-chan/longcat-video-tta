#!/usr/bin/env python3
"""PI briefing charts: wins and tradeoffs from the slim 2026-09-09 pack."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SLIM = ROOT / "sweep_experiment/reports/briefing_charts_raw/2026-09-09_slim"
GAINS = ROOT / "sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv"
CORR = ROOT / "sweep_experiment/reports/per_video_analysis/2026-06-09/criteria_correlation_full.csv"
OUT = ROOT / "sweep_experiment/reports/briefing_charts_raw/figures_formal"
HEADLINE = ROOT / "sweep_experiment/reports/paper_tables/2026-06-08_headline_1000v.md"

C = {
    "ink": "#1d1d1f",
    "mute": "#6e6e73",
    "grid": "#e5e5ea",
    "win": "#0a7d32",
    "lose": "#c41e3a",
    "base": "#8e8e93",
    "ada": "#2f6fed",
    "lora": "#7d3cff",
    "search": "#0a7d32",
    "cost": "#c45c00",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "medium",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": C["ink"],
            "axes.labelcolor": C["ink"],
            "text.color": C["ink"],
            "xtick.color": C["ink"],
            "ytick.color": C["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str):
    v = row.get(key, "")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def chart_dpsnr_hist() -> None:
    rows = _read_csv(GAINS)
    ada = np.array([_f(r, "ADA_dpsnr") for r in rows], dtype=float)
    lora = np.array([_f(r, "LORA_R8_TTA_dpsnr") for r in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    for ax, vals, title, color in (
        (axes[0], ada, "AdaSteer vs. baseline (no-TTA)", C["ada"]),
        (axes[1], lora, "LoRA-r8 vs. baseline (no-TTA)", C["lora"]),
    ):
        vals = vals[np.isfinite(vals)]
        win = (vals > 0.1).mean() * 100
        lose = (vals < -0.1).mean() * 100
        clipped = vals[(vals >= -2.5) & (vals <= 2.5)]
        ax.hist(clipped, bins=40, color=color, alpha=0.85, edgecolor="white")
        ax.set_xlim(-2.5, 2.5)
        ax.axvline(0, color=C["ink"], lw=1.2)
        ax.axvline(np.mean(vals), color=C["lose"], ls="--", lw=1.2, label=f"mean {np.mean(vals):+.3f} dB")
        ax.set_title(title)
        ax.set_xlabel("ΔPSNR (dB), clipped ±2.5")
        ax.text(
            0.98,
            0.96,
            f"win >0.1 dB: {win:.0f}%\nlose <−0.1 dB: {lose:.0f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=C["ink"],
        )
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("Number of videos (N=999)")
    fig.suptitle("Per-video ΔPSNR versus baseline (no-TTA)", y=1.04)
    _save(fig, "01_dpsnr_vs_baseline.png")


def chart_oracle() -> None:
    rows = _read_csv(GAINS)
    base = np.array([_f(r, "NOTTA_psnr") for r in rows], dtype=float)
    ada = np.array([_f(r, "ADA_psnr") for r in rows], dtype=float)
    lora = np.array([_f(r, "LORA_R8_TTA_psnr") for r in rows], dtype=float)
    two = np.maximum(base, ada)
    three = np.maximum(two, lora)
    psnr = [np.mean(base), np.mean(ada), np.mean(two), np.mean(three)]
    fvd = [155.94, 156.22, None, 149.57]
    labels = ["Baseline (no-TTA)", "Always-on AdaSteer", "2-way oracle", "3-way oracle"]
    colors = [C["base"], C["ada"], C["win"], C["win"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    axes[0].bar(labels, psnr, color=colors)
    axes[0].set_ylabel("mean PSNR (dB)")
    axes[0].set_ylim(17.7, 18.3)
    for i, v in enumerate(psnr):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    axes[0].set_title("Mean PSNR")
    axes[0].tick_params(axis="x", rotation=15)
    fv, fl, fc = [], [], []
    for lab, val, col in zip(labels, fvd, colors):
        if val is not None:
            fv.append(val)
            fl.append(lab)
            fc.append(col)
    axes[1].bar(fl, fv, color=fc)
    axes[1].set_ylabel("FVD ↓")
    axes[1].set_ylim(145, 160)
    axes[1].invert_yaxis()
    for i, v in enumerate(fv):
        axes[1].text(i, v - 0.35, f"{v:.1f}", ha="center", fontsize=10)
    axes[1].set_title("FVD (lower is better)")
    axes[1].tick_params(axis="x", rotation=15)
    fig.suptitle(
        "Oracle selection versus always-on AdaSteer and baseline (no-TTA)",
        y=1.04,
    )
    _save(fig, "02_oracle_vs_baseline.png")


def chart_lora_trade() -> None:
    # Headline table (pack summaries omit some LoRA VBench dims).
    dims = ["Aesthetic", "Dynamic", "Imaging Q", "Subject"]
    notta = [0.395, 0.565, 0.649, 0.907]
    ada = [0.396, 0.568, 0.649, 0.907]
    lora = [0.442, 0.596, 0.615, 0.902]
    x = np.arange(len(dims))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(x - w, notta, w, label="Baseline (no-TTA)", color=C["base"])
    ax.bar(x, ada, w, label="AdaSteer", color=C["ada"])
    ax.bar(x + w, lora, w, label="LoRA-r8", color=C["lora"])
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.set_ylim(0.35, 1.05)
    for xs, vals in ((x - w, notta), (x, ada), (x + w, lora)):
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.legend(frameon=False)
    ax.set_title("VBench++ population scores by method")
    ax.set_ylabel("VBench++ (population mean)")
    _save(fig, "03_vbench_by_method_scores.png")


def chart_fvd_methods() -> None:
    # Same I3D suite as the oracle chart (fvd_summary.json), not the
    # rounded headline merged_summary (154.7 / 153.4 / 157.9).
    data = json.loads((SLIM / "longcat_oracle/fvd_summary.json").read_text())
    labels = ["Baseline (no-TTA)", "AdaSteer", "LoRA-r8"]
    vals = [data["always_notta"], data["always_ada"], data["always_lora"]]
    colors = [C["base"], C["ada"], C["lora"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("FVD (lower is better)")
    ax.set_ylim(148, 164)
    ax.set_title("FVD versus baseline (no-TTA)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=11)
    _save(fig, "03b_fvd_vs_baseline.png")


def chart_ood() -> None:
    gains = {r["video_id"]: _f(r, "ADA_dpsnr") for r in _read_csv(GAINS)}
    ood = _read_csv(SLIM / "longcat_gate/diffusion_ood_scores.csv")
    xs, ys = [], []
    for r in ood:
        d = gains.get(r["video_id"])
        o = _f(r, "mean_diffusion_loss_uncond")
        if d is None or o is None:
            continue
        xs.append(o)
        ys.append(d)
    xs = np.array(xs)
    ys = np.array(ys)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    q = np.array_split(np.arange(len(xs)), 5)
    q_vals = [ys[idx] for idx in q]
    q_mean = [v.mean() for v in q_vals]
    q_n = [len(v) for v in q_vals]
    q_min = [v.min() for v in q_vals]
    q_max = [v.max() for v in q_vals]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].scatter(xs, ys, s=8, alpha=0.25, c=C["ada"], linewidths=0)
    z = np.polyfit(xs, ys, 1)
    xx = np.linspace(xs.min(), xs.max(), 50)
    axes[0].plot(xx, np.polyval(z, xx), color=C["lose"], lw=2)
    axes[0].axhline(0, color=C["ink"], lw=1)
    axes[0].set_xlabel("Diffusion OOD (FM loss)")
    axes[0].set_ylabel("AdaSteer ΔPSNR (dB)")
    axes[0].set_title("Per-video scatter")
    colors = [C["win"] if v > 0 else C["lose"] for v in q_mean]
    axes[1].bar(range(1, 6), q_mean, color=colors)
    axes[1].axhline(0, color=C["ink"], lw=1)
    axes[1].set_xticks(range(1, 6))
    axes[1].set_xticklabels([f"Q{i}\nn={n}" for i, n in enumerate(q_n, 1)])
    axes[1].set_ylim(-0.28, 0.28)
    axes[1].set_xlabel("OOD quintile (low → high)")
    axes[1].set_ylabel("mean ΔPSNR (dB)")
    axes[1].set_title("Mean ΔPSNR by OOD quintile")
    for i, (m, lo, hi) in enumerate(zip(q_mean, q_min, q_max), 1):
        if abs(m) < 0.02:
            above = i % 2 == 1
            y_txt, va = (0.055, "bottom") if above else (-0.055, "top")
        elif m >= 0:
            y_txt, va = m + 0.012, "bottom"
        else:
            y_txt, va = m - 0.012, "top"
        axes[1].text(
            i,
            y_txt,
            f"{m:+.3f}\n[{lo:+.2f}, {hi:+.2f}]",
            ha="center",
            va=va,
            fontsize=8,
        )
    fig.suptitle("Diffusion OOD versus AdaSteer ΔPSNR", y=1.04)
    _save(fig, "04_ood_vs_adasteer_quintiles.png")


def chart_router() -> None:
    # Method performance (not oracle-captured %). Oracle is a ceiling line.
    # VBench dims and per-pack FVD were not tabulated; VBench = total,
    # FVD = matched N=1000 policies that exist (not the three packs).
    methods = ["OOD block", "VAE 130-d", "9-d pixel+embed"]
    # Δ vs fixed S10 (2026-07-07 / 2026-07-21).
    vb200 = [0.0069, 0.0136, 0.0291]
    vb1000 = [-0.0015, -0.0069, -0.0038]
    # PSNR Δ vs fixed. OOD pack was not reported.
    psnr200 = [np.nan, -0.046, 0.009]
    psnr1000 = [np.nan, -0.002, 0.025]
    x = np.arange(len(methods))
    w = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4))

    def _grouped(ax, y200, y1000, ylabel, title, oracle_200, oracle_1000, fmt, ylim):
        ax.bar(x - w / 2, np.nan_to_num(y200, nan=0.0), w, label="N=200", color=C["win"])
        ax.bar(x + w / 2, np.nan_to_num(y1000, nan=0.0), w, label="N=1000", color=C["lose"])
        ax.axhline(0, color=C["ink"], lw=1)
        if oracle_200 is not None:
            ax.axhline(oracle_200, color=C["win"], ls="--", lw=1.1, label=f"N=200 oracle {oracle_200:{fmt}}")
        if oracle_1000 is not None:
            ax.axhline(oracle_1000, color=C["lose"], ls="--", lw=1.1, label=f"N=1000 oracle {oracle_1000:{fmt}}")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.legend(frameon=False, fontsize=8)
        pad = 0.015 * (ylim[1] - ylim[0])
        for dx, vals in ((-w / 2, y200), (w / 2, y1000)):
            for i, v in enumerate(vals):
                if not np.isfinite(v):
                    ax.text(i + dx, pad, "—", ha="center", va="bottom", fontsize=8, color=C["mute"])
                    continue
                ax.text(
                    i + dx,
                    v + (pad if v >= 0 else -pad),
                    f"{v:{fmt}}",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7,
                )

    _grouped(
        axes[0, 0],
        vb200,
        vb1000,
        "VBench++ Δ vs. fixed S10",
        "VBench++ total (oracle = ceiling)",
        0.140,
        0.0978,
        "+.3f",
        (-0.03, 0.175),
    )
    _grouped(
        axes[0, 1],
        psnr200,
        psnr1000,
        "PSNR Δ vs. fixed S10 (dB)",
        "PSNR (oracle = ceiling; OOD not reported)",
        0.748,
        0.382,
        "+.3f",
        (-0.12, 0.85),
    )

    # Matched FVD on the N=1000 series. Not split by the three packs.
    fvd_labs = ["Baseline\n(no-TTA)", "Fixed\nAdaSteer", "VBench router\n(A+B+C)", "PSNR oracle\n(ceiling)"]
    fvd_vals = [81.22, 84.77, 82.67, 72.28]
    fvd_cols = [C["base"], C["mute"], C["ada"], C["win"]]
    axes[1, 0].bar(fvd_labs, fvd_vals, color=fvd_cols)
    axes[1, 0].set_ylabel("FVD (lower is better)")
    axes[1, 0].set_title("FVD, N=1000 matched (not split by pack)")
    axes[1, 0].set_ylim(68, 90)
    axes[1, 0].axhline(72.28, color=C["win"], ls="--", lw=1.1)
    for i, v in enumerate(fvd_vals):
        axes[1, 0].text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)

    o_labs = ["N=200\noracle", "N=1000\noracle", "N=1000\nnoise floor"]
    o_vals = [0.140, 0.0978, 0.0985]
    o_cols = [C["win"], C["ada"], C["mute"]]
    axes[1, 1].bar(o_labs, o_vals, color=o_cols)
    axes[1, 1].set_ylabel("VBench++ Δ vs. fixed S10 (raw)")
    axes[1, 1].set_title("Oracle size vs. noise floor")
    axes[1, 1].set_ylim(0, 0.18)
    for i, v in enumerate(o_vals):
        axes[1, 1].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=10)
    fig.suptitle("Router method performance: N=200 versus N=1000", y=1.02)
    _save(fig, "05_router_metrics_n200_n1000.png")


def chart_router_disagree() -> None:
    rows = _read_csv(SLIM / "longcat_gate/router_objective_alignment/per_video_alignment.csv")
    agree = np.mean([int(r["picks_agree"]) for r in rows]) * 100
    oracle = np.mean([int(r["oracles_agree"]) for r in rows]) * 100
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(
        ["Routers select\nthe same config", "VBench oracle =\nPSNR oracle"],
        [agree, oracle],
        color=[C["ada"], C["lose"]],
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of videos")
    ax.set_title("Agreement of VBench and PSNR routing objectives")
    ax.text(0, agree + 2, f"{agree:.0f}%", ha="center")
    ax.text(1, oracle + 2, f"{oracle:.0f}%", ha="center")
    _save(fig, "06_router_objective_agreement.png")


def chart_ar_drift() -> None:
    data = json.loads((SLIM / "longcat_ar/longhorizon_sweep_notta_native_12ch/merged_summary.json").read_text())
    curves = data["drift_curves"]
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    chunks = np.arange(1, 13)
    for key, label, color in (
        ("sharpness", "Sharpness (HF)", C["lose"]),
        ("temporal_motion", "Temporal motion", C["ada"]),
        ("contrast", "Contrast", C["lora"]),
    ):
        m = np.array(curves[key]["mean"], dtype=float)
        pct = 100.0 * (m / m[0] - 1.0)
        ax.plot(chunks, pct, marker="o", color=color, label=label)
    ax.axhline(0, color=C["ink"], lw=1)
    ax.set_xlabel("Chunk (native ~60 s / 12 chunks, N=8)")
    ax.set_ylabel("% vs. chunk 1")
    ax.legend(frameon=False)
    ax.set_title("Native autoregressive drift over 12 chunks")
    _save(fig, "07_ar_drift_12chunks.png")


def _cite128() -> dict[str, list[dict]]:
    rows = _read_csv(SLIM / "wan/cite128_per_video.csv")
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    return by


def _dyn_pct(rows: list[dict]) -> tuple[float, int, int]:
    vals = [_f(r, "dynamic_degree") for r in rows]
    vals = [v for v in vals if v is not None]
    n = len(vals)
    k = sum(1 for v in vals if v >= 0.5)
    return 100.0 * k / n if n else 0.0, k, n


def chart_cite128_win_cost() -> None:
    by = _cite128()
    order = [
        ("notta", "Self Forcing", 108),
        ("rolling_notta", "Rolling", 47),
        ("sf_pseudo", "Gated search", 294),
        ("sf_always_search", "Always-search", 354),
    ]
    labels = [lab for _, lab, _ in order]
    dyn, wall, iq, subj = [], [], [], []
    for key, _, w in order:
        rows = by[key]
        d, _, _ = _dyn_pct(rows)
        dyn.append(d)
        wall.append(w)
        iq.append(np.median([_f(r, "imaging_quality") for r in rows if _f(r, "imaging_quality") is not None]))
        subj.append(np.median([_f(r, "subject_consistency") for r in rows if _f(r, "subject_consistency") is not None]))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    colors = [C["base"], C["mute"], C["ada"], C["win"]]
    axes[0].bar(labels, dyn, color=colors)
    axes[0].set_ylabel("Dynamic Degree (% of clips)")
    axes[0].set_title("Dynamic Degree")
    for i, v in enumerate(dyn):
        axes[0].text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=10)
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(labels, wall, color=colors)
    axes[1].set_ylabel("Wall s / clip (job / 96)")
    axes[1].set_title("Wall time")
    for i, v in enumerate(wall):
        axes[1].text(i, v + 6, str(v), ha="center", fontsize=10)
    axes[1].tick_params(axis="x", rotation=15)
    fig.suptitle(
        "Official Dynamic Degree and generation wall time (cite-128)",
        y=1.04,
    )
    _save(fig, "08_cite128_dyn_wall.png")


def chart_cite128_flip() -> None:
    by = _cite128()
    sf = {r["clip"]: _f(r, "dynamic_degree") for r in by["notta"]}
    alw = {r["clip"]: _f(r, "dynamic_degree") for r in by["sf_always_search"]}
    keys = sorted(set(sf) & set(alw))
    both_dead = both_live = gained = lost = 0
    for k in keys:
        a = (sf[k] or 0) >= 0.5
        b = (alw[k] or 0) >= 0.5
        if a and b:
            both_live += 1
        elif (not a) and b:
            gained += 1
        elif a and (not b):
            lost += 1
        else:
            both_dead += 1
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    labs = ["Unchanged (static)", "Unchanged (dynamic)", "Became dynamic", "Became static"]
    vals = [both_dead, both_live, gained, lost]
    cols = [C["base"], C["ada"], C["win"], C["lose"]]
    ax.bar(labs, vals, color=cols)
    ax.set_ylabel("clips (N=128)")
    ax.set_title("Per-clip Dynamic Degree transitions: Always-search vs. Self Forcing")
    for i, v in enumerate(vals):
        ax.text(i, v + 1, str(v), ha="center")
    _save(fig, "09_dyn_transitions_search.png")


def chart_cite128_hold() -> None:
    by = _cite128()
    order = [
        ("notta", "Self Forcing"),
        ("rolling_notta", "Rolling"),
        ("sf_pseudo", "Gated"),
        ("sf_always_search", "Always"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for key, lab in order:
        rows = by[key]
        x = np.median([_f(r, "subject_consistency") for r in rows if _f(r, "subject_consistency") is not None])
        y = np.median([_f(r, "imaging_quality") for r in rows if _f(r, "imaging_quality") is not None])
        d, k, n = _dyn_pct(rows)
        ax.scatter(x, y, s=80 + 8 * k, label=f"{lab}  Dyn {d:.0f}%", zorder=3)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Subject Consistency (median)")
    ax.set_ylabel("Imaging Quality (median)")
    ax.set_title("Subject consistency versus imaging quality (cite-128)")
    ax.legend(frameon=False, loc="lower left")
    _save(fig, "10_subject_vs_iq_cite128.png")


def chart_prefix_trade() -> None:
    cap = _read_csv(SLIM / "wan/caption32_per_video.csv")
    pre = _read_csv(SLIM / "wan/prefix32_per_video.csv")
    sf = [r for r in cap if r["method"] == "notta"]
    seed = [r for r in pre if r["method"] == "seed_bon"]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for rows, lab, color in (
        (sf, "Self Forcing (same 32)", C["base"]),
        (seed, "Prefix-match", C["win"]),
    ):
        x = [_f(r, "subject_consistency") for r in rows]
        y = [_f(r, "tail_motion") for r in rows]
        ax.scatter(x, y, s=36, alpha=0.8, c=color, label=lab)
    ax.set_xlabel("Subject Consistency")
    ax.set_ylabel("Tail motion")
    ax.set_title(
        "Prefix-match versus Self Forcing: subject consistency and tail motion (N=32)"
    )
    ax.legend(frameon=False)
    _save(fig, "11_prefix_subject_tail.png")


def chart_path_iq() -> None:
    # N=8 harvests. Label N=8. Cite caption SF first-8 from caption32 notta.
    cap = [r for r in _read_csv(SLIM / "wan/caption32_per_video.csv") if r["method"] == "notta"]
    first8 = {r["clip"] for r in cap}
    # official first-8 leftover clips are panda_0000..0007
    sf8 = [r for r in cap if r["clip"] in {f"panda_{i:04d}" for i in range(8)}]
    packs = [
        ("SF (first 8)", sf8),
        ("nwarp", [r for r in _read_csv(SLIM / "wan/v2v_panda_caption_nwarp_8v_per_video.csv") if r["method"] == "sf_nwarp"]),
        ("pwarp", [r for r in _read_csv(SLIM / "wan/v2v_panda_caption_pwarp_8v_per_video.csv") if r["method"] == "sf_pwarp"]),
        ("Wan-extend", _read_csv(SLIM / "wan/v2v_panda_caption_wanext_8v_per_video.csv")),
    ]
    labels, iq, dyn, subj = [], [], [], []
    for lab, rows in packs:
        labels.append(lab)
        iq.append(np.median([_f(r, "imaging_quality") for r in rows if _f(r, "imaging_quality") is not None]))
        d, _, _ = _dyn_pct(rows)
        dyn.append(d)
        subj.append(np.median([_f(r, "subject_consistency") for r in rows if _f(r, "subject_consistency") is not None]))
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].bar(x, iq, color=[C["base"], C["lose"], C["mute"], C["lose"]])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Imaging Quality (median)")
    axes[0].set_title("Imaging Quality (median)")
    axes[1].bar(x, dyn, color=[C["base"], C["lose"], C["mute"], C["ada"]])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Dyn (% of clips)")
    axes[1].set_title("Dynamic Degree (% of clips)")
    fig.suptitle(
        "Path interventions on imaging quality and Dynamic Degree (N=8)",
        y=1.04,
    )
    _save(fig, "12_path_iq_dyn_n8.png")
    _ = first8


def main() -> None:
    _style()
    chart_dpsnr_hist()
    chart_oracle()
    chart_lora_trade()
    chart_fvd_methods()
    chart_ood()
    chart_router()
    chart_router_disagree()
    chart_ar_drift()
    chart_cite128_win_cost()
    chart_cite128_flip()
    chart_cite128_hold()
    chart_prefix_trade()
    chart_path_iq()
    print(f"done → {OUT}")


if __name__ == "__main__":
    main()
