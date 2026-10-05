#!/usr/bin/env python3
"""
Plot long-horizon DRIFT curves from `drift_stats.json`
(produced by comparison_methods/scripts/longhorizon_drift_rollout.py).

For each GT-free drift signal (sharpness, motion, colorfulness, saturation,
brightness) plus the cross-chunk seam ratio (and optional GT PSNR), we:
  * average per-chunk values across videos (mean +/- SEM),
  * fit an OLS trend of value vs chunk index (pooling all videos' per-chunk
    points) and report slope, a normal-approx two-sided p-value, and the
    percent change from chunk 0 to the last chunk,
  * render a multi-panel PNG.

Reading the curves
------------------
A signal that DRIFTS (monotone slope, small p) over chunks == the model
degrades under long autoregressive rollout == HEADROOM for a correction method.
Flat curves (large p) == LongCat is already stable at this horizon == the
problem is too easy and we should harden it further (longer horizon / weaker
base model / OOD).

No scipy dependency (p-value via normal approximation to the OLS t-statistic).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


SIGNALS = ["sharpness", "motion", "colorfulness", "saturation", "brightness"]
# motion[0] is NaN by construction (no previous frame); the per-chunk mean uses
# nanmean so it is fine, but flag direction of "worse":
WORSE_WHEN = {  # for the human-readable verdict only
    "sharpness": "down (blur)",
    "motion": "down (freeze) / up (chaos)",
    "colorfulness": "down (fade)",
    "saturation": "drift either way",
    "brightness": "drift either way",
    "seam_ratio": "up (worse stitching)",
    "gt_psnr": "down (leaves real future)",
}


def _ols_slope_p(xs, ys):
    """OLS slope of ys~xs with normal-approx two-sided p-value. NaNs dropped."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    n = xs.size
    if n < 3 or np.ptp(xs) == 0:
        return float("nan"), float("nan"), n
    xbar = xs.mean()
    sxx = float(((xs - xbar) ** 2).sum())
    b = float(((xs - xbar) * (ys - ys.mean())).sum() / sxx)
    a = float(ys.mean() - b * xbar)
    resid = ys - (a + b * xs)
    dof = n - 2
    s2 = float((resid ** 2).sum() / dof)
    se = math.sqrt(s2 / sxx) if sxx > 0 else float("nan")
    if not se or math.isnan(se):
        return b, float("nan"), n
    t = b / se
    # two-sided p via normal approx: erfc(|t|/sqrt2)
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return b, p, n


def _collect(per_video, key):
    """Return (nchunks, mean[chunk], sem[chunk], pooled_x, pooled_y)."""
    seqs = []
    for r in per_video:
        if not r.get("success"):
            continue
        if key == "seam_ratio":
            seq = [s.get("ratio", float("nan")) for s in r.get("seams", [])]
            # seams start at chunk 1; pad chunk 0 with NaN for alignment
            seq = [float("nan")] + seq
        elif key == "gt_psnr":
            seq = r.get("gt_psnr_per_chunk", [])
        else:
            seq = r.get("per_chunk", {}).get(key, [])
        if seq:
            seqs.append(np.asarray(seq, dtype=np.float64))
    if not seqs:
        return 0, [], [], [], []
    nchunks = max(len(s) for s in seqs)
    mat = np.full((len(seqs), nchunks), np.nan)
    for i, s in enumerate(seqs):
        mat[i, :len(s)] = s
    mean = np.nanmean(mat, axis=0)
    cnt = np.sum(np.isfinite(mat), axis=0)
    sem = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(cnt, 1))
    pooled_x, pooled_y = [], []
    for i in range(mat.shape[0]):
        for c in range(nchunks):
            if np.isfinite(mat[i, c]):
                pooled_x.append(c)
                pooled_y.append(mat[i, c])
    return nchunks, mean, sem, pooled_x, pooled_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, help="drift_stats.json path")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--title", default=None)
    ap.add_argument("--md-out", default=None, help="optional markdown slope table")
    args = ap.parse_args()

    data = json.loads(Path(args.stats).read_text())
    pv = data.get("per_video", [])
    nsucc = sum(1 for r in pv if r.get("success"))
    keys = SIGNALS + ["seam_ratio"]
    if any(r.get("gt_psnr_per_chunk") for r in pv if r.get("success")):
        keys.append("gt_psnr")

    ncols = 3
    nrows = int(math.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    rows = []
    for ax, key in zip(axes, keys):
        nchunks, mean, sem, px, py = _collect(pv, key)
        if nchunks == 0:
            ax.set_title(f"{key} (no data)")
            continue
        x = np.arange(nchunks)
        ax.errorbar(x, mean, yerr=sem, marker="o", capsize=3, lw=1.8)
        slope, pval, n = _ols_slope_p(px, py)
        # % change chunk0 -> last (using first/last finite means)
        finite = np.where(np.isfinite(mean))[0]
        pct = float("nan")
        if finite.size >= 2 and mean[finite[0]] not in (0.0,) and np.isfinite(mean[finite[0]]):
            pct = 100.0 * (mean[finite[-1]] - mean[finite[0]]) / abs(mean[finite[0]])
        ax.set_title(f"{key}  slope={slope:.3g} p={pval:.3g}\nΔ0→N={pct:+.1f}%  ({WORSE_WHEN.get(key,'')})",
                     fontsize=9)
        ax.set_xlabel("chunk index (autoregressive step)")
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
        rows.append((key, slope, pval, pct, n))

    for ax in axes[len(keys):]:
        ax.axis("off")

    ttl = args.title or (f"LongCat true-AR drift  (N={nsucc} videos, "
                         f"cond={data.get('num_cond_frames')} chunk_gen={data.get('chunk_gen')} "
                         f"num_chunks={data.get('num_chunks')})")
    fig.suptitle(ttl, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"[plot] wrote {args.out}")

    # slope table
    hdr = "| signal | slope/chunk | p (2-sided) | Δ chunk0→N | n pts | worse when |"
    sep = "|---|---:|---:|---:|---:|---|"
    lines = [hdr, sep]
    for key, slope, pval, pct, n in rows:
        lines.append(f"| {key} | {slope:.4g} | {pval:.3g} | {pct:+.1f}% | {n} | {WORSE_WHEN.get(key,'')} |")
    table = "\n".join(lines)
    print("\n" + table)
    if args.md_out:
        Path(args.md_out).write_text(
            f"# Long-horizon drift slopes\n\n{ttl}\n\n{table}\n\n"
            "Small p + monotone slope on sharpness/motion/colorfulness ⇒ real "
            "drift ⇒ headroom for a correction method. Flat (large p) ⇒ LongCat "
            "stable at this horizon ⇒ harden further (longer horizon / weaker "
            "base / OOD).\n")
        print(f"[plot] wrote {args.md_out}")


if __name__ == "__main__":
    main()
