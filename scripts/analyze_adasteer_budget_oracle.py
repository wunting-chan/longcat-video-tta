#!/usr/bin/env python3
"""Analyze AdaSteer step×LR budget grid as a per-video oracle (H9).

After the ``panda_1000v_adasteer_budget`` (or pilot) sweep completes and
chunks are merged, this script loads per-video PSNR from every grid config,
computes:

  * Oracle-best PSNR uplift vs fixed headline AdaSteer (S10/LR5e-3)
  * Oracle-best PSNR uplift vs always-NOTTA (delta_steps=0), when present
  * OOD-quintile stratification of oracle winners and mean PSNR
  * Quintile-adaptive policy: per OOD quintile, pick the grid config with
    highest mean PSNR in that quintile (deployable upper bound)
  * Bootstrap 95% CI for population mean oracle uplift (per-video resampling)
  * Full population metrics table (PSNR/SSIM/LPIPS/FVD/FID) for every grid config
  * Oracle row: per-video argmax-PSNR routing for PSNR/SSIM/LPIPS; FVD/FID from
    ``--oracle-fvd-json`` when a budget-oracle FVD eval has been run (see
    ``sweep_experiment/scripts/run_budget_oracle_fvd.py``)

Usage:
    python scripts/analyze_adasteer_budget_oracle.py --bootstrap \\
        --series-root sweep_experiment/results/panda_ood_budget_pilot \\
        --baseline-series-root sweep_experiment/results/panda_1000v_standard \\
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\
        --output sweep_experiment/reports/per_video_analysis/2026-06-20/adasteer_budget_oracle_pilot.md

    python scripts/analyze_adasteer_budget_oracle.py --bootstrap
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_per_video_tta_gain import load_per_video_metrics  # noqa: E402
from scripts.caption_utils import canonical_video_id

DEFAULT_SERIES = _REPO_ROOT / "sweep_experiment/results/panda_ood_budget_pilot"
DEFAULT_OOD = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv"
)

OOD_COL = "mean_diffusion_loss_caption"
FIXED_ADA_RUN_ID = "S10_LR5e3"
NOTTA_RUN_ID = "NOTTA"
HEADLINE_PSNR_RUN_ID = "S2_LR1e2"

# H9 pilot headline pairwise PSNR comparisons (per-video paired diffs).
DEFAULT_PAIRWISE_COMPARISONS: Tuple[Tuple[str, str], ...] = (
    (HEADLINE_PSNR_RUN_ID, FIXED_ADA_RUN_ID),
    (HEADLINE_PSNR_RUN_ID, NOTTA_RUN_ID),
    (FIXED_ADA_RUN_ID, NOTTA_RUN_ID),
)

# 12-config pilot subset (LR 1e-3, 5e-3, 1e-2 × steps 2, 5, 10, 20).
PILOT_GRID_RUN_ORDER: Tuple[str, ...] = (
    "S2_LR1e3", "S2_LR5e3", "S2_LR1e2",
    "S5_LR1e3", "S5_LR5e3", "S5_LR1e2",
    "S10_LR1e3", "S10_LR5e3", "S10_LR1e2",
    "S20_LR1e3", "S20_LR5e3", "S20_LR1e2",
)

METRIC_KEYS: Tuple[str, ...] = ("psnr", "ssim", "lpips", "fvd", "fid")

_RUN_ID_RE = re.compile(r"^S(\d+)_LR(.+)$")


def paired_bootstrap_mean_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 5000,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float], Optional[float], int, Optional[bool]]:
    """Bootstrap CI for mean(a - b) on paired finite entries."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mask = ~(np.isnan(aa) | np.isnan(bb))
    d = aa[mask] - bb[mask]
    n = int(d.size)
    if n == 0:
        return None, None, None, 0, None
    mean = float(d.mean())
    if n < 2:
        return mean, None, None, n, None
    rng = np.random.default_rng(seed)
    boot = [float(d[rng.integers(0, n, size=n)].mean()) for _ in range(n_boot)]
    boot_arr = np.asarray(boot, dtype=np.float64)
    ci_lo = float(np.percentile(boot_arr, 2.5))
    ci_hi = float(np.percentile(boot_arr, 97.5))
    excludes_zero = bool((ci_lo > 0.0) or (ci_hi < 0.0))
    return mean, ci_lo, ci_hi, n, excludes_zero


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) via continued fraction (stdlib)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = _log_beta(a, b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - ln_beta) / a
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f *= d
    for m in range(1, 200):
        m2 = 2 * m
        num = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return front * (f - 1.0)


def _student_t_cdf(t: float, df: int) -> float:
    x = df / (df + t * t)
    ib = _betainc(df / 2.0, 0.5, x)
    if t >= 0.0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def paired_ttest_pvalue(diffs: Sequence[float]) -> Optional[float]:
    """Two-sided paired t-test H0: mean(diff) = 0 (ttest_1samp on diffs)."""
    a = np.asarray([x for x in diffs if not np.isnan(x)], dtype=float)
    n = int(a.size)
    if n < 2:
        return None
    if scipy_stats is not None:
        res = scipy_stats.ttest_1samp(a, 0.0, alternative="two-sided")
        return float(res.pvalue)
    mean = float(a.mean())
    std = float(a.std(ddof=1))
    if std == 0.0:
        return 0.0 if mean != 0.0 else 1.0
    t_stat = mean / (std / math.sqrt(n))
    return 2.0 * (1.0 - _student_t_cdf(abs(t_stat), n - 1))


def wilcoxon_pvalue(diffs: Sequence[float]) -> Optional[float]:
    """Two-sided Wilcoxon signed-rank on diffs (requires scipy)."""
    if scipy_stats is None:
        return None
    a = np.asarray([x for x in diffs if not np.isnan(x)], dtype=float)
    if a.size < 2:
        return None
    if np.allclose(a, 0.0):
        return 1.0
    try:
        res = scipy_stats.wilcoxon(a, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        return None
    return float(res.pvalue)


def _fmt_pvalue(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _fmt_sig(p: Optional[float], alpha: float = 0.05) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return "yes" if p < alpha else "no"


def compute_pairwise_comparison(
    table: Dict[str, Dict[str, float]],
    run_a: str,
    run_b: str,
    *,
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, object]:
    """Paired PSNR comparison run_a − run_b with bootstrap CI and p-values."""
    psnr_a: List[float] = []
    psnr_b: List[float] = []
    diffs: List[float] = []
    for _vid, row in sorted(table.items()):
        pa = row.get(run_a)
        pb = row.get(run_b)
        if pa is None or pb is None or np.isnan(pa) or np.isnan(pb):
            continue
        psnr_a.append(float(pa))
        psnr_b.append(float(pb))
        diffs.append(float(pa) - float(pb))
    n = len(diffs)
    mean_d, ci_lo, ci_hi, _n_booted, excl = paired_bootstrap_mean_diff(
        psnr_a, psnr_b, n_boot=n_boot, seed=seed,
    )
    t_p = paired_ttest_pvalue(diffs)
    w_p = wilcoxon_pvalue(diffs)
    return {
        "run_a": run_a,
        "run_b": run_b,
        "label": f"`{run_a}` − `{run_b}`",
        "n": n,
        "mean_delta": mean_d,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_excludes_zero": excl,
        "t_pvalue": t_p,
        "wilcoxon_pvalue": w_p,
        "t_sig_005": _fmt_sig(t_p),
    }


def format_pairwise_psnr_tests_md(
    comparisons: Sequence[Dict[str, object]],
    *,
    n_boot: int = 5000,
    seed: int = 42,
    scipy_available: bool = scipy_stats is not None,
) -> List[str]:
    """Markdown table for headline pairwise PSNR tests (H9 pilot)."""
    wilcoxon_hdr = "Wilcoxon p" if scipy_available else "Wilcoxon p (scipy N/A)"
    lines = [
        "## Pairwise PSNR comparisons (H9 pilot)",
        "",
        "Per-video paired ΔPSNR (config A − config B). Bootstrap resamples videos "
        f"(B={n_boot}, seed={seed}). Paired *t*-test: `scipy.stats.ttest_1samp` on "
        "per-video diffs (two-sided H₀: mean Δ = 0).",
        "",
        f"| Comparison | N | Mean Δ (dB) | Bootstrap 95% CI | CI excludes 0? | "
        f"Paired *t*-test p | p < 0.05? | {wilcoxon_hdr} |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        mean_d = row.get("mean_delta")
        ci_lo = row.get("ci_lo")
        ci_hi = row.get("ci_hi")
        excl = row.get("ci_excludes_zero")
        mean_s = f"{float(mean_d):+.3f}" if mean_d is not None else "—"
        if ci_lo is not None and ci_hi is not None:
            ci_s = f"[{float(ci_lo):+.3f}, {float(ci_hi):+.3f}]"
        else:
            ci_s = "—"
        excl_s = "yes" if excl else ("no" if excl is not None else "—")
        t_p = row.get("t_pvalue")
        w_p = row.get("wilcoxon_pvalue")
        w_s = _fmt_pvalue(w_p if isinstance(w_p, float) or w_p is None else None)
        if not scipy_available:
            w_s = "—"
        lines.append(
            f"| {row['label']} | {row['n']} | {mean_s} | {ci_s} | {excl_s} | "
            f"{_fmt_pvalue(t_p if isinstance(t_p, float) or t_p is None else None)} | "
            f"{row.get('t_sig_005', '—')} | {w_s} |"
        )
    if not scipy_available:
        lines += [
            "",
            "> Wilcoxon signed-rank test requires `scipy` (install on cluster: "
            "`pip install scipy` or use conda env with scipy).",
        ]
    lines.append("")
    return lines


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 5000,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[bool]]:
    """Per-video bootstrap CI for the mean."""
    a = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if a.size == 0:
        return None, None, None, None
    mean = float(a.mean())
    if a.size < 2:
        return mean, None, None, None
    rng = np.random.default_rng(seed)
    boot_means = []
    n = a.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means.append(float(a[idx].mean()))
    boot_arr = np.asarray(boot_means, dtype=np.float64)
    ci_lo = float(np.percentile(boot_arr, 2.5))
    ci_hi = float(np.percentile(boot_arr, 97.5))
    ci_excludes_zero = bool((ci_lo > 0.0) or (ci_hi < 0.0))
    return mean, ci_lo, ci_hi, ci_excludes_zero


def _stats(arr: Sequence[float]) -> Tuple[int, float, float, float, float]:
    a = np.asarray([x for x in arr if not np.isnan(x)], dtype=float)
    if a.size == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    return (
        int(a.size),
        float(np.mean(a)),
        float(np.median(a)),
        float(np.percentile(a, 25)),
        float(np.percentile(a, 75)),
    )


def _fmt_stats(label: str, arr: Sequence[float]) -> str:
    n, mean, med, p25, p75 = _stats(arr)
    if n == 0:
        return f"| {label} | 0 | — | — | — | — |"
    return (
        f"| {label} | {n} | {mean:.3f} dB | {med:.3f} dB | "
        f"{p25:.3f} dB | {p75:.3f} dB |"
    )


def load_ood_quintiles(path: Path, n_bins: int = 5) -> Dict[str, int]:
    """Return {video_id: quintile 1..n_bins} from OOD CSV."""
    rows: List[Tuple[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = canonical_video_id(row.get("video_id", ""))
            v = row.get(OOD_COL, "")
            if not vid or v in ("", None):
                continue
            try:
                ood = float(v)
            except ValueError:
                continue
            if np.isnan(ood):
                continue
            rows.append((vid, ood))
    if not rows:
        return {}
    vals = np.asarray([x[1] for x in rows], dtype=float)
    edges = np.quantile(vals, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    out: Dict[str, int] = {}
    for vid, ood in rows:
        idx = int(np.digitize([ood], edges[1:-1], right=False)[0])
        idx = min(max(idx, 0), max(len(edges) - 2, 0))
        out[vid] = idx + 1
    return out


def _has_per_video_summaries(run_dir: Path) -> bool:
    """True when chunk or flat summaries exist (merged alone is aggregate-only)."""
    if any(run_dir.glob("chunk_*/summary.json")):
        return True
    if any(run_dir.glob("chunk_*/results.json")):
        return True
    for flat_name in ("summary.json", "merged_summary.json"):
        flat = run_dir / flat_name
        if not flat.exists():
            continue
        try:
            with flat.open(encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(blob, dict):
            for key in ("results", "per_video_results", "per_video"):
                v = blob.get(key)
                if isinstance(v, list) and v:
                    return True
    return False


def load_run_psnr(run_dir: Path) -> Dict[str, float]:
    """Load per-video PSNR from chunk summaries (preferred) or flat summary files."""
    out: Dict[str, float] = {}
    for vid, metrics in load_per_video_metrics(run_dir).items():
        psnr = metrics.get("psnr")
        if psnr is not None:
            out[vid] = float(psnr)
    return out


def load_run_all_metrics(run_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load per-video PSNR/SSIM/LPIPS from chunk summaries."""
    out: Dict[str, Dict[str, float]] = {}
    for vid, metrics in load_per_video_metrics(run_dir).items():
        row: Dict[str, float] = {}
        for k in ("psnr", "ssim", "lpips"):
            v = metrics.get(k)
            if v is not None and not np.isnan(v):
                row[k] = float(v)
        if row:
            out[vid] = row
    return out


def load_merged_summary(run_dir: Path) -> Dict[str, object]:
    """Population metrics from ``merged_summary.json`` (post merge_chunks)."""
    path = run_dir / "merged_summary.json"
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _fmt_metric(val: object, *, decimals: int = 3, pct: bool = False) -> str:
    if val is None:
        return "—"
    try:
        x = float(val)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(x):
        return "—"
    if pct:
        return f"{x * 100:.2f}%"
    if decimals == 0:
        return f"{x:.0f}"
    return f"{x:.{decimals}f}"


def _sorted_grid_run_ids(run_ids: Sequence[str]) -> List[str]:
    order = {rid: i for i, rid in enumerate(PILOT_GRID_RUN_ORDER)}
    grid = [r for r in run_ids if r not in (NOTTA_RUN_ID,) and _RUN_ID_RE.match(r)]
    return sorted(grid, key=lambda r: (order.get(r, 999), r))


def compute_oracle_metric_means(
    metrics_by_run: Dict[str, Dict[str, Dict[str, float]]],
    grid_runs: Sequence[str],
    vids: Sequence[str],
    *,
    winner_metric: str = "psnr",
) -> Dict[str, Optional[float]]:
    """Per-video oracle pick by ``winner_metric``, then mean PSNR/SSIM/LPIPS."""
    out: Dict[str, List[float]] = {k: [] for k in ("psnr", "ssim", "lpips")}
    for vid in vids:
        pick_row: Dict[str, float] = {}
        for rid in grid_runs:
            m = metrics_by_run.get(rid, {}).get(vid, {})
            v = m.get(winner_metric)
            if v is not None and not np.isnan(v):
                pick_row[rid] = float(v)
        w = oracle_winner(pick_row, grid_runs)
        if w is None:
            continue
        m = metrics_by_run.get(w, {}).get(vid, {})
        for k in out:
            v = m.get(k)
            if v is not None and not np.isnan(v):
                out[k].append(float(v))
    return {
        k: (float(np.mean(vals)) if vals else None)
        for k, vals in out.items()
    }


def build_config_metrics_table(
    runs: Dict[str, Path],
    grid_runs: Sequence[str],
    *,
    oracle_means: Optional[Dict[str, Optional[float]]] = None,
    oracle_fvd_blob: Optional[dict] = None,
) -> Tuple[List[str], List[List[str]]]:
    """Return (header, rows) for markdown/CSV export."""
    header = [
        "run_id", "steps", "lr", "N",
        "PSNR (dB)", "SSIM", "LPIPS", "FVD", "FID",
    ]
    rows: List[List[str]] = []
    for run_id in _sorted_grid_run_ids(list(grid_runs)):
        run_dir = runs.get(run_id)
        if run_dir is None:
            continue
        merged = load_merged_summary(run_dir)
        steps, lr = parse_run_hparams(run_id)
        lr_s = f"{lr:.0e}" if lr is not None else "—"
        n = merged.get("num_successful") or merged.get("num_videos") or "—"
        rows.append([
            run_id,
            str(steps) if steps is not None else "—",
            lr_s,
            str(n),
            _fmt_metric(merged.get("psnr")),
            _fmt_metric(merged.get("ssim"), decimals=4),
            _fmt_metric(merged.get("lpips"), decimals=4),
            _fmt_metric(merged.get("fvd"), decimals=1),
            _fmt_metric(merged.get("fid"), decimals=1),
        ])

    if oracle_means is not None:
        fvd = fid = None
        if oracle_fvd_blob:
            fvd = oracle_fvd_blob.get("fvd")
            fid = oracle_fvd_blob.get("fid")
        rows.append([
            "ORACLE (best PSNR/video)",
            "—", "—", "—",
            _fmt_metric(oracle_means.get("psnr")),
            _fmt_metric(oracle_means.get("ssim"), decimals=4),
            _fmt_metric(oracle_means.get("lpips"), decimals=4),
            _fmt_metric(fvd, decimals=1),
            _fmt_metric(fid, decimals=1),
        ])
    return header, rows


def format_oracle_pick_table_md(
    winners: Dict[str, int],
    gains_by_winner: Dict[str, List[float]],
    *,
    grid_run_order: Sequence[str] = PILOT_GRID_RUN_ORDER,
    fixed_run: str = FIXED_ADA_RUN_ID,
    overall_gain_vs_fixed: Optional[float] = None,
    bootstrap_ci: Optional[Tuple[float, float]] = None,
) -> Tuple[List[str], List[List[str]]]:
    """Return (markdown lines, CSV rows) for Table 2 oracle pick frequency."""
    n_oracle = sum(winners.values())
    header = ["Config", "Oracle picks", "Freq", f"Mean ΔPSNR vs `{fixed_run}`"]
    csv_header = ["run_id", "oracle_picks", "freq_pct", "mean_delta_psnr_vs_fixed_db"]
    rows: List[List[str]] = []
    md_lines: List[str] = [
        "## Table 2 — Oracle pick frequency + routing uplift",
        "",
        f"Denominator: **N = {n_oracle}** pilot videos with a grid oracle winner "
        f"(not the NOTTA union size when `--baseline-series-root` is used).",
        "",
        f"*Mean ΔPSNR when picked = winner PSNR − fixed `{fixed_run}` PSNR "
        "on videos where that config wins.*",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for run_id in grid_run_order:
        cnt = winners.get(run_id, 0)
        freq = 100.0 * cnt / n_oracle if n_oracle else 0.0
        gains = gains_by_winner.get(run_id, [])
        if gains:
            mean_gain = float(np.mean(gains))
            gain_s = f"{mean_gain:+.3f} dB"
        else:
            mean_gain = None
            gain_s = "—"
        md_lines.append(
            f"| `{run_id}` | {cnt} | {freq:.1f}% | {gain_s} |"
        )
        rows.append([
            run_id,
            str(cnt),
            f"{freq:.2f}",
            "" if mean_gain is None else f"{mean_gain:.6f}",
        ])

    if overall_gain_vs_fixed is not None:
        uplift_s = f"{overall_gain_vs_fixed:+.3f} dB"
        if bootstrap_ci is not None:
            uplift_s += f" [{bootstrap_ci[0]:+.3f}, {bootstrap_ci[1]:+.3f}]"
        md_lines.append(
            f"| **Overall oracle vs fixed `{fixed_run}`** | **{n_oracle}** | "
            f"**100.0%** | **{uplift_s}** |"
        )
        rows.append([
            f"ORACLE_vs_{fixed_run}",
            str(n_oracle),
            "100.00",
            f"{overall_gain_vs_fixed:.6f}",
        ])
    md_lines.append("")
    return md_lines, [csv_header] + rows


def format_metrics_table_md(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    title: str = "Full grid population metrics",
) -> List[str]:
    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.append("")
    return lines


def write_metrics_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _infer_baseline_series_root(series_root: Path) -> Path:
    """Guess standard sweep root for NOTTA baseline (matches retrieval analyzer)."""
    name = series_root.name.lower()
    if "ucf" in name:
        return _REPO_ROOT / "sweep_experiment/results/ucf101_932v_standard"
    if "panda" in name:
        return _REPO_ROOT / "sweep_experiment/results/panda_1000v_standard"
    return _REPO_ROOT / "sweep_experiment/results/panda_1000v_standard"


def discover_runs(series_root: Path) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    if not series_root.is_dir():
        return runs
    for child in sorted(series_root.iterdir()):
        if not child.is_dir():
            continue
        if not _has_per_video_summaries(child):
            continue
        psnrs = load_run_psnr(child)
        if psnrs:
            runs[child.name] = child
    return runs


def parse_run_hparams(run_id: str) -> Tuple[Optional[int], Optional[float]]:
    if run_id == NOTTA_RUN_ID:
        return 0, None
    m = _RUN_ID_RE.match(run_id)
    if not m:
        return None, None
    steps = int(m.group(1))
    lr_str = m.group(2).lower().replace("e", "e")
    lr_map = {
        "1e3": 1e-3,
        "2p5e3": 2.5e-3,
        "2.5e3": 2.5e-3,
        "5e3": 5e-3,
        "7p5e3": 7.5e-3,
        "7.5e3": 7.5e-3,
        "1e2": 1e-2,
    }
    lr = lr_map.get(lr_str)
    if lr is None:
        try:
            lr = float(lr_str.replace("p", "."))
        except ValueError:
            lr = None
    return steps, lr


def build_video_table(
    runs: Dict[str, Path],
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Return (run_ids, {video_id: {run_id: psnr}})."""
    psnr_by_run: Dict[str, Dict[str, float]] = {}
    for run_id, run_dir in runs.items():
        psnr_by_run[run_id] = load_run_psnr(run_dir)
    all_vids = sorted(set().union(*[set(d.keys()) for d in psnr_by_run.values()]))
    table: Dict[str, Dict[str, float]] = {vid: {} for vid in all_vids}
    for run_id, d in psnr_by_run.items():
        for vid, psnr in d.items():
            table[vid][run_id] = psnr
    run_ids = sorted(r for r in psnr_by_run if r != NOTTA_RUN_ID)
    if NOTTA_RUN_ID in psnr_by_run:
        run_ids = [NOTTA_RUN_ID] + run_ids
    return run_ids, table


def oracle_winner(row: Dict[str, float], candidates: Iterable[str]) -> Optional[str]:
    best: Optional[str] = None
    best_psnr = float("-inf")
    for rid in candidates:
        p = row.get(rid)
        if p is None or np.isnan(p):
            continue
        if p > best_psnr:
            best_psnr = p
            best = rid
    return best


def build_report(
    *,
    series_root: Path,
    baseline_series_root: Optional[Path],
    baseline_run_id: str,
    run_ids: List[str],
    table: Dict[str, Dict[str, float]],
    metrics_by_run: Dict[str, Dict[str, Dict[str, float]]],
    ood_quintile: Dict[str, int],
    fixed_run: str,
    bootstrap: bool,
    n_boot: int,
    bootstrap_seed: int,
    pairwise_tests: bool,
    pairwise_comparisons: Sequence[Tuple[str, str]],
    metrics_header: List[str],
    metrics_rows: List[List[str]],
    oracle_fvd_blob: Optional[dict],
) -> Tuple[str, List[List[str]]]:
    grid_runs = [r for r in run_ids if r not in (NOTTA_RUN_ID,)]
    vids = sorted(table.keys())
    n_union = len(vids)

    lines: List[str] = [
        "# AdaSteer budget-grid oracle analysis (H9)",
        "",
        f"**Series:** `{series_root}`",
        f"**Fixed headline AdaSteer:** `{fixed_run}` (S10/LR=5e-3).",
    ]
    if baseline_series_root is not None:
        lines.append(
            f"**NOTTA baseline:** `{baseline_run_id}` from `{baseline_series_root}` "
            f"(union N = {n_union} when joined)."
        )
    else:
        lines.append(f"**Union N = {n_union}** videos with PSNR across ≥1 run.")
    lines += ["", ""]

    if fixed_run not in run_ids:
        lines.append(
            f"> Warning: fixed run `{fixed_run}` not found in series; "
            "fixed-policy comparisons skipped."
        )
        lines.append("")

    notta_psnr: List[float] = []
    fixed_psnr: List[float] = []
    oracle_psnr: List[float] = []
    oracle_gain_vs_fixed: List[float] = []
    oracle_gain_vs_notta: List[float] = []
    winners: Dict[str, int] = {}
    gains_by_winner: Dict[str, List[float]] = {}

    for vid in vids:
        row = table[vid]
        w = oracle_winner(row, grid_runs)
        if w is None:
            continue
        winners[w] = winners.get(w, 0) + 1
        p_oracle = row[w]
        oracle_psnr.append(p_oracle)
        if fixed_run in row:
            fixed_psnr.append(row[fixed_run])
            gain_vs_fixed = p_oracle - row[fixed_run]
            oracle_gain_vs_fixed.append(gain_vs_fixed)
            gains_by_winner.setdefault(w, []).append(gain_vs_fixed)
        if NOTTA_RUN_ID in row:
            notta_psnr.append(row[NOTTA_RUN_ID])
            oracle_gain_vs_notta.append(p_oracle - row[NOTTA_RUN_ID])

    n_oracle = sum(winners.values())
    if n_oracle:
        lines.insert(
            4,
            f"**Pilot grid oracle N = {n_oracle}** videos with per-video PSNR across "
            "the budget grid (denominator for pick frequencies).",
        )

    def mean_psnr(arr: List[float]) -> float:
        a = np.asarray(arr, dtype=float)
        return float(np.mean(a)) if a.size else float("nan")

    lines += ["", ""]

    lines += format_metrics_table_md(
        metrics_header,
        metrics_rows,
        title="Full grid population metrics (merged summaries)",
    )
    if oracle_fvd_blob is None:
        lines += [
            "> **Oracle FVD/FID:** Global FVD does not decompose per-video from "
            "``merged_summary.json`` alone. Run budget-oracle FVD eval on cluster "
            "(requires saved mp4s — re-run with ``NO_SAVE_VIDEOS=0`` or use the "
            "1000v best-config jobs):",
            "> ```bash",
            "> python sweep_experiment/scripts/run_budget_oracle_fvd.py \\",
            f">     --series-root {series_root} \\",
            ">     --gt-cache gt_caches/panda_1000_longcat.npz",
            "> ```",
            "",
        ]
    else:
        lines += [
            f"> Oracle FVD/FID from ``{oracle_fvd_blob.get('_source', 'fvd.json')}``: "
            f"FVD={_fmt_metric(oracle_fvd_blob.get('fvd'), decimals=1)}, "
            f"FID={_fmt_metric(oracle_fvd_blob.get('fid'), decimals=1)} "
            f"(N={oracle_fvd_blob.get('num_valid_pairs', '—')} videos).",
            "",
        ]

    lines += [
        "## Population routing uplift",
        "",
        "| Policy | Mean PSNR | Δ vs always-NOTTA | Δ vs fixed AdaSteer |",
        "|---|---:|---:|---:|",
    ]
    if notta_psnr:
        lines.append(
            f"| Always NOTTA | {mean_psnr(notta_psnr):.3f} dB | 0.000 dB | — |"
        )
    if fixed_psnr:
        delta_n = (
            mean_psnr(fixed_psnr) - mean_psnr(notta_psnr)
            if notta_psnr
            else float("nan")
        )
        lines.append(
            f"| Fixed AdaSteer (`{fixed_run}`) | {mean_psnr(fixed_psnr):.3f} dB | "
            f"{delta_n:+.3f} dB | 0.000 dB |"
        )
    if oracle_psnr:
        d_n = (
            mean_psnr(oracle_psnr) - mean_psnr(notta_psnr)
            if notta_psnr
            else float("nan")
        )
        d_f = (
            mean_psnr(oracle_gain_vs_fixed) + mean_psnr(fixed_psnr) - mean_psnr(fixed_psnr)
            if fixed_psnr
            else float("nan")
        )
        d_f = mean_psnr(oracle_gain_vs_fixed) if oracle_gain_vs_fixed else float("nan")
        lines.append(
            f"| **Oracle (best grid PSNR)** | **{mean_psnr(oracle_psnr):.3f} dB** | "
            f"**{d_n:+.3f} dB** | **{d_f:+.3f} dB** |"
        )
    lines.append("")

    if pairwise_tests:
        pw_rows: List[Dict[str, object]] = []
        for run_a, run_b in pairwise_comparisons:
            if run_a not in run_ids or run_b not in run_ids:
                print(
                    f"[warn] pairwise skip: missing run(s) `{run_a}` or `{run_b}`",
                    file=sys.stderr,
                )
                continue
            pw_rows.append(
                compute_pairwise_comparison(
                    table, run_a, run_b, n_boot=n_boot, seed=bootstrap_seed,
                )
            )
        if pw_rows:
            lines += format_pairwise_psnr_tests_md(
                pw_rows,
                n_boot=n_boot,
                seed=bootstrap_seed,
                scipy_available=scipy_stats is not None,
            )

    bootstrap_ci: Optional[Tuple[float, float]] = None
    if bootstrap and oracle_gain_vs_fixed:
        uplift = mean_psnr(oracle_gain_vs_fixed)
        _m, ci_lo, ci_hi, excl = bootstrap_mean_ci(
            oracle_gain_vs_fixed, n_boot=n_boot, seed=bootstrap_seed,
        )
        if ci_lo is not None:
            bootstrap_ci = (ci_lo, ci_hi)
            sig = "yes" if excl else "no"
            lines += [
                f"**Bootstrap oracle uplift vs fixed AdaSteer** "
                f"(per-video, B={n_boot}, seed={bootstrap_seed}): "
                f"mean Δ={uplift:+.3f} dB, 95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}] dB, "
                f"CI excludes 0: {sig}.",
                "",
            ]

    oracle_pick_rows: List[List[str]] = []

    if winners and n_oracle:
        top = sorted(winners.items(), key=lambda x: -x[1])[:8]
        parts = [
            f"`{k}` {v} ({100 * v / n_oracle:.1f}%)" for k, v in top
        ]
        lines += ["**Oracle config picks (top):** " + " · ".join(parts), ""]
        table2_md, oracle_pick_rows = format_oracle_pick_table_md(
            winners,
            gains_by_winner,
            fixed_run=fixed_run,
            overall_gain_vs_fixed=(
                mean_psnr(oracle_gain_vs_fixed) if oracle_gain_vs_fixed else None
            ),
            bootstrap_ci=bootstrap_ci,
        )
        lines += table2_md

    lines += [
        "| Metric | N | Mean | Median | p25 | p75 |",
        "|---|---:|---:|---:|---:|---:|",
        _fmt_stats("Oracle ΔPSNR vs fixed AdaSteer", oracle_gain_vs_fixed),
        _fmt_stats("Oracle ΔPSNR vs NOTTA", oracle_gain_vs_notta),
        "",
    ]

    # OOD quintile stratification
    if ood_quintile:
        lines += [
            "## OOD quintile stratification",
            "",
            f"OOD column: `{OOD_COL}` (low=Q1, high=Q5).",
            "",
        ]
        q_rows: Dict[int, List[str]] = {}
        for vid in vids:
            q = ood_quintile.get(vid)
            if q is not None:
                q_rows.setdefault(q, []).append(vid)

        lines += [
            "### Mean PSNR by OOD quintile and config",
            "",
            "| quintile | N | fixed AdaSteer | oracle-best | best grid run |",
            "|---|---:|---:|---:|---|",
        ]
        quintile_best_run: Dict[int, str] = {}
        for q in sorted(q_rows.keys()):
            vids_q = q_rows[q]
            fixed_vals = [
                table[v][fixed_run]
                for v in vids_q
                if fixed_run in table[v]
            ]
            oracle_vals = []
            best_counts: Dict[str, int] = {}
            for v in vids_q:
                w = oracle_winner(table[v], grid_runs)
                if w and w in table[v]:
                    oracle_vals.append(table[v][w])
                    best_counts[w] = best_counts.get(w, 0) + 1
            best_run = max(best_counts, key=best_counts.get) if best_counts else "—"
            quintile_best_run[q] = best_run
            lines.append(
                f"| Q{q} | {len(vids_q)} | "
                f"{np.mean(fixed_vals):.3f} dB | "
                f"{np.mean(oracle_vals):.3f} dB | `{best_run}` |"
            )
        lines.append("")

        # H9 pattern check: high OOD → more steps / lower LR?
        lines += [
            "### H9 pattern check (high OOD → more steps, lower LR?)",
            "",
            "| quintile | modal oracle run | steps | LR |",
            "|---|---|---:|---:|",
        ]
        for q in sorted(quintile_best_run.keys()):
            rid = quintile_best_run[q]
            steps, lr = parse_run_hparams(rid)
            lr_s = f"{lr:.0e}" if lr is not None else "—"
            steps_s = str(steps) if steps is not None else "—"
            lines.append(f"| Q{q} | `{rid}` | {steps_s} | {lr_s} |")
        lines.append("")

        # Quintile-adaptive deployable policy
        adaptive_psnr: List[float] = []
        for vid in vids:
            q = ood_quintile.get(vid)
            if q is None:
                continue
            rid = quintile_best_run.get(q)
            if rid and rid in table[vid]:
                adaptive_psnr.append(table[vid][rid])
        if adaptive_psnr and fixed_psnr:
            adapt_mean = mean_psnr(adaptive_psnr)
            fixed_mean = mean_psnr(fixed_psnr)
            lines += [
                "### Quintile-adaptive policy (pick modal-best run per OOD quintile)",
                "",
                f"- Mean PSNR: **{adapt_mean:.3f} dB** vs fixed AdaSteer "
                f"{fixed_mean:.3f} dB (**{adapt_mean - fixed_mean:+.3f} dB**).",
                "",
            ]

    lines += [
        "## Interpretation notes",
        "",
        "- Positive oracle uplift vs fixed S10/LR5e-3 means per-video budget "
        "routing has headroom even if population fixed-budget TTA ≈ 0.",
        "- H9 predicts high-OOD quintiles favour *more* steps and *lower* LR; "
        "check the pattern table above (opposite sign would extend the H5 falsification).",
        "",
    ]
    return "\n".join(lines), oracle_pick_rows


def main() -> int:
    ap = argparse.ArgumentParser(description="AdaSteer budget-grid oracle analysis (H9)")
    ap.add_argument("--series-root", type=Path, default=DEFAULT_SERIES)
    ap.add_argument(
        "--baseline-series-root",
        type=Path,
        default=None,
        help=(
            "Standard sweep root containing NOTTA (default: inferred from "
            "--series-root name, e.g. panda_1000v_standard for panda pilots)."
        ),
    )
    ap.add_argument(
        "--baseline-run-id",
        type=str,
        default=NOTTA_RUN_ID,
        help="Baseline method subdir under --baseline-series-root (default: NOTTA).",
    )
    ap.add_argument("--ood-csv", type=Path, default=DEFAULT_OOD)
    ap.add_argument("--fixed-run-id", type=str, default=FIXED_ADA_RUN_ID)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional CSV path for full grid metrics table (default: alongside --output)",
    )
    ap.add_argument(
        "--oracle-picks-csv",
        type=Path,
        default=None,
        help="Optional CSV for Table 2 oracle pick frequency (default: *_picks.csv)",
    )
    ap.add_argument(
        "--oracle-fvd-json",
        type=Path,
        default=None,
        help="Optional fvd.json from run_budget_oracle_fvd.py for oracle FVD/FID row",
    )
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument(
        "--pairwise-tests",
        action="store_true",
        help=(
            "Emit paired PSNR comparison table (S2_LR1e2 vs S10_LR5e3, vs NOTTA, "
            "S10_LR5e3 vs NOTTA) with bootstrap 95%% CI and paired t-test p-values."
        ),
    )
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    args = ap.parse_args()

    if not args.series_root.is_dir():
        print(f"[error] series root not found: {args.series_root}", file=sys.stderr)
        return 2

    runs = discover_runs(args.series_root)
    if not runs:
        print(f"[error] no runs with PSNR found under {args.series_root}", file=sys.stderr)
        return 2

    baseline_series_root = args.baseline_series_root
    if baseline_series_root is None:
        baseline_series_root = _infer_baseline_series_root(args.series_root)

    if args.baseline_run_id not in runs:
        baseline_dir = baseline_series_root / args.baseline_run_id
        if baseline_dir.is_dir():
            baseline_psnr = load_run_psnr(baseline_dir)
            if baseline_psnr:
                runs[args.baseline_run_id] = baseline_dir
                print(
                    f"[info] loaded {len(baseline_psnr)} NOTTA PSNR rows from "
                    f"{baseline_dir}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[warn] baseline dir has no PSNR: {baseline_dir}",
                    file=sys.stderr,
                )
        else:
            print(
                f"[warn] baseline dir missing: {baseline_dir} "
                "(NOTTA comparisons skipped)",
                file=sys.stderr,
            )

    run_ids, table = build_video_table(runs)
    grid_runs = [r for r in run_ids if r not in (NOTTA_RUN_ID,)]
    vids = sorted(table.keys())

    metrics_by_run: Dict[str, Dict[str, Dict[str, float]]] = {}
    for run_id in grid_runs:
        if run_id in runs:
            metrics_by_run[run_id] = load_run_all_metrics(runs[run_id])

    oracle_means = compute_oracle_metric_means(metrics_by_run, grid_runs, vids)

    oracle_fvd_blob: Optional[dict] = None
    oracle_fvd_path = args.oracle_fvd_json
    if oracle_fvd_path is None:
        default_fvd = (
            _REPO_ROOT
            / "sweep_experiment/reports/budget_oracle_fvd/oracle_best_psnr/fvd.json"
        )
        if default_fvd.exists():
            oracle_fvd_path = default_fvd
    if oracle_fvd_path and oracle_fvd_path.exists():
        try:
            with oracle_fvd_path.open(encoding="utf-8") as f:
                oracle_fvd_blob = json.load(f)
            oracle_fvd_blob["_source"] = str(oracle_fvd_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] could not read oracle FVD json: {exc}", file=sys.stderr)

    metrics_header, metrics_rows = build_config_metrics_table(
        runs,
        grid_runs,
        oracle_means=oracle_means,
        oracle_fvd_blob=oracle_fvd_blob,
    )

    ood_q: Dict[str, int] = {}
    if args.ood_csv.exists():
        ood_q = load_ood_quintiles(args.ood_csv)
    else:
        print(f"[warn] OOD CSV not found: {args.ood_csv}", file=sys.stderr)

    report, oracle_pick_rows = build_report(
        series_root=args.series_root,
        baseline_series_root=baseline_series_root
        if args.baseline_run_id in run_ids
        else None,
        baseline_run_id=args.baseline_run_id,
        run_ids=run_ids,
        table=table,
        metrics_by_run=metrics_by_run,
        ood_quintile=ood_q,
        fixed_run=args.fixed_run_id,
        bootstrap=args.bootstrap or args.pairwise_tests,
        n_boot=args.n_boot,
        bootstrap_seed=args.bootstrap_seed,
        pairwise_tests=args.pairwise_tests,
        pairwise_comparisons=DEFAULT_PAIRWISE_COMPARISONS,
        metrics_header=metrics_header,
        metrics_rows=metrics_rows,
        oracle_fvd_blob=oracle_fvd_blob,
    )
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"\nWrote {args.output}", file=sys.stderr)
        csv_path = args.metrics_csv or args.output.with_suffix(".csv")
        write_metrics_csv(csv_path, metrics_header, metrics_rows)
        print(f"Wrote {csv_path}", file=sys.stderr)
        if oracle_pick_rows:
            picks_path = args.oracle_picks_csv or args.output.with_name(
                args.output.stem + "_picks.csv"
            )
            write_metrics_csv(picks_path, oracle_pick_rows[0], oracle_pick_rows[1:])
            print(f"Wrote {picks_path}", file=sys.stderr)
    elif args.metrics_csv:
        write_metrics_csv(args.metrics_csv, metrics_header, metrics_rows)
        print(f"Wrote {args.metrics_csv}", file=sys.stderr)
        if oracle_pick_rows and args.oracle_picks_csv:
            write_metrics_csv(
                args.oracle_picks_csv, oracle_pick_rows[0], oracle_pick_rows[1:],
            )
            print(f"Wrote {args.oracle_picks_csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
