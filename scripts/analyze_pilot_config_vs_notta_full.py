#!/usr/bin/env python3
"""Unified per-config vs NOTTA vs routing metrics for the OOD budget pilot.

Answers two questions in a single report, using ONLY precomputed per-video
metrics (no new generation):

  Task 1 (population): one row per policy — NOTTA, each of the 12 AdaSteer
    configs, oracle-PSNR routing, oracle-VBench routing — with PSNR / SSIM /
    LPIPS / VBench-total / FVD / FID, plus how much routing improves over
    fixing to any single config.

  Task 2 (OOD quintiles): the same NOTTA / 12-config / routing comparison
    broken out across the 5 OOD quintiles, for PSNR and VBench-total.

Data sources (all precomputed):
  * Per-video PSNR/SSIM/LPIPS  -> chunk summaries (load_per_video_metrics)
  * Per-video VBench dims       -> chunk vbench backfill (load_per_video_vbench)
  * Population FVD/FID          -> merged_summary.json (per config)
  * NOTTA baseline (same ids)   -> panda_1000v_standard/NOTTA by canonical id
  * OOD quintiles               -> diffusion_ood_scores.csv

FVD/FID for NOTTA-on-subset and for routing policies require set-level frames
(the pilot ran NO_SAVE_VIDEOS=1), so those cells are reported as "—". Per-config
FVD/FID come straight from each run's merged_summary.

Usage:
    python3 scripts/analyze_pilot_config_vs_notta_full.py \\
        --series-root sweep_experiment/results/panda_ood_budget_pilot \\
        --baseline-series-root sweep_experiment/results/panda_1000v_standard \\
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\
        --output sweep_experiment/reports/per_video_analysis/$(date +%Y-%m-%d)/pilot_config_vs_notta_full.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_adasteer_budget_oracle import (  # noqa: E402
    DEFAULT_OOD,
    DEFAULT_SERIES,
    FIXED_ADA_RUN_ID,
    NOTTA_RUN_ID,
    PILOT_GRID_RUN_ORDER,
    _infer_baseline_series_root,
    build_video_table,
    discover_runs,
    load_merged_summary,
    load_ood_quintiles,
    load_run_psnr,
    oracle_winner,
    parse_run_hparams,
)
from scripts.analyze_adasteer_budget_vbench_oracle import (  # noqa: E402
    load_vbench_by_run,
    vbench_total_score,
)
from scripts.analyze_per_video_tta_gain import load_per_video_metrics  # noqa: E402
from scripts.analyze_per_video_vbench_agreement import (  # noqa: E402
    VBENCH_DIMS,
    select_active_dims,
)

PIXEL_METRICS = ("psnr", "ssim", "lpips")
# Higher is better for these; lower is better for lpips/fvd/fid.
DECIMALS = {"psnr": 3, "ssim": 4, "lpips": 4, "vbench": 4, "fvd": 1, "fid": 1}

ORACLE_PSNR = "ORACLE-PSNR-route"
ORACLE_VBENCH = "ORACLE-VBench-route"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _mean_finite(vals: Sequence[float]) -> Optional[float]:
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    return float(a.mean()) if a.size else None


def _fmt(x: Optional[float], metric: str) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{DECIMALS.get(metric, 3)}f}"


def _fmt_delta(x: Optional[float], metric: str) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{DECIMALS.get(metric, 3)}f}"


def _paired_delta(
    a_map: Dict[str, float],
    b_map: Dict[str, float],
    vids: Sequence[str],
) -> Optional[float]:
    """mean(a - b) over vids where both are finite."""
    diffs: List[float] = []
    for v in vids:
        a = a_map.get(v)
        b = b_map.get(v)
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            continue
        diffs.append(float(a) - float(b))
    return _mean_finite(diffs)


# --------------------------------------------------------------------------- #
# core assembly
# --------------------------------------------------------------------------- #
def build_metric_maps(
    runs: Dict[str, Path],
    grid_runs: Sequence[str],
    include_notta: bool,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """metric -> run_id -> {vid: value} for pixel metrics + vbench.

    Returns dict with keys 'psnr','ssim','lpips','vbench'. Each run_id includes
    the grid configs and (optionally) NOTTA.
    """
    ids = list(grid_runs) + ([NOTTA_RUN_ID] if include_notta and NOTTA_RUN_ID in runs else [])

    pixel: Dict[str, Dict[str, Dict[str, float]]] = {m: {} for m in PIXEL_METRICS}
    for rid in ids:
        per_vid = load_per_video_metrics(runs[rid])
        for m in PIXEL_METRICS:
            pixel[m][rid] = {
                vid: float(row[m])
                for vid, row in per_vid.items()
                if row.get(m) is not None and np.isfinite(row.get(m))
            }

    # VBench total per run.
    vb_by_run = load_vbench_by_run(runs, ids)
    active_dims = select_active_dims(
        {k: v for k, v in vb_by_run.items() if k in grid_runs},
        min_videos=10,
    )
    if not active_dims:
        active_dims = list(VBENCH_DIMS)
    vbench: Dict[str, Dict[str, float]] = {}
    for rid in ids:
        run_vb = vb_by_run.get(rid, {})
        row: Dict[str, float] = {}
        for vid, dmap in run_vb.items():
            tot = vbench_total_score(dmap, active_dims)
            if tot is not None:
                row[vid] = tot
        vbench[rid] = row

    out = dict(pixel)
    out["vbench"] = vbench
    out["_active_dims"] = active_dims  # type: ignore[assignment]
    return out


def routing_realized(
    metric_maps: Dict[str, Dict[str, Dict[str, float]]],
    grid_runs: Sequence[str],
    vids: Sequence[str],
    winner_metric: str,
) -> Dict[str, Dict[str, float]]:
    """Per-video oracle pick by winner_metric; return realized {metric: {vid: val}}."""
    winner_table = metric_maps[winner_metric]
    realized: Dict[str, Dict[str, float]] = {
        m: {} for m in list(PIXEL_METRICS) + ["vbench"]
    }
    for vid in vids:
        row = {rid: winner_table.get(rid, {}).get(vid) for rid in grid_runs}
        row = {k: v for k, v in row.items() if v is not None and np.isfinite(v)}
        w = oracle_winner(row, grid_runs)
        if w is None:
            continue
        for m in realized:
            val = metric_maps[m].get(w, {}).get(vid)
            if val is not None and np.isfinite(val):
                realized[m][vid] = float(val)
    return realized


def _policy_mean(
    metric_maps: Dict[str, Dict[str, Dict[str, float]]],
    realized_psnr: Dict[str, Dict[str, float]],
    realized_vbench: Dict[str, Dict[str, float]],
    policy: str,
    metric: str,
    vids: Sequence[str],
) -> Optional[float]:
    if policy == ORACLE_PSNR:
        m = realized_psnr.get(metric, {})
    elif policy == ORACLE_VBENCH:
        m = realized_vbench.get(metric, {})
    else:
        m = metric_maps[metric].get(policy, {})
    return _mean_finite([m.get(v) for v in vids])


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def build_report(
    *,
    series_root: Path,
    runs: Dict[str, Path],
    grid_runs: List[str],
    vids: List[str],
    metric_maps: Dict[str, Dict[str, Dict[str, float]]],
    realized_psnr: Dict[str, Dict[str, float]],
    realized_vbench: Dict[str, Dict[str, float]],
    merged_by_run: Dict[str, dict],
    ood_quintile: Dict[str, int],
    active_dims: List[str],
) -> str:
    has_notta = NOTTA_RUN_ID in runs
    fixed = FIXED_ADA_RUN_ID
    lines: List[str] = [
        "# Pilot config vs NOTTA vs routing — full metrics",
        "",
        f"**Series:** `{series_root}`",
        f"**Videos (pilot union):** {len(vids)}",
        f"**Grid configs:** {len(grid_runs)} of 12",
        f"**NOTTA baseline:** {'present (by canonical id)' if has_notta else 'MISSING'}",
        f"**VBench active dims:** {len(active_dims)}/{len(VBENCH_DIMS)} "
        f"({', '.join(active_dims)})",
        "",
        "FVD/FID are population (set-level) values from each run's "
        "`merged_summary.json`. NOTTA-subset and routing FVD/FID need saved "
        "frames (pilot ran `NO_SAVE_VIDEOS=1`) and are shown as `—`.",
        "",
        "> **VBench caveat:** the `VBench` column is the *unweighted mean of the 7 "
        "raw dims* (`vbench_total_score`), dominated by the largest-scale dims "
        "(imaging_quality/MUSIQ ~0–100, aesthetic ~0–10). It is **not** the "
        "normalized VBench++ score (~0.77) used in the slides, so do not "
        "cross-compare the two. Use the per-dimension VBench oracle report "
        "(`analyze_adasteer_budget_vbench_oracle.py`) for the real perceptual "
        "story; NOTTA > TTA here likely reflects imaging_quality dropping under "
        "adaptation, not overall quality.",
        "",
    ]

    # --------------------------------------------------------------------- #
    # Task 1 — population table
    # --------------------------------------------------------------------- #
    policies: List[str] = []
    if has_notta:
        policies.append(NOTTA_RUN_ID)
    policies += list(grid_runs) + [ORACLE_PSNR, ORACLE_VBENCH]

    lines += [
        "## Task 1 — Population metrics per policy",
        "",
        "PSNR/SSIM/VBench: higher is better. LPIPS/FVD/FID: lower is better.",
        "",
        "| Policy | steps | LR | N | PSNR (dB) | SSIM | LPIPS | VBench | FVD | FID |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pol in policies:
        steps, lr = parse_run_hparams(pol) if pol in PILOT_GRID_RUN_ORDER else (None, None)
        steps_s = str(steps) if steps is not None else "—"
        lr_s = f"{lr:.0e}" if lr is not None else "—"
        n = len(
            [
                v
                for v in vids
                if metric_maps["psnr"].get(pol if pol not in (ORACLE_PSNR, ORACLE_VBENCH) else fixed, {}).get(v)
                is not None
            ]
        )
        psnr = _policy_mean(metric_maps, realized_psnr, realized_vbench, pol, "psnr", vids)
        ssim = _policy_mean(metric_maps, realized_psnr, realized_vbench, pol, "ssim", vids)
        lpips = _policy_mean(metric_maps, realized_psnr, realized_vbench, pol, "lpips", vids)
        vb = _policy_mean(metric_maps, realized_psnr, realized_vbench, pol, "vbench", vids)
        if pol in (ORACLE_PSNR, ORACLE_VBENCH):
            fvd = fid = None
            n = len(realized_psnr["psnr"] if pol == ORACLE_PSNR else realized_vbench["vbench"])
        elif pol == NOTTA_RUN_ID:
            fvd = fid = None  # subset FVD not available from 999v merged
        else:
            merged = merged_by_run.get(pol, {})
            fvd = merged.get("fvd")
            fid = merged.get("fid")
        label = f"**{pol}**" if pol in (fixed, ORACLE_PSNR, ORACLE_VBENCH, NOTTA_RUN_ID) else f"`{pol}`"
        lines.append(
            f"| {label} | {steps_s} | {lr_s} | {n} | {_fmt(psnr,'psnr')} | "
            f"{_fmt(ssim,'ssim')} | {_fmt(lpips,'lpips')} | {_fmt(vb,'vbench')} | "
            f"{_fmt(fvd,'fvd')} | {_fmt(fid,'fid')} |"
        )
    lines.append("")

    # --------------------------------------------------------------------- #
    # Task 1b — how much routing improves over each fixed policy
    # --------------------------------------------------------------------- #
    lines += [
        "## Task 1b — Routing improvement over each fixed policy (paired)",
        "",
        "`ΔPSNR` = mean(oracle-PSNR-route − fixed policy) on paired videos. "
        "`ΔVBench` = mean(oracle-VBench-route − fixed policy). Positive = routing "
        "beats fixing to that policy.",
        "",
        "| Fixed policy | ΔPSNR vs routing (dB) | ΔVBench vs routing |",
        "|---|---:|---:|",
    ]
    fixed_policies = ([NOTTA_RUN_ID] if has_notta else []) + list(grid_runs)
    for pol in fixed_policies:
        d_psnr = _paired_delta(realized_psnr["psnr"], metric_maps["psnr"].get(pol, {}), vids)
        d_vb = _paired_delta(realized_vbench["vbench"], metric_maps["vbench"].get(pol, {}), vids)
        label = f"**{pol}**" if pol in (fixed, NOTTA_RUN_ID) else f"`{pol}`"
        lines.append(f"| {label} | {_fmt_delta(d_psnr,'psnr')} | {_fmt_delta(d_vb,'vbench')} |")
    lines.append("")

    # --------------------------------------------------------------------- #
    # Task 2 — per OOD quintile
    # --------------------------------------------------------------------- #
    if ood_quintile:
        q_vids: Dict[int, List[str]] = {}
        for v in vids:
            q = ood_quintile.get(v)
            if q is not None:
                q_vids.setdefault(q, []).append(v)
        quintiles = sorted(q_vids.keys())

        for metric, route_policy, route_map in (
            ("psnr", ORACLE_PSNR, realized_psnr.get("psnr", {})),
            ("vbench", ORACLE_VBENCH, realized_vbench.get("vbench", {})),
        ):
            header_cfgs = list(grid_runs)
            col_ids = (([NOTTA_RUN_ID] if has_notta else []) + header_cfgs)
            head = (
                "| Quintile | N | "
                + " | ".join(
                    (NOTTA_RUN_ID if c == NOTTA_RUN_ID else c) for c in col_ids
                )
                + f" | {route_policy} |"
            )
            sep = "|" + "---|" * (len(col_ids) + 3)
            lines += [
                f"## Task 2 — {metric.upper()} by OOD quintile "
                f"(NOTTA vs 12 configs vs routing)",
                "",
                "Q1 = most in-distribution (lowest OOD), Q5 = most OOD. "
                "Mean over videos in the quintile.",
                "",
                head,
                sep,
            ]
            for q in quintiles:
                qv = q_vids[q]
                cells: List[str] = []
                for c in col_ids:
                    mval = _mean_finite([metric_maps[metric].get(c, {}).get(v) for v in qv])
                    cells.append(_fmt(mval, metric))
                rval = _mean_finite([route_map.get(v) for v in qv])
                lines.append(
                    f"| Q{q} | {len(qv)} | " + " | ".join(cells) + f" | {_fmt(rval, metric)} |"
                )
            # ALL row
            cells = []
            for c in col_ids:
                mval = _mean_finite([metric_maps[metric].get(c, {}).get(v) for v in vids])
                cells.append(_fmt(mval, metric))
            rall = _mean_finite([route_map.get(v) for v in vids])
            lines.append(
                f"| **All** | {len(vids)} | " + " | ".join(cells) + f" | {_fmt(rall, metric)} |"
            )
            lines.append("")

        # compact per-quintile summary: best fixed config, NOTTA, routing
        lines += [
            "## Task 2b — Per-quintile summary (best fixed vs NOTTA vs routing)",
            "",
            "Best fixed = grid config with highest mean in that quintile "
            "(PSNR and VBench chosen independently).",
            "",
            "| Quintile | N | metric | NOTTA | best fixed (mean) | routing | "
            "Δ route−NOTTA | Δ route−bestfixed |",
            "|---|---:|---|---:|---|---:|---:|---:|",
        ]
        for metric, route_map in (
            ("psnr", realized_psnr.get("psnr", {})),
            ("vbench", realized_vbench.get("vbench", {})),
        ):
            for q in quintiles:
                qv = q_vids[q]
                notta_m = (
                    _mean_finite([metric_maps[metric].get(NOTTA_RUN_ID, {}).get(v) for v in qv])
                    if has_notta
                    else None
                )
                best_rid = None
                best_val = None
                for c in grid_runs:
                    mval = _mean_finite([metric_maps[metric].get(c, {}).get(v) for v in qv])
                    if mval is None:
                        continue
                    if best_val is None or mval > best_val:
                        best_val = mval
                        best_rid = c
                route_m = _mean_finite([route_map.get(v) for v in qv])
                d_notta = (
                    route_m - notta_m if route_m is not None and notta_m is not None else None
                )
                d_best = (
                    route_m - best_val if route_m is not None and best_val is not None else None
                )
                best_s = f"`{best_rid}` ({_fmt(best_val, metric)})" if best_rid else "—"
                lines.append(
                    f"| Q{q} | {len(qv)} | {metric.upper()} | {_fmt(notta_m, metric)} | "
                    f"{best_s} | {_fmt(route_m, metric)} | {_fmt_delta(d_notta, metric)} | "
                    f"{_fmt_delta(d_best, metric)} |"
                )
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Routing rows are the **oracle** (per-video best config) — the "
        "deployable upper bound. The learned 9-d OOF router realizes only a "
        "fraction of this gap (≈20.8% of the VBench-oracle gap, ≈7.2% of the "
        "PSNR-oracle gap at N=200). Multiply the routing Δ by those factors for "
        "a realistic deployed estimate, or run the deploy-router scripts for "
        "exact realized rows.",
        "- NOTTA is joined by canonical video id from the standard series, so "
        "PSNR/SSIM/LPIPS/VBench are on the *same* videos. FVD/FID for the subset "
        "would need a NOTTA-on-200 set-level eval (frames not saved in pilot).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Pilot config vs NOTTA vs routing (full metrics)")
    ap.add_argument("--series-root", type=Path, default=DEFAULT_SERIES)
    ap.add_argument("--baseline-series-root", type=Path, default=None)
    ap.add_argument("--baseline-run-id", type=str, default=NOTTA_RUN_ID)
    ap.add_argument("--ood-csv", type=Path, default=DEFAULT_OOD)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if not args.series_root.is_dir():
        print(f"[error] series root not found: {args.series_root}", file=sys.stderr)
        return 2

    runs = discover_runs(args.series_root)
    if not runs:
        print(f"[error] no runs with PSNR under {args.series_root}", file=sys.stderr)
        return 2

    baseline_root = args.baseline_series_root or _infer_baseline_series_root(args.series_root)
    if args.baseline_run_id not in runs:
        baseline_dir = baseline_root / args.baseline_run_id
        if baseline_dir.is_dir() and load_run_psnr(baseline_dir):
            runs[args.baseline_run_id] = baseline_dir
            print(f"[info] joined NOTTA baseline from {baseline_dir}", file=sys.stderr)
        else:
            print(
                f"[warn] NOTTA baseline missing/empty: {baseline_dir} "
                "(NOTTA rows will be blank)",
                file=sys.stderr,
            )

    _run_ids, psnr_table = build_video_table(runs)
    grid_runs = [r for r in PILOT_GRID_RUN_ORDER if r in runs]
    if FIXED_ADA_RUN_ID not in grid_runs:
        print(f"[warn] fixed run {FIXED_ADA_RUN_ID} not found in grid", file=sys.stderr)

    # Restrict the analysis set to the pilot videos (those covered by >=1 grid
    # config). Otherwise the NOTTA baseline (full standard series, ~999 videos)
    # would inflate the union and make NOTTA/quintile means cover a different,
    # larger pool than the 200-video config rows.
    grid_set = set(grid_runs)
    vids = sorted(v for v, row in psnr_table.items() if any(r in row for r in grid_set))
    print(
        f"[info] analysis restricted to {len(vids)} pilot videos with grid coverage "
        f"(NOTTA joined on this subset only)",
        file=sys.stderr,
    )

    metric_maps = build_metric_maps(runs, grid_runs, include_notta=NOTTA_RUN_ID in runs)
    active_dims = metric_maps.pop("_active_dims")  # type: ignore[assignment]

    realized_psnr = routing_realized(metric_maps, grid_runs, vids, "psnr")
    realized_vbench = routing_realized(metric_maps, grid_runs, vids, "vbench")

    merged_by_run = {rid: load_merged_summary(runs[rid]) for rid in grid_runs}

    ood_quintile = load_ood_quintiles(args.ood_csv) if args.ood_csv.is_file() else {}
    if not ood_quintile:
        print(f"[warn] no OOD quintiles from {args.ood_csv}", file=sys.stderr)

    report = build_report(
        series_root=args.series_root,
        runs=runs,
        grid_runs=grid_runs,
        vids=vids,
        metric_maps=metric_maps,
        realized_psnr=realized_psnr,
        realized_vbench=realized_vbench,
        merged_by_run=merged_by_run,
        ood_quintile=ood_quintile,
        active_dims=list(active_dims),
    )
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"\nWrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
