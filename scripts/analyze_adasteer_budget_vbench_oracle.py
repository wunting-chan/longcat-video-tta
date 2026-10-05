#!/usr/bin/env python3
"""Budget-grid sliding-config oracle analysis for VBench++ (mirrors H9 PSNR script).

Per video, pick the AdaSteer step×LR config that maximizes:
  * VBench++ total (mean of available dims)
  * Each individual VBench dimension

Reports population uplift vs fixed S10/LR5e-3 and vs NOTTA, OOD-quintile modal
winners, quintile-adaptive deployable policy, PSNR-oracle vs VBench-oracle
agreement, and per-config population VBench means.

Requires per-video VBench under each grid run (``run_vbench_backfill.py``).

Usage:
    python3 scripts/analyze_adasteer_budget_vbench_oracle.py --bootstrap \\
        --series-root sweep_experiment/results/panda_ood_budget_pilot \\
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\
        --output sweep_experiment/reports/per_video_analysis/$(date +%Y-%m-%d)/adasteer_budget_vbench_oracle_pilot.md
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    bootstrap_mean_ci,
    build_video_table,
    discover_runs,
    load_ood_quintiles,
    load_run_psnr,
    oracle_winner,
    parse_run_hparams,
)
from scripts.analyze_per_video_vbench_agreement import (  # noqa: E402
    VBENCH_DIMS,
    count_videos_with_all_dims,
    load_per_video_vbench,
    select_active_dims,
    vbench_dim_counts,
)
from scripts.summarize_vbench_population_per_video import DIM_SHORT  # noqa: E402

ORACLE_TARGETS: Tuple[str, ...] = ("vbench_total",) + tuple(VBENCH_DIMS)


def filter_vbench_grid_runs(
    vb_by_run: Dict[str, Dict[str, Dict[str, float]]],
    grid_runs: Sequence[str],
    *,
    min_videos: int,
) -> Tuple[List[str], List[str]]:
    """Drop configs with no usable per-video VBench (e.g. missing backfill)."""
    included: List[str] = []
    excluded: List[str] = []
    for rid in grid_runs:
        vb = vb_by_run.get(rid, {})
        counts = vbench_dim_counts(vb)
        if max(counts.values(), default=0) >= min_videos:
            included.append(rid)
        else:
            excluded.append(rid)
    return included, excluded


def grid_runs_for_dim(
    vb_by_run: Dict[str, Dict[str, Dict[str, float]]],
    grid_runs: Sequence[str],
    dim: str,
    *,
    min_videos: int,
) -> List[str]:
    return [
        rid for rid in grid_runs
        if vbench_dim_counts(vb_by_run.get(rid, {})).get(dim, 0) >= min_videos
    ]


def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def _fmt_delta(x: Optional[float], nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{nd}f}"


def vbench_total_score(dmap: Dict[str, float], dims: Sequence[str]) -> Optional[float]:
    vals = [dmap.get(d) for d in dims]
    if not vals or any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
        return None
    return float(np.mean(vals))


def load_vbench_by_run(
    runs: Dict[str, Path],
    run_ids: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rid in run_ids:
        if rid not in runs:
            continue
        out[rid] = load_per_video_vbench(runs[rid])
    return out


def build_score_table(
    vb_by_run: Dict[str, Dict[str, Dict[str, float]]],
    run_ids: Sequence[str],
    vids: Sequence[str],
    dims: Sequence[str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]]:
    """Return (total_table, dim_table) keyed by video then run_id."""
    total: Dict[str, Dict[str, float]] = {vid: {} for vid in vids}
    by_dim: Dict[str, Dict[str, Dict[str, float]]] = {
        d: {vid: {} for vid in vids} for d in dims
    }
    for rid in run_ids:
        run_vb = vb_by_run.get(rid, {})
        for vid in vids:
            dmap = run_vb.get(vid, {})
            if not dmap:
                continue
            tot = vbench_total_score(dmap, dims)
            if tot is not None:
                total[vid][rid] = tot
            for d in dims:
                v = dmap.get(d)
                if v is not None and not math.isnan(v):
                    by_dim[d][vid][rid] = float(v)
    return total, by_dim


def analyze_oracle(
    table: Dict[str, Dict[str, float]],
    grid_runs: Sequence[str],
    vids: Sequence[str],
    *,
    fixed_run: str,
    notta_table: Optional[Dict[str, Dict[str, float]]] = None,
) -> dict:
    """Oracle on one scalar metric (total or single dim)."""
    oracle_vals: List[float] = []
    fixed_vals: List[float] = []
    notta_vals: List[float] = []
    gain_vs_fixed: List[float] = []
    gain_vs_notta: List[float] = []
    winners: Dict[str, int] = {}

    for vid in vids:
        row = table.get(vid, {})
        w = oracle_winner(row, grid_runs)
        if w is None:
            continue
        winners[w] = winners.get(w, 0) + 1
        ov = row[w]
        oracle_vals.append(ov)
        if fixed_run in row:
            fv = row[fixed_run]
            fixed_vals.append(fv)
            gain_vs_fixed.append(ov - fv)
        if notta_table is not None:
            nr = notta_table.get(vid, {})
            if NOTTA_RUN_ID in nr:
                nv = nr[NOTTA_RUN_ID]
                notta_vals.append(nv)
                gain_vs_notta.append(ov - nv)

    def _mean(a: List[float]) -> Optional[float]:
        if not a:
            return None
        return float(np.mean(a))

    return {
        "n": len(oracle_vals),
        "oracle_mean": _mean(oracle_vals),
        "fixed_mean": _mean(fixed_vals),
        "notta_mean": _mean(notta_vals),
        "gain_vs_fixed": _mean(gain_vs_fixed),
        "gain_vs_notta": _mean(gain_vs_notta),
        "gain_vs_fixed_list": gain_vs_fixed,
        "winners": winners,
    }


def quintile_modal_winners(
    table: Dict[str, Dict[str, float]],
    grid_runs: Sequence[str],
    vids: Sequence[str],
    ood_quintile: Dict[str, int],
) -> Dict[int, str]:
    q_vids: Dict[int, List[str]] = {}
    for vid in vids:
        q = ood_quintile.get(vid)
        if q is not None:
            q_vids.setdefault(q, []).append(vid)
    out: Dict[int, str] = {}
    for q, vlist in q_vids.items():
        counts: Dict[str, int] = {}
        for vid in vlist:
            w = oracle_winner(table.get(vid, {}), grid_runs)
            if w:
                counts[w] = counts.get(w, 0) + 1
        if counts:
            out[q] = max(counts, key=counts.get)
    return out


def quintile_adaptive_mean(
    table: Dict[str, Dict[str, float]],
    vids: Sequence[str],
    ood_quintile: Dict[str, int],
    quintile_best: Dict[int, str],
) -> Optional[float]:
    vals: List[float] = []
    for vid in vids:
        q = ood_quintile.get(vid)
        if q is None:
            continue
        rid = quintile_best.get(q)
        if rid and rid in table.get(vid, {}):
            vals.append(table[vid][rid])
    return float(np.mean(vals)) if vals else None


def psnr_vbench_agreement(
    psnr_table: Dict[str, Dict[str, float]],
    vbench_table: Dict[str, Dict[str, float]],
    grid_runs: Sequence[str],
    vids: Sequence[str],
) -> dict:
    agree = 0
    total = 0
    for vid in vids:
        pr = psnr_table.get(vid, {})
        vr = vbench_table.get(vid, {})
        pw = oracle_winner(pr, grid_runs)
        vw = oracle_winner(vr, grid_runs)
        if pw is None or vw is None:
            continue
        total += 1
        if pw == vw:
            agree += 1
    return {"n": total, "agree": agree, "rate": (agree / total if total else None)}


def coverage_report(
    vb_by_run: Dict[str, Dict[str, Dict[str, float]]],
    grid_runs: Sequence[str],
) -> List[str]:
    lines = [
        "## VBench coverage (per grid config)",
        "",
        "| Config | Videos (any dim) | All 7 dims |",
        "|---|---:|---:|",
    ]
    for rid in grid_runs:
        vb = vb_by_run.get(rid, {})
        n_any = len(vb)
        n_all = count_videos_with_all_dims(vb, VBENCH_DIMS)
        lines.append(f"| `{rid}` | {n_any} | {n_all} |")
    lines.append("")
    return lines


def build_report(
    *,
    series_root: Path,
    grid_runs: List[str],
    vids: List[str],
    active_dims: List[str],
    stats_by_target: Dict[str, dict],
    quintile_by_target: Dict[str, Dict[int, str]],
    adaptive_by_target: Dict[str, Optional[float]],
    fixed_means_by_target: Dict[str, Optional[float]],
    agreement: dict,
    vb_by_run: Dict[str, Dict[str, Dict[str, float]]],
    ood_quintile: Dict[str, int],
    bootstrap: bool,
    n_boot: int,
    seed: int,
    psnr_oracle_gain: Optional[float],
    excluded_grid_runs: Sequence[str],
) -> str:
    lines = [
        "# AdaSteer budget-grid VBench++ oracle analysis",
        "",
        f"**Series:** `{series_root}`",
        f"**Fixed AdaSteer:** `{FIXED_ADA_RUN_ID}` (S10/LR=5e-3).",
        f"**VBench grid configs:** {len(grid_runs)} of 12 "
        f"({', '.join(f'`{r}`' for r in grid_runs)})",
        f"**Active VBench dims:** {', '.join(active_dims)} ({len(active_dims)}/{len(VBENCH_DIMS)})",
        f"**Union N:** {len(vids)} videos with PSNR; oracle denominators vary by VBench coverage.",
        "",
    ]
    if excluded_grid_runs:
        lines += [
            f"**Excluded from VBench oracle (no backfill):** "
            f"{', '.join(f'`{r}`' for r in excluded_grid_runs)}",
            "",
        ]

    lines += coverage_report(vb_by_run, grid_runs)

    lines += [
        "## Population routing uplift (VBench-driven oracle)",
        "",
        "Oracle picks the grid config with **max VBench** per video (not PSNR). "
        "Δ columns are mean(oracle − baseline) on paired videos.",
        "",
        "| Oracle target | N | Oracle mean | Fixed mean | NOTTA mean | Δ vs fixed | Δ vs NOTTA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in ORACLE_TARGETS:
        if target not in stats_by_target:
            continue
        st = stats_by_target[target]
        label = "VBench total" if target == "vbench_total" else DIM_SHORT.get(target, target)
        lines.append(
            f"| {label} | {st['n']} | {_fmt(st['oracle_mean'])} | "
            f"{_fmt(st['fixed_mean'])} | {_fmt(st['notta_mean'])} | "
            f"{_fmt_delta(st['gain_vs_fixed'])} | {_fmt_delta(st['gain_vs_notta'])} |"
        )
    lines.append("")

    if bootstrap:
        lines += [
            "### Bootstrap 95% CI — VBench-total oracle Δ vs fixed AdaSteer",
            "",
            "| Stat | Value |",
            "|---|---:|",
        ]
        st = stats_by_target.get("vbench_total", {})
        gains = st.get("gain_vs_fixed_list") or []
        if gains:
            mean, lo, hi, excl = bootstrap_mean_ci(gains, n_boot=n_boot, seed=seed)
            lines.append(f"| Mean Δ | {_fmt_delta(mean)} |")
            lines.append(f"| 95% CI | [{_fmt_delta(lo)}, {_fmt_delta(hi)}] |")
            lines.append(f"| CI excludes 0 | {'yes' if excl else 'no'} |")
        else:
            lines.append("| — | no paired data |")
        lines.append("")

    lines += [
        "## PSNR oracle vs VBench-total oracle (config agreement)",
        "",
        f"- Videos with both oracles: **{agreement['n']}**",
        f"- Same config picked: **{agreement['agree']}** "
        f"({100 * agreement['rate']:.1f}%)" if agreement.get("rate") is not None else "",
        "",
    ]
    if psnr_oracle_gain is not None:
        vt = stats_by_target.get("vbench_total", {})
        lines += [
            f"- PSNR-oracle uplift vs fixed (from PSNR script): **{psnr_oracle_gain:+.3f} dB**",
            f"- VBench-total oracle Δ vs fixed: **{_fmt_delta(vt.get('gain_vs_fixed'))}**",
            "",
        ]

    for target in ORACLE_TARGETS:
        if target not in stats_by_target:
            continue
        label = "VBench total" if target == "vbench_total" else DIM_SHORT.get(target, target)
        st = stats_by_target[target]
        winners = st.get("winners") or {}
        if not winners:
            continue
        n = st["n"]
        lines += [
            f"## Oracle pick frequency — {label}",
            "",
            "| Config | Picks | % |",
            "|---|---:|---:|",
        ]
        for rid, cnt in sorted(winners.items(), key=lambda x: -x[1]):
            lines.append(f"| `{rid}` | {cnt} | {100 * cnt / n:.1f}% |")
        lines.append("")

    if ood_quintile:
        lines += [
            "## OOD quintile stratification (VBench-total oracle)",
            "",
            f"OOD column: `mean_diffusion_loss_caption` (low=Q1, high=Q5).",
            "",
            "| Quintile | N | Fixed VBench | Oracle VBench | Modal config | Steps | LR |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
        q_vids: Dict[int, List[str]] = {}
        for vid in vids:
            q = ood_quintile.get(vid)
            if q is not None:
                q_vids.setdefault(q, []).append(vid)

        total_table = stats_by_target.get("_total_table", {})
        fixed_run = FIXED_ADA_RUN_ID
        q_best = quintile_by_target.get("vbench_total", {})
        for q in sorted(q_vids.keys()):
            vlist = q_vids[q]
            fixed_v: List[float] = []
            oracle_v: List[float] = []
            for vid in vlist:
                row = total_table.get(vid, {})
                if fixed_run in row:
                    fixed_v.append(row[fixed_run])
                w = oracle_winner(row, grid_runs)
                if w and w in row:
                    oracle_v.append(row[w])
            rid = q_best.get(q, "—")
            steps, lr = parse_run_hparams(rid) if rid != "—" else (None, None)
            lr_s = f"{lr:.0e}" if lr is not None else "—"
            steps_s = str(steps) if steps is not None else "—"
            lines.append(
                f"| Q{q} | {len(vlist)} | {_fmt(np.mean(fixed_v) if fixed_v else None)} | "
                f"{_fmt(np.mean(oracle_v) if oracle_v else None)} | `{rid}` | {steps_s} | {lr_s} |"
            )
        lines.append("")

        lines += [
            "### Quintile-adaptive VBench policy (modal oracle config per quintile)",
            "",
            "| Target | Adaptive mean | Fixed mean | Δ vs fixed |",
            "|---|---:|---:|---:|",
        ]
        for target in ORACLE_TARGETS:
            if target not in adaptive_by_target:
                continue
            label = "VBench total" if target == "vbench_total" else DIM_SHORT.get(target, target)
            adapt = adaptive_by_target[target]
            fixed = fixed_means_by_target.get(target)
            delta = (adapt - fixed) if adapt is not None and fixed is not None else None
            lines.append(
                f"| {label} | {_fmt(adapt)} | {_fmt(fixed)} | {_fmt_delta(delta)} |"
            )
        lines.append("")

        lines += [
            "### Per-dimension modal oracle config by OOD quintile",
            "",
        ]
        for target in VBENCH_DIMS:
            if target not in quintile_by_target:
                continue
            lines.append(f"**{DIM_SHORT.get(target, target)}** (`{target}`):")
            lines.append("")
            lines.append("| Quintile | Modal config | Steps | LR |")
            lines.append("|---|---|---:|---:|")
            for q in sorted(quintile_by_target[target].keys()):
                rid = quintile_by_target[target][q]
                steps, lr = parse_run_hparams(rid)
                lr_s = f"{lr:.0e}" if lr is not None else "—"
                steps_s = str(steps) if steps is not None else "—"
                lines.append(f"| Q{q} | `{rid}` | {steps_s} | {lr_s} |")
            lines.append("")

    lines += [
        "## Interpretation notes",
        "",
        "- If VBench-oracle Δ ≪ PSNR-oracle Δ (in comparable units), perceptual "
        "routing needs different features than pixel routing (cf. method-level "
        "PSNR vs VBench-total oracle gap on 999v).",
        "- Compare quintile modal configs here to the PSNR H9 table: high-OOD may "
        "prefer different steps/LR for VBench than for PSNR.",
        "- Quintile-adaptive policy captures only a fraction of oracle headroom when "
        "modal configs differ across quintiles but within-quintile variance is large.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Budget-grid VBench++ oracle analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--series-root", type=Path, default=DEFAULT_SERIES)
    ap.add_argument("--baseline-series-root", type=Path, default=None)
    ap.add_argument("--baseline-run-id", type=str, default=NOTTA_RUN_ID)
    ap.add_argument("--ood-csv", type=Path, default=DEFAULT_OOD)
    ap.add_argument("--fixed-run-id", type=str, default=FIXED_ADA_RUN_ID)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--min-videos", type=int, default=10, help="Min per-video scores per dim to include")
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

    run_ids, psnr_table = build_video_table(runs)
    grid_runs = [r for r in run_ids if r not in (NOTTA_RUN_ID,) and r.startswith("S")]
    order = {rid: i for i, rid in enumerate(PILOT_GRID_RUN_ORDER)}
    grid_runs = sorted(grid_runs, key=lambda r: (order.get(r, 999), r))
    vids = sorted(psnr_table.keys())

    vb_by_run = load_vbench_by_run(runs, run_ids)
    grid_runs_all = list(grid_runs)
    grid_runs, excluded_grid_runs = filter_vbench_grid_runs(
        vb_by_run, grid_runs, min_videos=args.min_videos,
    )
    if excluded_grid_runs:
        print(
            f"[warn] excluding {len(excluded_grid_runs)} configs with no VBench: "
            f"{', '.join(excluded_grid_runs)}",
            file=sys.stderr,
        )
    if not grid_runs:
        print(
            "[error] no grid configs with per-video VBench — "
            "run submit_budget_pilot_vbench_backfill.sh first",
            file=sys.stderr,
        )
        for rid in grid_runs_all:
            counts = vbench_dim_counts(vb_by_run.get(rid, {}))
            print(f"  {rid}: {counts}", file=sys.stderr)
        return 2

    active_dims = select_active_dims(
        {k: v for k, v in vb_by_run.items() if k in grid_runs},
        min_videos=args.min_videos,
    )
    if not active_dims:
        print(
            "[error] no VBench dims shared across included grid configs — "
            "check partial backfills",
            file=sys.stderr,
        )
        for rid in grid_runs:
            counts = vbench_dim_counts(vb_by_run.get(rid, {}))
            print(f"  {rid}: {counts}", file=sys.stderr)
        return 2

    total_table, dim_tables = build_score_table(vb_by_run, grid_runs, vids, active_dims)

    notta_vb = vb_by_run.get(NOTTA_RUN_ID, {})
    notta_total_table: Dict[str, Dict[str, float]] = {}
    for vid in vids:
        tot = vbench_total_score(notta_vb.get(vid, {}), active_dims)
        if tot is not None:
            notta_total_table[vid] = {NOTTA_RUN_ID: tot}

    ood_quintile = load_ood_quintiles(args.ood_csv) if args.ood_csv.is_file() else {}

    stats_by_target: Dict[str, dict] = {}
    stats_by_target["_total_table"] = total_table

    stats_by_target["vbench_total"] = analyze_oracle(
        total_table,
        grid_runs,
        vids,
        fixed_run=args.fixed_run_id,
        notta_table=notta_total_table,
    )
    for d in active_dims:
        dim_runs = grid_runs_for_dim(vb_by_run, grid_runs, d, min_videos=args.min_videos)
        stats_by_target[d] = analyze_oracle(
            dim_tables[d],
            dim_runs,
            vids,
            fixed_run=args.fixed_run_id,
            notta_table=None,
        )

    quintile_by_target: Dict[str, Dict[int, str]] = {}
    adaptive_by_target: Dict[str, Optional[float]] = {}
    fixed_means_by_target: Dict[str, Optional[float]] = {}

    tables_for_policy = {"vbench_total": total_table, **dim_tables}
    for target, tbl in tables_for_policy.items():
        if target not in stats_by_target and target != "_total_table":
            continue
        if target.startswith("_"):
            continue
        if target == "vbench_total":
            policy_runs = grid_runs
        elif target in VBENCH_DIMS:
            policy_runs = grid_runs_for_dim(
                vb_by_run, grid_runs, target, min_videos=args.min_videos,
            )
        else:
            policy_runs = grid_runs
        quintile_by_target[target] = quintile_modal_winners(
            tbl, policy_runs, vids, ood_quintile,
        )
        adaptive_by_target[target] = quintile_adaptive_mean(tbl, vids, ood_quintile, quintile_by_target[target])
        fixed_vals = [
            tbl[vid][args.fixed_run_id]
            for vid in vids
            if args.fixed_run_id in tbl.get(vid, {})
        ]
        fixed_means_by_target[target] = float(np.mean(fixed_vals)) if fixed_vals else None

    agreement = psnr_vbench_agreement(psnr_table, total_table, grid_runs, vids)

    psnr_oracle_gain: Optional[float] = None
    psnr_gains: List[float] = []
    for vid in vids:
        row = psnr_table.get(vid, {})
        w = oracle_winner(row, grid_runs)
        if w and args.fixed_run_id in row:
            psnr_gains.append(row[w] - row[args.fixed_run_id])
    if psnr_gains:
        psnr_oracle_gain = float(np.mean(psnr_gains))

    report = build_report(
        series_root=args.series_root,
        grid_runs=grid_runs,
        vids=vids,
        active_dims=active_dims,
        stats_by_target=stats_by_target,
        quintile_by_target=quintile_by_target,
        adaptive_by_target=adaptive_by_target,
        fixed_means_by_target=fixed_means_by_target,
        agreement=agreement,
        vb_by_run=vb_by_run,
        ood_quintile=ood_quintile,
        bootstrap=args.bootstrap,
        n_boot=args.n_boot,
        seed=args.bootstrap_seed,
        psnr_oracle_gain=psnr_oracle_gain,
        excluded_grid_runs=excluded_grid_runs,
    )

    out = args.output
    if out is None:
        out = (
            _REPO_ROOT
            / "sweep_experiment/reports/per_video_analysis"
            / "adasteer_budget_vbench_oracle_pilot.md"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")

    picks_csv = out.with_name(out.stem + "_oracle_picks.csv")
    with picks_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target", "config", "picks", "pct"])
        for target in ORACLE_TARGETS:
            st = stats_by_target.get(target, {})
            n = st.get("n") or 0
            for rid, cnt in sorted((st.get("winners") or {}).items(), key=lambda x: -x[1]):
                w.writerow([target, rid, cnt, f"{100 * cnt / n:.2f}" if n else ""])
    print(f"Wrote {picks_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
