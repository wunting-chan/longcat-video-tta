#!/usr/bin/env python3
"""Quantify per-video routing win magnitudes for presentation slides.

Reads ``per_video_gains.csv`` from analyze_per_video_tta_gain.py and reports:
  - Oracle (best PSNR) uplift vs always-NOTTA / ADA / LoRA
  - Head-to-head win magnitudes when LoRA vs AdaSteer wins on ΔPSNR
  - Oracle winner breakdowns (NOTTA / ADA / LoRA)

Usage:
    python scripts/analyze_routing_win_magnitudes.py
    python scripts/analyze_routing_win_magnitudes.py \\
        --gains-csv sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv \\
        --output sweep_experiment/reports/per_video_analysis/2026-06-09/routing_win_magnitudes.md
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_GAINS = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv"
)

BASELINE = "NOTTA"
ADA = "ADA"
LORA = "LORA_R8_TTA"


def _f(row: dict, key: str) -> float:
    v = row.get(key, "")
    if v is None or v == "":
        return float("nan")
    return float(v)


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


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 5000,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[bool]]:
    """Per-video bootstrap CI for the mean.

    Returns (mean, ci_lo, ci_hi, ci_excludes_zero).
    """
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


def _fmt_stats(label: str, arr: Sequence[float]) -> str:
    n, mean, med, p25, p75 = _stats(arr)
    if n == 0:
        return f"| {label} | 0 | — | — | — | — |"
    return (
        f"| {label} | {n} | {mean:.3f} dB | {med:.3f} dB | "
        f"{p25:.3f} dB | {p75:.3f} dB |"
    )


def load_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def oracle_winner(row: dict) -> str:
    psnrs = {
        BASELINE: _f(row, f"{BASELINE}_psnr"),
        ADA: _f(row, f"{ADA}_psnr"),
        LORA: _f(row, f"{LORA}_psnr"),
    }
    return max(psnrs, key=lambda k: psnrs[k])


def oracle_win_margin(row: dict) -> Tuple[str, float]:
    """Return (winner, PSNR margin over second-best method)."""
    psnrs = {
        BASELINE: _f(row, f"{BASELINE}_psnr"),
        ADA: _f(row, f"{ADA}_psnr"),
        LORA: _f(row, f"{LORA}_psnr"),
    }
    winner = max(psnrs, key=lambda k: psnrs[k])
    second_best = max(v for k, v in psnrs.items() if k != winner)
    return winner, psnrs[winner] - second_best


def _oracle_winner_margin_table(
    rows: List[dict],
) -> Tuple[List[str], Dict[str, List[float]]]:
    """Per-winner win margins (winner PSNR − second-best PSNR)."""
    margins: Dict[str, List[float]] = {BASELINE: [], ADA: [], LORA: []}
    for r in rows:
        winner, margin = oracle_win_margin(r)
        margins[winner].append(margin)

    display = {
        BASELINE: "NOTTA",
        ADA: "AdaSteer",
        LORA: "LoRA",
    }
    n = len(rows)
    lines = [
        "## Oracle winner breakdown (win margin over second-best)",
        "",
        "Win margin = winner absolute PSNR − second-best absolute PSNR on that video.",
        "",
        "| Oracle winner | N | Share | Mean win margin | Median margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in (ADA, BASELINE, LORA):
        arr = margins[key]
        cnt = len(arr)
        share = 100.0 * cnt / n if n else 0.0
        if cnt == 0:
            lines.append(f"| {display[key]} | 0 | 0.0% | — | — |")
            continue
        _, mean, med, _, _ = _stats(arr)
        lines.append(
            f"| {display[key]} | {cnt} | {share:.1f}% | {mean:.3f} dB | {med:.3f} dB |"
        )
    lines.append("")
    return lines, margins


def build_report(rows: List[dict], bootstrap: bool = False,
                 n_boot: int = 5000, bootstrap_seed: int = 42) -> str:
    n = len(rows)
    lines: List[str] = [
        "# Routing win magnitudes",
        "",
        f"**N = {n}** videos (exclude corrupt/missing clips from denominator).",
        "",
    ]

    notta_psnr = [_f(r, f"{BASELINE}_psnr") for r in rows]
    ada_psnr = [_f(r, f"{ADA}_psnr") for r in rows]
    lora_psnr = [_f(r, f"{LORA}_psnr") for r in rows]
    ada_d = [_f(r, f"{ADA}_dpsnr") for r in rows]
    lora_d = [_f(r, f"{LORA}_dpsnr") for r in rows]

    oracle_psnr: List[float] = []
    oracle_gain: List[float] = []
    winners: Dict[str, int] = {BASELINE: 0, ADA: 0, LORA: 0}

    for r in rows:
        w = oracle_winner(r)
        winners[w] += 1
        p = {
            BASELINE: _f(r, f"{BASELINE}_psnr"),
            ADA: _f(r, f"{ADA}_psnr"),
            LORA: _f(r, f"{LORA}_psnr"),
        }[w]
        oracle_psnr.append(p)
        oracle_gain.append(p - _f(r, f"{BASELINE}_psnr"))

    def mean_psnr(arr: Iterable[float]) -> float:
        a = np.asarray(list(arr), dtype=float)
        return float(np.mean(a))

    skip_ada = [
        max(_f(r, f"{BASELINE}_psnr"), _f(r, f"{LORA}_psnr"))
        if _f(r, f"{ADA}_dpsnr") <= 0
        else _f(r, f"{ADA}_psnr")
        for r in rows
    ]
    skip_both = [
        _f(r, f"{BASELINE}_psnr")
        if max(_f(r, f"{ADA}_dpsnr"), _f(r, f"{LORA}_dpsnr")) <= 0
        else max(_f(r, f"{BASELINE}_psnr"), _f(r, f"{ADA}_psnr"), _f(r, f"{LORA}_psnr"))
        for r in rows
    ]

    oracle_uplift = mean_psnr(oracle_psnr) - mean_psnr(notta_psnr)
    lines += [
        "## Oracle routing uplift",
        "",
        "| Policy | Mean PSNR | Δ vs always-NOTTA |",
        "|---|---:|---:|",
        f"| Always NOTTA | {mean_psnr(notta_psnr):.3f} dB | 0.000 dB |",
        f"| Always AdaSteer | {mean_psnr(ada_psnr):.3f} dB | "
        f"{mean_psnr(ada_psnr) - mean_psnr(notta_psnr):+.3f} dB |",
        f"| Always LoRA | {mean_psnr(lora_psnr):.3f} dB | "
        f"{mean_psnr(lora_psnr) - mean_psnr(notta_psnr):+.3f} dB |",
        f"| **Oracle (best PSNR)** | **{mean_psnr(oracle_psnr):.3f} dB** | "
        f"**{oracle_uplift:+.3f} dB** |",
        f"| Skip AdaSteer if ΔPSNR ≤ 0 | {mean_psnr(skip_ada):.3f} dB | "
        f"{mean_psnr(skip_ada) - mean_psnr(notta_psnr):+.3f} dB |",
        f"| Skip both TTA if ΔPSNR ≤ 0 | {mean_psnr(skip_both):.3f} dB | "
        f"{mean_psnr(skip_both) - mean_psnr(notta_psnr):+.3f} dB |",
        "",
    ]
    if bootstrap:
        _m, ci_lo, ci_hi, excl_zero = bootstrap_mean_ci(
            oracle_gain, n_boot=n_boot, seed=bootstrap_seed,
        )
        if ci_lo is not None and ci_hi is not None:
            sig = "yes" if excl_zero else "no"
            lines += [
                f"**Bootstrap oracle uplift** (per-video, B={n_boot}, "
                f"seed={bootstrap_seed}): "
                f"mean Δ={oracle_uplift:+.3f} dB, "
                f"95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}] dB, "
                f"CI excludes 0: {sig}.",
                "",
            ]
    lines += [
        f"**Oracle picks:** NOTTA {winners[BASELINE]} ({100*winners[BASELINE]/n:.1f}%) · "
        f"AdaSteer {winners[ADA]} ({100*winners[ADA]/n:.1f}%) · "
        f"LoRA {winners[LORA]} ({100*winners[LORA]/n:.1f}%)",
        "",
    ]
    margin_lines, _ = _oracle_winner_margin_table(rows)
    lines += margin_lines
    lines += [
        "| Metric | N | Mean | Median | p25 | p75 |",
        "|---|---:|---:|---:|---:|---:|",
        _fmt_stats("Oracle ΔPSNR vs NOTTA", oracle_gain),
        "",
        f"{sum(1 for g in oracle_gain if g > 0)} / {n} videos ({100*sum(1 for g in oracle_gain if g > 0)/n:.1f}%) "
        "have oracle gain > 0.",
        "",
    ]

    lora_beats_ada = [i for i, r in enumerate(rows) if lora_d[i] > ada_d[i]]
    ada_beats_lora = [i for i, r in enumerate(rows) if ada_d[i] > lora_d[i]]

    lines += [
        "## Head-to-head",
        "",
        f"| LoRA beats AdaSteer (ΔPSNR) | {len(lora_beats_ada)} | {100*len(lora_beats_ada)/n:.1f}% |",
        f"| AdaSteer beats LoRA | {len(ada_beats_lora)} | {100*len(ada_beats_lora)/n:.1f}% |",
        "",
        "## When LoRA beats AdaSteer on ΔPSNR",
        "",
        "| Metric | N | Mean | Median | p25 | p75 |",
        "|---|---:|---:|---:|---:|---:|",
        _fmt_stats("LoRA ΔPSNR vs NOTTA", [lora_d[i] for i in lora_beats_ada]),
        _fmt_stats("Margin: LoRA Δ − Ada Δ", [lora_d[i] - ada_d[i] for i in lora_beats_ada]),
        "",
        "## When AdaSteer beats LoRA on ΔPSNR",
        "",
        "| Metric | N | Mean | Median | p25 | p75 |",
        "|---|---:|---:|---:|---:|---:|",
        _fmt_stats("AdaSteer ΔPSNR vs NOTTA", [ada_d[i] for i in ada_beats_lora]),
        _fmt_stats("Margin: Ada Δ − LoRA Δ", [ada_d[i] - lora_d[i] for i in ada_beats_lora]),
        "",
    ]

    notta_wins = [r for r in rows if oracle_winner(r) == BASELINE]
    lines += [
        "## When NOTTA wins oracle (best absolute PSNR)",
        "",
        "| Metric | N | Mean | Median | p25 | p75 |",
        "|---|---:|---:|---:|---:|---:|",
        _fmt_stats(
            "NOTTA absolute PSNR",
            [_f(r, f"{BASELINE}_psnr") for r in notta_wins],
        ),
        _fmt_stats(
            "Margin over AdaSteer PSNR",
            [_f(r, f"{BASELINE}_psnr") - _f(r, f"{ADA}_psnr") for r in notta_wins],
        ),
        _fmt_stats(
            "Margin over LoRA PSNR",
            [_f(r, f"{BASELINE}_psnr") - _f(r, f"{LORA}_psnr") for r in notta_wins],
        ),
        _fmt_stats(
            "Margin over best alternative",
            [
                _f(r, f"{BASELINE}_psnr")
                - max(_f(r, f"{ADA}_psnr"), _f(r, f"{LORA}_psnr"))
                for r in notta_wins
            ],
        ),
        "",
    ]

    ada_wins = [r for r in rows if oracle_winner(r) == ADA]
    lora_wins = [r for r in rows if oracle_winner(r) == LORA]
    lines += [
        "## AdaSteer oracle wins",
        "",
        "| Metric | N | Mean | Median | p25 | p75 |",
        "|---|---:|---:|---:|---:|---:|",
        _fmt_stats("Ada ΔPSNR vs NOTTA", [_f(r, f"{ADA}_dpsnr") for r in ada_wins]),
        _fmt_stats(
            "Margin over LoRA PSNR",
            [_f(r, f"{ADA}_psnr") - _f(r, f"{LORA}_psnr") for r in ada_wins],
        ),
        "",
        "## LoRA oracle wins",
        "",
        "| Metric | N | Mean | Median | p25 | p75 |",
        "|---|---:|---:|---:|---:|---:|",
        _fmt_stats("LoRA ΔPSNR vs NOTTA", [_f(r, f"{LORA}_dpsnr") for r in lora_wins]),
        _fmt_stats(
            "Margin over AdaSteer PSNR",
            [_f(r, f"{LORA}_psnr") - _f(r, f"{ADA}_psnr") for r in lora_wins],
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Routing win magnitude stats for slides")
    ap.add_argument("--gains-csv", type=Path, default=DEFAULT_GAINS)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--bootstrap", action="store_true",
                    help="Add per-video bootstrap 95%% CI for oracle mean uplift.")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    args = ap.parse_args()

    if not args.gains_csv.exists():
        print(f"[error] gains CSV not found: {args.gains_csv}", file=sys.stderr)
        return 2

    rows = load_rows(args.gains_csv)
    report = build_report(
        rows,
        bootstrap=args.bootstrap,
        n_boot=args.n_boot,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"\nWrote {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
