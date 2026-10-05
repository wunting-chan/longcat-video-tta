#!/usr/bin/env python3
"""Build oracle policy dirs (if needed) and compute FVD for each policy.

Uses ``eval_fvd.py`` with the precomputed GT I3D cache so scores match the
headline ``panda_1000v_standard`` online-FVD protocol (~154–158 for NOTTA).

Usage:
    python sweep_experiment/scripts/run_phase1_oracle_fvd.py \\
        --gt-cache gt_caches/panda_1000_longcat.npz

    # Reuse existing policy dirs under phase1_oracle_fvd/:
    python sweep_experiment/scripts/run_phase1_oracle_fvd.py --skip-build
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Matches panda_1000v_standard / online FVD accumulator in common.py.
DEFAULT_NUM_COND_FRAMES = 14
DEFAULT_NUM_GEN_FRAMES = 14

DEFAULT_POLICIES = [
    "always_notta",
    "always_ada",
    "always_lora",
    "oracle_best_psnr",
    "oracle_skip_ada_nonpos",
    "oracle_skip_both_nonpos",
    "oracle_top50_ada_dpsnr",
]

MIN_LINKED_DEFAULT = 900


def _clear_stale_fvd_outputs(output_root: Path, policies: list) -> None:
    """Remove prior fvd.json so a failed run cannot be mistaken for success."""
    for policy in policies:
        fvd_json = output_root / policy / "fvd.json"
        if fvd_json.exists():
            fvd_json.unlink()
    summary_path = output_root / "fvd_summary.json"
    if summary_path.exists():
        summary_path.unlink()


def _read_linked_count(output_root: Path, policy: str) -> int:
    manifest_path = output_root / policy / "manifest.json"
    if not manifest_path.exists():
        return 0
    try:
        blob = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    return int(blob.get("linked_videos") or 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-1 oracle FVD batch eval")
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("sweep_experiment/reports/phase1_oracle_fvd"),
    )
    ap.add_argument(
        "--gains-csv",
        type=Path,
        default=Path(
            "sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv"
        ),
    )
    ap.add_argument(
        "--series-root",
        type=Path,
        default=Path("sweep_experiment/results/panda_1000v_standard"),
    )
    ap.add_argument(
        "--gt-cache",
        type=Path,
        default=Path("gt_caches/panda_1000_longcat.npz"),
    )
    ap.add_argument(
        "--policies", nargs="*", default=None,
        help=f"Policies to evaluate (default: {DEFAULT_POLICIES})",
    )
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--force", action="store_true", help="Pass --force to eval_fvd")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--min-linked", type=int, default=MIN_LINKED_DEFAULT,
        help="Skip FVD eval for policies with fewer symlinked videos",
    )
    args = ap.parse_args()

    policies = args.policies or DEFAULT_POLICIES

    if not args.skip_build:
        _clear_stale_fvd_outputs(args.output_root, policies)
        build_cmd = [
            sys.executable,
            str(_REPO_ROOT / "sweep_experiment/scripts/build_oracle_policy_dirs.py"),
            "--gains-csv", str(args.gains_csv),
            "--series-root", str(args.series_root),
            "--output-root", str(args.output_root),
            "--clean",
            "--min-linked", str(args.min_linked),
            "--policies", *policies,
        ]
        print("Building policy dirs...")
        print(" ", " ".join(build_cmd))
        rc = subprocess.call(build_cmd, cwd=str(_REPO_ROOT))
        if rc != 0:
            print(
                "ERROR: build_oracle_policy_dirs failed — "
                "FVD eval aborted (stale fvd.json cleared).",
                file=sys.stderr,
            )
            return rc

    if not args.gt_cache.exists():
        print(f"ERROR: GT cache missing: {args.gt_cache}", file=sys.stderr)
        print("  Run: python sweep_experiment/scripts/precompute_gt_features.py ...",
              file=sys.stderr)
        return 2

    eval_script = _REPO_ROOT / "sweep_experiment/scripts/eval_fvd.py"
    summary = {}
    exit_code = 0
    for policy in policies:
        gen_dir = args.output_root / policy / "videos"
        out_json = args.output_root / policy / "fvd.json"
        linked = _read_linked_count(args.output_root, policy)
        if linked < args.min_linked:
            print(
                f"SKIP {policy}: only {linked} symlinks "
                f"(need >= {args.min_linked}); not writing fvd.json",
                file=sys.stderr,
            )
            exit_code = 1
            continue
        if not gen_dir.is_dir() or not any(gen_dir.glob("*.mp4")):
            print(f"SKIP {policy}: no videos in {gen_dir}", file=sys.stderr)
            exit_code = 1
            continue

        cmd = [
            sys.executable, str(eval_script),
            "--gen-dir", str(gen_dir),
            "--gt-cache", str(args.gt_cache),
            "--num-cond-frames", str(DEFAULT_NUM_COND_FRAMES),
            "--num-gen-frames", str(DEFAULT_NUM_GEN_FRAMES),
            "--min-videos", "256",
            "--output", str(out_json),
            "--device", args.device,
        ]
        if args.force:
            cmd.append("--force")

        print(f"\n=== FVD: {policy} ===")
        print(" ", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(_REPO_ROOT))
        if rc != 0:
            print(f"ERROR: eval_fvd failed for {policy}", file=sys.stderr)
            return rc

        with out_json.open(encoding="utf-8") as f:
            blob = json.load(f)
        summary[policy] = blob.get("fvd")

    summary_path = args.output_root / "fvd_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Oracle FVD summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {summary_path}")
    return exit_code if exit_code else 0


if __name__ == "__main__":
    raise SystemExit(main())
