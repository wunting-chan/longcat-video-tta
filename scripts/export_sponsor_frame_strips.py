#!/usr/bin/env python3
"""Paper-style film strips for the sponsor note. Generic row labels only.

Two input modes:

* ``--root`` (default, cluster): read cite-128 mp4s, sample eight timestamps,
  and draw a per-row motion trace under the frames.
* ``--from-sheet``: re-lay the frames of an existing 4-column contact sheet
  (used on the laptop, where the mp4s are not synced). No motion trace.

Per-row VBench numbers come from the cite-128 per-clip CSV. Does not name or
render unpublished methods.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "sweep_experiment/reports/briefing_charts_raw/2026-09-09_slim/wan/cite128_per_video.csv"
OUT = ROOT / "sweep_experiment/reports/paper_tables/sponsor_summer_2026_figures"

TIMES = (1.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 29.0)
SHEET_TIMES = (1.0, 10.0, 20.0, 29.0)
# 9 prefix latents = 33 pixel frames of real video at 16 fps.
CONTEXT_S = 33 / 16.0

BLUE = "#4D6BFE"
BLUE_DK = "#2B3FB8"
SLATE = "#8A94A8"
SLATE_LT = "#C5CCD9"
INK = "#1F2329"
MUTE = "#6B7280"
RED = "#E0685A"
FILM = "#15171C"
HOLE = "#3A3E48"

ROWS = [
    ("notta", "Published few-step baseline", SLATE),
    ("rolling_notta", "Published streaming baseline", SLATE),
    ("sf_always_search", "Always-on seed search", BLUE),
]

# clip, title, systems shown, output name
CLIPS = [
    ("panda_0003", "Clip A · seed search starts motion; both rows share the same late-tail texture collapse",
     ("notta", "sf_always_search"), "fig11_filmstrip_clip_a"),
    ("panda_0001", "Clip B · neither system moves; seed search does not invent motion",
     ("notta", "sf_always_search"), "fig12_filmstrip_clip_b"),
    ("panda_0031", "Clip C · counted as 'living', but the scene breaks down",
     ("notta", "sf_always_search"), "fig13_filmstrip_clip_c"),
    ("panda_0018", "Clip D · all three systems; seed search keeps the scene and its colour to 29 s",
     ("notta", "rolling_notta", "sf_always_search"), "fig14_filmstrip_clip_d"),
]


def _metrics(clip: str) -> dict[str, dict[str, float]]:
    import csv

    keys = ("imaging_quality", "subject_consistency", "temporal_flickering", "dynamic_degree")
    with open(CSV, newline="") as fh:
        return {
            row["method"]: {k: float(row[k]) for k in keys}
            for row in csv.DictReader(fh)
            if row["clip"] == clip
        }


def _chip(m: dict[str, float]) -> str:
    living = "living" if m["dynamic_degree"] >= 0.5 else "still"
    return (
        f"IQ {m['imaging_quality']:.1f}  ·  subject {m['subject_consistency']:.2f}\n"
        f"flicker {m['temporal_flickering']:.3f}  ·  {living}"
    )


# ------------------------------------------------------------------ inputs


def _find(root: Path, clip: str, method: str) -> Path | None:
    pat = re.compile(rf"_{re.escape(clip)}_h30s_{re.escape(method)}_s\d+\.mp4$")
    hits = sorted(p for p in root.glob(f"**/*{clip}*.mp4") if pat.search(p.name))
    return hits[0] if hits else None


def _read_video(mp4: Path) -> tuple[list[np.ndarray], float]:
    import imageio.v3 as iio

    try:
        fps = float(iio.immeta(mp4).get("fps") or 16.0)
    except Exception:
        fps = 16.0
    frames = list(iio.imread(mp4, index=None))
    return frames, fps


def _motion_trace(frames: list[np.ndarray], fps: float) -> tuple[np.ndarray, np.ndarray]:
    small = np.stack([f[::8, ::8].astype(np.float32).mean(-1) for f in frames])
    diff = np.abs(np.diff(small, axis=0)).mean(axis=(1, 2))
    k = max(1, int(round(fps / 2)))
    diff = np.convolve(diff, np.ones(k) / k, mode="same")
    t = (np.arange(len(diff)) + 1) / fps
    return t, diff


def _sheet_tiles(png: Path, rows: int, cols: int) -> list[list[np.ndarray]]:
    from PIL import Image

    img = np.asarray(Image.open(png).convert("RGB"))
    ink = (img < 235).any(-1)

    def runs(profile: np.ndarray, thresh: float, min_len: int) -> list[tuple[int, int]]:
        on = profile > thresh
        out, start = [], None
        for i, v in enumerate(np.append(on, False)):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= min_len:
                    out.append((start, i))
                start = None
        return out

    row_runs = runs(ink.mean(1), 0.5, 60)
    if len(row_runs) != rows:
        raise RuntimeError(f"{png.name}: found {len(row_runs)} frame rows, expected {rows}")
    tiles = []
    for y0, y1 in row_runs:
        col_runs = runs(ink[y0:y1].mean(0), 0.02, 60)
        if len(col_runs) != cols:
            raise RuntimeError(f"{png.name}: found {len(col_runs)} frame columns, expected {cols}")
        tiles.append([img[y0 + 2:y1 - 2, x0 + 2:x1 - 2] for x0, x1 in col_runs])
    return tiles


# ------------------------------------------------------------------ render


def _render(title, rows, times, out_name, traces=None, note=None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    n_rows, n_cols = len(rows), len(times)
    fh, fw = rows[0][3][0].shape[:2]
    aspect = fh / fw

    label_w = 2.95
    tile_w = 1.55
    gap = 0.07
    film_pad = 0.17
    tile_h = tile_w * aspect
    strip_h = tile_h + 2 * film_pad
    row_gap = 0.34
    ruler_h = 0.62
    title_h = 0.42
    trace_h = 1.05 if traces else 0.0
    note_h = 0.34 if note else 0.0
    strip_w = n_cols * tile_w + (n_cols - 1) * gap + 2 * 0.12
    W = label_w + strip_w + 0.15
    H = title_h + ruler_h + n_rows * strip_h + (n_rows - 1) * row_gap + trace_h + note_h + 0.1

    fig = plt.figure(figsize=(W, H))
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, W)
    canvas.set_ylim(H, 0)
    canvas.axis("off")

    canvas.text(0.1, title_h * 0.55, title, fontsize=12, fontweight="bold", color=INK, va="center")

    x0 = label_w + 0.12
    col_x = [x0 + c * (tile_w + gap) for c in range(n_cols)]
    ruler_y = title_h + ruler_h
    n_ctx = sum(t <= CONTEXT_S for t in times)
    if n_ctx:
        cx1 = col_x[n_ctx - 1] + tile_w
        canvas.add_patch(Rectangle((col_x[0], ruler_y - 0.44), cx1 - col_x[0], 0.12, color=SLATE, lw=0))
        canvas.text((col_x[0] + cx1) / 2, ruler_y - 0.5, "context frames (real)", ha="center",
                    va="bottom", fontsize=8.5, color=MUTE)
    gx0 = col_x[n_ctx] if n_ctx < n_cols else None
    if gx0 is not None:
        gx1 = col_x[-1] + tile_w
        canvas.add_patch(Rectangle((gx0, ruler_y - 0.44), gx1 - gx0, 0.12, color=BLUE, lw=0))
        canvas.text((gx0 + gx1) / 2, ruler_y - 0.5, "generated continuation  →", ha="center",
                    va="bottom", fontsize=8.5, color=BLUE_DK, fontweight="bold")
    for c, t in enumerate(times):
        canvas.text(col_x[c] + tile_w / 2, ruler_y - 0.1, f"{t:.0f} s", ha="center", va="bottom",
                    fontsize=9, color=INK)

    y = ruler_y
    for r, (label, color, chip, frames) in enumerate(rows):
        canvas.add_patch(FancyBboxPatch((label_w, y), strip_w, strip_h,
                                        boxstyle="round,pad=0,rounding_size=0.06", fc=FILM, ec="none"))
        hole_w, hole_h = 0.09, 0.07
        hx = label_w + 0.1
        while hx < label_w + strip_w - 0.1:
            for hy in (y + (film_pad - hole_h) / 2, y + strip_h - (film_pad + hole_h) / 2):
                canvas.add_patch(FancyBboxPatch((hx, hy), hole_w, hole_h,
                                                boxstyle="round,pad=0,rounding_size=0.015",
                                                fc=HOLE, ec="none"))
            hx += 0.2
        for c, frame in enumerate(frames):
            ax = fig.add_axes([col_x[c] / W, 1 - (y + film_pad + tile_h) / H, tile_w / W, tile_h / H])
            ax.imshow(frame, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        canvas.add_patch(Rectangle((0.1, y + 0.1), 0.06, strip_h - 0.2, color=color, lw=0))
        canvas.text(0.26, y + 0.14, label, fontsize=9.5, fontweight="bold", color=INK, va="top")
        canvas.text(0.26, y + 0.48, chip, fontsize=8, color=MUTE, va="top", linespacing=1.5)
        y += strip_h + row_gap

    if traces:
        y_t = y - row_gap + 0.18
        ax = fig.add_axes([col_x[0] / W, 1 - (y_t + trace_h - 0.28) / H,
                           (col_x[-1] + tile_w - col_x[0]) / W, (trace_h - 0.35) / H])
        for (label, color), (t, v) in traces:
            ax.plot(t, v, color=color, lw=1.4, label=label, ls="--" if color == INK else "-")
        ax.axvspan(0, CONTEXT_S, color=SLATE_LT, alpha=0.4, lw=0)
        for tt in times:
            ax.axvline(tt, color=MUTE, lw=0.5, ls=":")
        ax.set_xlim(0, 30)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("seconds", fontsize=8, labelpad=1)
        ax.legend(fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 0.98),
                  ncol=len(traces), borderaxespad=0)
        canvas.text(label_w - 0.1, y_t + (trace_h - 0.35) / 2, "frame-to-frame\nchange", ha="right",
                    va="center", fontsize=8, color=MUTE)
    if note:
        canvas.text(label_w, H - 0.2, note, fontsize=8, color=MUTE, va="center")

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{out_name}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / out_name}.png/.pdf")


# ------------------------------------------------------------------ modes


def from_videos(root: Path) -> int:
    missing = 0
    for clip, title, systems, name in CLIPS:
        stats = _metrics(clip)
        rows, traces = [], []
        for method in systems:
            label, color = next((l, c) for m, l, c in ROWS if m == method)
            mp4 = _find(root, clip, method)
            if mp4 is None:
                print(f"missing {clip} / {method} under {root}")
                missing += 1
                rows = None
                break
            frames, fps = _read_video(mp4)
            picks = [frames[min(int(round(t * fps)), len(frames) - 1)] for t in TIMES]
            rows.append((label, color, _chip(stats[method]), picks))
            trace_color = BLUE if method == "sf_always_search" else (SLATE if method == "notta" else INK)
            traces.append(((label, trace_color), _motion_trace(frames, fps)))
        if rows:
            _render(title, rows, TIMES, name, traces=traces)
    return 1 if missing else 0


def from_sheets(sheet_dir: Path) -> int:
    sheets = {"panda_0003": "fig7_frames_became_living.png", "panda_0001": "fig8_frames_stayed_static.png"}
    for clip, title, systems, name in CLIPS:
        if clip not in sheets:
            continue
        tiles = _sheet_tiles(sheet_dir / sheets[clip], rows=2, cols=4)
        stats = _metrics(clip)
        rows = [
            ("Published few-step baseline", SLATE, _chip(stats["notta"]), tiles[0]),
            ("Always-on seed search", BLUE, _chip(stats["sf_always_search"]), tiles[1]),
        ]
        _render(title, rows, SHEET_TIMES, name)
    return 0


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/scratch/wc3013/longcat-video-tta/wan_experiment/results/v2v_panda_caption_128v")
    ap.add_argument("--from-sheet", type=Path, default=None,
                    help="directory holding the old 4-column sheets (laptop fallback)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    OUT = args.out
    if args.from_sheet:
        return from_sheets(args.from_sheet)
    return from_videos(Path(args.root))


if __name__ == "__main__":
    raise SystemExit(main())
