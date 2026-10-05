#!/usr/bin/env python3
"""
Build a per-SEGMENT Panda-70M retrieval pool from the existing
panda_2048_480p videos + the Panda-70M segment-level metadata.

Why this exists (and why it doesn't re-download anything from YouTube):
  - The previous Panda pool builder downloaded full source videos via
    yt-dlp and emitted a single row per source with a bracketed multi-
    caption string (e.g. "['caption A', 'caption B', 'caption C']").
    That gives 2048 unique caption embeddings -- not enough retrieval
    diversity, and the bracketed strings are noisy as similarity
    queries.
  - Panda-70M's source metadata stores PER-SEGMENT timestamps and
    captions for every video as parallel nested lists. The 800K-row
    `panda70m_training_2m.csv` we already have on disk has 3 segments
    per row.
  - We already downloaded the full 480p source videos. ffmpeg-cutting
    those locally into per-segment clips with their own clean
    per-segment captions costs 0 YouTube downloads, ~1-2 h of CPU
    time, and gives us 2048 x 3 = ~6,144 retrieval entries with
    UCF-grade single-string captions.

Schema this builder writes (matches `ucf101_pool_max/metadata.csv`):
  index, filename, videoID, source_video_id, video_path, caption,
  category, chunk_index, chunk_start_sec, duration, fps, frames,
  width, height

  - `videoID`         = "<source_videoID>_seg<idx>"  (unique row key)
  - `source_video_id` = the original Panda/YouTube videoID
                        (same for every segment of one source video).
                        common.py uses this for same-source exclusion
                        at retrieval time, identical to the UCF flow.
  - `chunk_index`     = 0..N-1 within the source video.
  - `chunk_start_sec` = segment start time, parsed from the
                        Panda-70M timestamp column.

Quality filters (defaults match the Panda-70M paper):
  - desirable_filtering == "desirable"
        (drop low_desirable_score, screen_in_screen, etc.)
  - min_duration <= seg <= max_duration
  - matching_score >= min_score   (captions with cosine alignment to
                                   the auto-generated CLIP score; the
                                   paper recommends >= 0.42)

Resume:
  - Already-encoded segments (mp4 exists, > 100 KB) are skipped.
  - Sources whose `source_video_id` already appears in manifest.jsonl
    are not re-cut (probed but not re-extracted).

Pipeline:
  1. Read `<source_pool>/metadata.csv` -> map videoID -> existing
     480p mp4 path on disk.
  2. Stream the Panda-70M segment-level CSV; for each row whose
     videoID matches our pool, parse the parallel nested lists,
     filter, and queue ffmpeg cut jobs.
  3. Run ffmpeg cuts in parallel (ThreadPoolExecutor; one worker per
     segment, since the input file already lives on disk).
  4. ffprobe each output to populate fps / frames / width / height,
     append to manifest.jsonl, rebuild metadata.csv.
  5. (Optional) Validate min frame count, write valid_subset.csv,
     truncate metadata.csv to first `--target-valid` valid rows.

CPU-only job. Submit via datasets/build_panda_segment_pool.sbatch.

Usage:
  python scripts/build_panda_segment_pool.py \
      --source-pool /scratch/wc3013/longcat-video-tta/datasets/panda_2048_480p \
      --source-metadata /scratch/wc3013/longcat-video-tta/datasets/panda_pool_10k/panda70m_training_2m.csv \
      --new-dataset /scratch/wc3013/longcat-video-tta/datasets/panda_segment_pool \
      --workers 16
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

# Match the UCF chunker so common.py treats both pools identically.
TARGET_WIDTH_DEFAULT = 832
TARGET_HEIGHT_DEFAULT = 480
RESIZE_SIZE_THRESHOLD_BYTES = 100 * 1024
MIN_SEGMENT_SECONDS = 2.0
MAX_SEGMENT_SECONDS = 20.0
MIN_MATCHING_SCORE_DEFAULT = 0.0  # Panda paper recommends >= 0.42


# ============================================================================
# Source-pool loading
# ============================================================================

def _load_source_pool_index(source_pool: Path) -> Dict[str, Path]:
    """Map videoID -> existing 480p mp4 path under source_pool."""
    meta_path = source_pool / "metadata.csv"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found", file=sys.stderr)
        sys.exit(2)

    out: Dict[str, Path] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "videoID" not in (reader.fieldnames or []):
            print(f"ERROR: 'videoID' column missing in {meta_path}",
                  file=sys.stderr)
            sys.exit(2)
        for row in reader:
            vid = row.get("videoID")
            if not vid:
                continue
            path_field = row.get("path") or row.get("video_path") or ""
            p = Path(path_field) if path_field else None
            if p is None or not p.exists():
                fname = row.get("filename") or ""
                candidate = source_pool / "videos" / fname
                if candidate.exists():
                    p = candidate
            if p is not None and p.exists():
                out[vid] = p
    return out


# ============================================================================
# Panda-70M segment-metadata parsing
# ============================================================================

_TS_RE = re.compile(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$")


def _parse_timestamp(ts: str) -> float:
    """Parse '0:00:16.300' (HH:MM:SS.fff) -> seconds float."""
    m = _TS_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Bad timestamp: {ts!r}")
    h, m_, s_ = m.groups()
    return int(h) * 3600 + int(m_) * 60 + float(s_)


def _safe_literal_list(raw: str) -> Optional[list]:
    """ast.literal_eval a stringified list, returning None on parse error."""
    if not raw:
        return None
    try:
        v = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    return v if isinstance(v, list) else None


def _iter_segments_for_video(
    row: Dict[str, str],
    *,
    require_desirable: bool,
    min_score: float,
    min_seg_sec: float,
    max_seg_sec: float,
) -> Iterable[Dict[str, object]]:
    """Yield dicts of {seg_idx, start_sec, end_sec, duration, caption,
    matching_score} for each well-formed, filter-passing segment in a
    Panda-70M training_2m row."""
    timestamps = _safe_literal_list(row.get("timestamp", ""))
    captions = _safe_literal_list(row.get("caption", ""))
    scores = _safe_literal_list(row.get("matching_score", ""))
    filters = _safe_literal_list(row.get("desirable_filtering", "")) or []
    if not timestamps or not captions:
        return
    n = min(len(timestamps), len(captions))
    for idx in range(n):
        ts_pair = timestamps[idx]
        caption = captions[idx]
        if not (isinstance(ts_pair, (list, tuple)) and len(ts_pair) == 2):
            continue
        if not isinstance(caption, str) or not caption.strip():
            continue
        try:
            start_sec = _parse_timestamp(str(ts_pair[0]))
            end_sec = _parse_timestamp(str(ts_pair[1]))
        except ValueError:
            continue
        duration = end_sec - start_sec
        if duration < min_seg_sec or duration > max_seg_sec:
            continue
        score = 0.0
        if scores is not None and idx < len(scores):
            try:
                score = float(scores[idx])
            except (TypeError, ValueError):
                score = 0.0
        if score < min_score:
            continue
        if require_desirable and idx < len(filters):
            f_label = filters[idx]
            if isinstance(f_label, str) and f_label != "desirable":
                continue
        yield {
            "seg_idx": idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration": duration,
            "caption": caption.strip(),
            "matching_score": score,
        }


# ============================================================================
# ffmpeg cutting + ffprobe
# ============================================================================

def _encode_segment(
    src: Path,
    dst: Path,
    start_sec: float,
    duration: float,
    target_w: int,
    target_h: int,
    crf: int = 23,
    preset: str = "fast",
) -> bool:
    """Encode a single segment with -ss <start> -i <src> -t <duration>."""
    if dst.exists() and dst.stat().st_size > RESIZE_SIZE_THRESHOLD_BYTES:
        return True
    tmp = dst.with_suffix(".chunk.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-ss", f"{start_sec:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-vf", f"scale={target_w}:{target_h}",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-an",
        str(tmp),
    ]
    try:
        res = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False
    if res.returncode != 0 or not tmp.exists() or \
            tmp.stat().st_size <= RESIZE_SIZE_THRESHOLD_BYTES:
        tmp.unlink(missing_ok=True)
        return False
    try:
        os.replace(str(tmp), str(dst))
    except OSError:
        tmp.unlink(missing_ok=True)
        return False
    return True


def _probe_chunk_stream(path: Path) -> Tuple[float, float, int, int, int]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "default=noprint_wrappers=1:nokey=0",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=15)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return (0.0, 0.0, 0, 0, 0)
    kv: Dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    w = int(kv.get("width") or 0)
    h = int(kv.get("height") or 0)
    fps_raw = kv.get("r_frame_rate", "0/1")
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    try:
        duration = float(kv.get("duration", "0") or 0.0)
    except ValueError:
        duration = 0.0
    try:
        nb_frames = int(kv.get("nb_frames", "0") or 0)
    except ValueError:
        nb_frames = 0
    if nb_frames == 0 and fps > 0 and duration > 0:
        nb_frames = int(round(fps * duration))
    return (duration, fps, nb_frames, w, h)


# ============================================================================
# Build pipeline
# ============================================================================

def _build_pool(
    source_pool: Path,
    source_metadata: Path,
    new_dir: Path,
    *,
    require_desirable: bool,
    min_score: float,
    min_seg_sec: float,
    max_seg_sec: float,
    target_w: int,
    target_h: int,
    num_workers: int,
    max_segments_per_source: int,
    limit: int,
) -> int:
    new_videos = new_dir / "videos"
    new_videos.mkdir(parents=True, exist_ok=True)
    manifest_path = new_dir / "manifest.jsonl"

    # 1. videoID -> existing 480p mp4 path
    print("[1/5] Indexing source pool ...")
    pool_index = _load_source_pool_index(source_pool)
    print(f"      {len(pool_index)} source videos found in {source_pool}")
    if not pool_index:
        print("ERROR: source pool is empty.", file=sys.stderr)
        return 0

    # 2. Resume support: collect already-processed source_video_ids
    existing_sources: Set[str] = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = e.get("source_video_id")
                if not sid:
                    vid = str(e.get("videoID", ""))
                    m = re.match(r"^(.+)_seg\d+$", vid)
                    sid = m.group(1) if m else vid
                if sid:
                    existing_sources.add(sid)
        print(f"      Resume: {len(existing_sources)} sources already in "
              f"manifest.jsonl (will skip)")

    # 3. Stream Panda-70M metadata, build segment task list
    print(f"[2/5] Streaming Panda-70M metadata: {source_metadata}")
    segment_tasks: List[Dict[str, object]] = []
    matched_sources = 0
    seen_sources_in_csv: Set[str] = set()
    rows_seen = 0
    t0 = time.time()
    with open(source_metadata, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_seen += 1
            vid = row.get("videoID")
            if not vid or vid in seen_sources_in_csv:
                continue
            seen_sources_in_csv.add(vid)
            if vid not in pool_index:
                continue
            if vid in existing_sources:
                continue
            matched_sources += 1
            src_path = pool_index[vid]
            seg_count = 0
            for seg in _iter_segments_for_video(
                row,
                require_desirable=require_desirable,
                min_score=min_score,
                min_seg_sec=min_seg_sec,
                max_seg_sec=max_seg_sec,
            ):
                if max_segments_per_source > 0 and seg_count >= max_segments_per_source:
                    break
                seg_count += 1
                segment_id = f"{vid}_seg{seg['seg_idx']}"
                dst = new_videos / f"{segment_id}.mp4"
                segment_tasks.append({
                    "source_video_id": vid,
                    "src_path": src_path,
                    "segment_id": segment_id,
                    "seg_idx": seg["seg_idx"],
                    "start_sec": seg["start_sec"],
                    "duration": seg["duration"],
                    "caption": seg["caption"],
                    "matching_score": seg["matching_score"],
                    "dst": dst,
                })
            if rows_seen % 50_000 == 0:
                print(f"      ... {rows_seen} rows scanned, "
                      f"{matched_sources} matched, "
                      f"{len(segment_tasks)} segment tasks queued",
                      flush=True)

    print(f"      Done streaming. "
          f"{rows_seen} rows scanned in {time.time()-t0:.1f}s.")
    print(f"      Sources matched (after resume filter): {matched_sources}")
    print(f"      Total segment tasks queued:            {len(segment_tasks)}")

    if limit > 0 and len(segment_tasks) > limit:
        segment_tasks = segment_tasks[:limit]
        print(f"      --limit applied: trimmed to {len(segment_tasks)} tasks")

    if not segment_tasks:
        print("Nothing to encode (resume up-to-date or no matches).")
        return 0

    # 4. Parallel ffmpeg encoding
    print(f"[3/5] Encoding {len(segment_tasks)} segments "
          f"with {num_workers} workers ...")
    t_enc = time.time()
    n_done = 0
    n_ok = 0
    last_print = t_enc

    def _job(task: Dict[str, object]) -> Tuple[Dict[str, object], bool]:
        ok = _encode_segment(
            src=task["src_path"], dst=task["dst"],
            start_sec=float(task["start_sec"]),
            duration=float(task["duration"]),
            target_w=target_w, target_h=target_h,
        )
        return task, ok

    encoded: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=num_workers,
                            thread_name_prefix="pandaseg") as ex:
        futures = [ex.submit(_job, t) for t in segment_tasks]
        for fut in as_completed(futures):
            task, ok = fut.result()
            n_done += 1
            if ok:
                n_ok += 1
                encoded.append(task)
            now = time.time()
            if now - last_print >= 15.0 or n_done == len(segment_tasks):
                rate = n_done / max(1.0, now - t_enc)
                eta = (len(segment_tasks) - n_done) / max(1e-6, rate)
                print(f"      [{n_done}/{len(segment_tasks)}] "
                      f"ok={n_ok} rate={rate:.1f}/s eta={eta/60:.1f}min",
                      flush=True)
                last_print = now

    print(f"      Encoded {n_ok}/{len(segment_tasks)} segments in "
          f"{(time.time()-t_enc)/60:.1f} min.")

    # 5. Append to manifest, rebuild metadata.csv
    print("[4/5] Probing encoded files + appending to manifest.jsonl ...")
    metadata_rows: List[Dict[str, str]] = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metadata_rows.append({
                    "index": str(e.get("index", "")),
                    "filename": str(e.get("filename", "")),
                    "videoID": str(e.get("videoID", "")),
                    "source_video_id": str(
                        e.get("source_video_id", "") or e.get("videoID", "")
                    ),
                    "video_path": str(
                        e.get("path", "") or e.get("video_path", "")
                    ),
                    "caption": str(e.get("caption", "")),
                    "category": str(e.get("category", "")),
                    "chunk_index": str(e.get("chunk_index", "0")),
                    "chunk_start_sec": str(e.get("chunk_start_sec", "0.0")),
                    "duration": str(e.get("duration", "")),
                    "fps": str(e.get("fps", "")),
                    "frames": str(e.get("frames", "")),
                    "width": str(e.get("width", "")),
                    "height": str(e.get("height", "")),
                })

    next_index = len(metadata_rows)
    n_appended = 0
    with open(manifest_path, "a", encoding="utf-8") as f_man:
        for task in encoded:
            dst: Path = task["dst"]
            if not dst.exists():
                continue
            duration, fps, n_frames, width, height = _probe_chunk_stream(dst)
            row = {
                "index": str(next_index),
                "filename": dst.name,
                "videoID": str(task["segment_id"]),
                "source_video_id": str(task["source_video_id"]),
                "video_path": str(dst),
                "caption": str(task["caption"]),
                "category": "panda",
                "chunk_index": str(task["seg_idx"]),
                "chunk_start_sec": f"{float(task['start_sec']):.3f}",
                "duration": f"{duration:.3f}",
                "fps": f"{fps:.3f}",
                "frames": str(n_frames),
                "width": str(width),
                "height": str(height),
            }
            metadata_rows.append(row)
            f_man.write(json.dumps({
                "index": next_index,
                "filename": dst.name,
                "videoID": task["segment_id"],
                "source_video_id": task["source_video_id"],
                "caption": task["caption"],
                "category": "panda",
                "chunk_index": int(task["seg_idx"]),
                "chunk_start_sec": float(f"{float(task['start_sec']):.3f}"),
                "duration": float(f"{duration:.3f}"),
                "fps": float(f"{fps:.3f}"),
                "frames": n_frames,
                "width": width,
                "height": height,
                "matching_score": float(task["matching_score"]),
                "path": str(dst),
            }) + "\n")
            next_index += 1
            n_appended += 1

    meta_csv_path = new_dir / "metadata.csv"
    fieldnames = [
        "index", "filename", "videoID", "source_video_id", "video_path",
        "caption", "category", "chunk_index", "chunk_start_sec",
        "duration", "fps", "frames", "width", "height",
    ]
    with open(meta_csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"      Wrote {len(metadata_rows)} rows to {meta_csv_path} "
          f"({n_appended} newly appended)")
    return len(metadata_rows)


# ============================================================================
# Validate + finalize (lighter than UCF: keep all valid rows, no truncation)
# ============================================================================

def _run_validate(new_dir: Path, min_frames: int) -> int:
    validate_script = REPO_ROOT / "scripts" / "validate_dataset.py"
    if not validate_script.exists():
        print(f"  (validate_dataset.py not found; skipping validation)")
        return 0
    cmd = [
        sys.executable, str(validate_script),
        "--dataset-dir", str(new_dir),
        "--required-valid", "1",
        "--min-frames", str(min_frames),
        "--write-valid-subset", "valid_subset.csv",
        "--no-require-category",
    ]
    print("[5/5] Validating ...")
    print("  " + " ".join(cmd))
    return int(subprocess.run(cmd).returncode)


def _finalize_to_valid(new_dir: Path) -> int:
    valid_path = new_dir / "valid_subset.csv"
    meta_path = new_dir / "metadata.csv"
    if not valid_path.exists():
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                n = sum(1 for _ in csv.DictReader(f))
            print(f"      No valid_subset.csv; keeping metadata.csv as-is "
                  f"({n} rows)")
            return n
        return -1
    with open(valid_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"      Truncated metadata.csv to {len(rows)} valid rows -> "
          f"{meta_path}")
    return len(rows)


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-pool", required=True,
        help="Existing 480p Panda video pool with metadata.csv + videos/.",
    )
    parser.add_argument(
        "--source-metadata", required=True,
        help="Panda-70M segment-level CSV (e.g. panda70m_training_2m.csv).",
    )
    parser.add_argument(
        "--new-dataset", required=True,
        help="Output dir; will hold videos/ + metadata.csv + manifest.jsonl.",
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="ffmpeg parallelism (one source video can feed multiple "
             "concurrent segment cuts; default %(default)s).",
    )
    parser.add_argument(
        "--target-width", type=int, default=TARGET_WIDTH_DEFAULT,
    )
    parser.add_argument(
        "--target-height", type=int, default=TARGET_HEIGHT_DEFAULT,
    )
    parser.add_argument(
        "--min-segment-seconds", type=float, default=MIN_SEGMENT_SECONDS,
        help="Drop segments shorter than this (default %(default)s).",
    )
    parser.add_argument(
        "--max-segment-seconds", type=float, default=MAX_SEGMENT_SECONDS,
        help="Drop segments longer than this (default %(default)s).",
    )
    parser.add_argument(
        "--min-matching-score", type=float, default=MIN_MATCHING_SCORE_DEFAULT,
        help="Drop segments with caption-vs-video matching_score below "
             "this. The Panda paper recommends >= 0.42; default keeps all "
             "(%(default)s).",
    )
    parser.add_argument(
        "--require-desirable", action="store_true", default=True,
        help="Keep only segments with desirable_filtering == 'desirable'. "
             "Default: enabled.",
    )
    parser.add_argument(
        "--no-require-desirable", action="store_false",
        dest="require_desirable",
    )
    parser.add_argument(
        "--max-segments-per-source", type=int, default=0,
        help="Cap on segments emitted per source video (0 = no cap; "
             "default %(default)s).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Cap on total segments to encode (0 = unlimited; for "
             "smoke-testing).",
    )
    parser.add_argument(
        "--min-frames", type=int, default=48,
        help="Minimum decoded frames in a chunk (validate stage). Should "
             "be >= tta_total_frames; default %(default)s.",
    )
    parser.add_argument(
        "--skip-validate", action="store_true",
        help="Skip validate_dataset.py call (faster, but metadata.csv "
             "may include unreadable chunks).",
    )
    args = parser.parse_args()

    source_pool = Path(args.source_pool).resolve()
    source_metadata = Path(args.source_metadata).resolve()
    new_dir = Path(args.new_dataset).resolve()

    if not source_pool.exists():
        print(f"ERROR: source pool does not exist: {source_pool}",
              file=sys.stderr)
        return 2
    if not source_metadata.exists():
        print(f"ERROR: source metadata does not exist: {source_metadata}",
              file=sys.stderr)
        return 2

    print("=" * 78)
    print("Build Panda-70M PER-SEGMENT retrieval pool")
    print("=" * 78)
    print(f"  source pool        : {source_pool}")
    print(f"  source metadata    : {source_metadata}")
    print(f"  new dataset        : {new_dir}")
    print(f"  workers            : {args.workers}")
    print(f"  target resolution  : {args.target_width}x{args.target_height}")
    print(f"  segment duration   : "
          f"[{args.min_segment_seconds}, {args.max_segment_seconds}] s")
    print(f"  min matching score : {args.min_matching_score}")
    print(f"  require desirable  : {args.require_desirable}")
    print(f"  max segs/source    : "
          f"{args.max_segments_per_source if args.max_segments_per_source > 0 else 'no cap'}")
    print(f"  segment limit      : "
          f"{args.limit if args.limit > 0 else 'no cap'}")
    print("=" * 78)
    print()

    n_rows = _build_pool(
        source_pool=source_pool,
        source_metadata=source_metadata,
        new_dir=new_dir,
        require_desirable=args.require_desirable,
        min_score=args.min_matching_score,
        min_seg_sec=args.min_segment_seconds,
        max_seg_sec=args.max_segment_seconds,
        target_w=args.target_width,
        target_h=args.target_height,
        num_workers=args.workers,
        max_segments_per_source=args.max_segments_per_source,
        limit=args.limit,
    )

    if not args.skip_validate and n_rows > 0:
        _run_validate(new_dir, args.min_frames)
        n_kept = _finalize_to_valid(new_dir)
    else:
        n_kept = n_rows

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"  pool dir       : {new_dir}")
    print(f"  metadata.csv   : {n_kept} rows")
    print(f"  next step:")
    print(f"    sbatch --account=torch_pr_36_mren \\")
    print(f"        --export=ALL,POOL_DIR={new_dir} \\")
    print(f"        delta_experiment/sbatch/precompute_pool_embeddings.sbatch")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
