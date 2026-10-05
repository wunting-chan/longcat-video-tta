#!/usr/bin/env python3
"""Per-video feature extraction for the TTA-gain correlation analysis.

# ============================================================================
# TTA-VISIBLE FRAMES AUDIT  --  panda_1000v_standard  (2026-06-09)
# ============================================================================
# QUESTION ANSWERED HERE: "Which subset of each GT clip does the TTA loop
# actually see for the panda_1000v_standard runs whose per-video PSNRs we
# already have in sweep_experiment/reports/per_video_analysis/2026-06-09/
# per_video_gains.csv ?"
#
# AUDIT SOURCES (frame counts come from environment variables exported in
# the chunked submit script; runners default the TTA window to start at
# `gen_start_frame - tta_total_frames` and clamp to avoid GT-region leakage):
#
#   * sweep_experiment/sbatch/submit_standard_1000v_chunked.sh
#       NUM_FRAMES=28
#       NUM_COND_FRAMES=14
#       GEN_START_FRAME=48
#       TTA_TOTAL_FRAMES=48
#       TTA_CONTEXT_FRAMES=14
#       NUM_INFERENCE_STEPS=50, GUIDANCE_SCALE=4.0, RESOLUTION=480p
#   * delta_experiment/scripts/run_delta_a.py  (ADA / ADA_NOPROMPT)
#       lines 613-628  : defaults / safety clamps
#       lines 852-869  : tta_start = gen_start_frame - tta_total_frames;
#                        load_video_frames(...,  tta_total_frames, ...,
#                                          start_frame=max(0, tta_start))
#                        num_ctx_lat = 1 + (tta_context_frames - 1) // 4
#                        split_tta_latents(all_latents, num_ctx_lat, ...)
#   * lora_experiment/scripts/run_lora_tta.py  (LORA_R8_TTA / LORA_R8_TTA_NOPROMPT)
#       lines 817-831  : same defaults / safety clamps
#       lines 1137-1145, 1179-1188 : same tta_start / num_ctx_lat formula
#   * delta_experiment/scripts/run_tinylora.py  (TL_BARE_R2 / TL_TIED_R2)
#       lines 265-276  : same defaults / safety clamps
#       lines 471-484  : same tta_start / num_ctx_lat formula
#   * delta_experiment/scripts/common.py
#       load_video_frames():  decodes `num_frames` frames starting at byte-
#                             precise `start_frame` via PyAV, returns
#                             tensor [1, 3, T, H, W] in [-1, 1] after
#                             trilinear-resize to (T, 480, 832).
#
# RESULT: ALL THREE runners agree.  For panda_1000v_standard each video's
# TTA loop loads pixel frames [0:48] (since tta_total_frames == gen_start_frame
# == 48) and uses every one of those 48 frames:
#       - first 14 frames (tta_context_frames=14) are clean VAE-encoded
#         context  (-> 4 context latents with VAE temporal scale 4);
#       - the remaining 34 frames are the noised-and-denoised TTA training
#         target (further split into train/val per early-stopping holdout).
#   Generation then produces frames [48:62]  (num_frames=28 of which 14
#   are conditioning so 14 are genuinely new).  Frames [48:62] are GT
#   for the PSNR/SSIM/LPIPS we already have; they are NOT visible to
#   TTA (so any feature computed on them is "Tier 3 diagnostic, not
#   online-actionable" per the user's constraint).
#
# Concretely the model already had _every pre-anchor frame_ in pixel space
# for this config.  TTA sees the full pre-anchor clip; the visible window
# happens to be 100 % of the pre-anchor span.  This is the most generous
# read for the model and matches what is available at deploy time in this
# evaluation setup.
#
# Frame indices used by THIS SCRIPT (auto mode for panda_1000v_standard):
#       TTA-visible:        [0, 48)   ->  48 frames per video
#       Generation-target:  [48, 62)  ->  14 frames per video (Tier 3 only)
#
# AUTO-DETECTION CONSTANTS exposed below (named to match
# `submit_standard_1000v_chunked.sh` env vars):
#       TTA_TOTAL_FRAMES    = 48   (number of pre-anchor pixel frames TTA loads)
#       GEN_START_FRAME     = 48   (first frame the diffusion sampler emits)
#       NUM_FRAMES          = 28   (diffusion window length)
#       NUM_COND_FRAMES     = 14   (conditioning prefix; remainder are new)
# Derived:
#       AUTO_TTA_VISIBLE_RANGE = (max(0, GEN_START_FRAME - TTA_TOTAL_FRAMES),
#                                 GEN_START_FRAME)
#                              = (0, 48)
#       AUTO_GEN_TARGET_RANGE  = (GEN_START_FRAME,
#                                 GEN_START_FRAME + (NUM_FRAMES - NUM_COND_FRAMES))
#                              = (48, 62)
# Override either window from the CLI with `--tta-visible-frames A:B` or
# `--gen-target-frames A:B`; pass `--gen-target-frames none` to disable
# Tier-3 columns entirely.
# ============================================================================

Tier-1 features computed on TTA-visible frames + caption (the model's actual
TTA-time information, per the user's "we only care about the subsection of
data the model has access to during TTA" constraint).

Tier-3 features compare TTA-visible frames against the generation-target
frames (which are GT, not online-actionable).  They are still emitted as
diagnostics but the corresponding columns are documented in the schema
comment block below as not-online-actionable.

CSV schema (one row per ``video_id``):

    video_id, n_frames_used, tta_visible_range, gen_target_range, caption,

    # Tier 1 -- visible-only, online-actionable
    cut_count_pyscenedetect, cut_count_histogram, cut_density_per_frame,
    clip_text_image_sim_mean, clip_text_image_sim_var,
    clip_text_image_sim_min,
    dino_temporal_l2_mean,
    laplacian_variance_mean,
    rgb_histogram_entropy_mean,

    # Tier 3 -- DIAGNOSTIC ONLY, uses GT generation-target frames
    dino_tta_vs_genregion_sim,
    clip_text_genregion_sim_mean,

    # Provenance
    clip_model, dino_model, hist_bins_per_channel, hist_bhattacharyya_thresh

Dependencies (the user's cluster env already has all of these for the
existing CLIP-gate / dynamicness pipeline; see env_setup/01_setup_longcat_env.sbatch):

    torch, transformers (CLIPModel/CLIPProcessor + AutoImageProcessor/AutoModel
        for DINOv2), av (PyAV), opencv-python, numpy, scenedetect (PySceneDetect).

PySceneDetect is treated as OPTIONAL: if the import fails the column
``cut_count_pyscenedetect`` (and ``cut_density_per_frame`` which derives
from it) become NaN and a one-line WARNING is printed.  The histogram-based
``cut_count_histogram`` is always populated as the backup.

CLI mirrors ``scripts/analyze_per_video_tta_gain.py`` conventions:

    python3 scripts/extract_video_features_for_tta.py \\
        --videos-dir datasets/panda_1000_480p \\
        --captions-csv datasets/panda_1000_480p/metadata.csv \\
        --tta-visible-frames auto \\
        --output sweep_experiment/reports/per_video_analysis/2026-06-09/video_features.csv \\
        --device cuda \\
        --batch-size 16
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.caption_utils import (
    canonical_video_id as _canonical_video_id,
    load_resolved_captions_csv,
    resolve_caption_for_clip,
)

# Lazy / guarded imports so the script can still print --help on machines
# that do not have torch installed.  All heavy imports happen inside main()
# or the per-feature helpers.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# panda_1000v_standard frame geometry — sourced verbatim from
# `sweep_experiment/sbatch/submit_standard_1000v_chunked.sh` env vars and
# matched by all four runners (delta_a / lora_tta / tinylora; see the audit
# block at the top of this file for line numbers).  Editing one of these
# four constants automatically reflows AUTO_TTA_VISIBLE_RANGE /
# AUTO_GEN_TARGET_RANGE so the script stays consistent with the runners.
TTA_TOTAL_FRAMES: int = 48      # pre-anchor pixel frames the TTA loop loads
GEN_START_FRAME: int = 48       # first frame the diffusion sampler emits
NUM_FRAMES: int = 28            # diffusion window length
NUM_COND_FRAMES: int = 14       # conditioning prefix; (NUM_FRAMES - NUM_COND_FRAMES) are new

# Derived: TTA-visible window (the runners' `start_frame=max(0, gen_start_frame
# - tta_total_frames)` slice of length tta_total_frames).
AUTO_TTA_VISIBLE_RANGE: Tuple[int, int] = (
    max(0, GEN_START_FRAME - TTA_TOTAL_FRAMES),
    GEN_START_FRAME,
)
# Derived: generation-target window — frames the diffusion sampler emits as
# genuinely new content (the conditioning prefix is dropped because those
# frames overlap TTA-visible).  Disjoint from AUTO_TTA_VISIBLE_RANGE by
# construction whenever GEN_START_FRAME >= TTA_TOTAL_FRAMES.
AUTO_GEN_TARGET_RANGE: Tuple[int, int] = (
    GEN_START_FRAME,
    GEN_START_FRAME + (NUM_FRAMES - NUM_COND_FRAMES),
)

# Histogram cut detector: Bhattacharyya distance threshold.  Calibrated so
# that obvious hard cuts in panda_100 sample clips fire (>= 0.4 is the
# OpenCV-tutorial "strong dissimilarity" band for HISTCMP_BHATTACHARYYA),
# and most slow camera pans / talking-head segments stay quiet.  Pinned
# here so the column is reproducible across runs.
HIST_BHATTACHARYYA_THRESH: float = 0.40
HIST_BINS_PER_CHANNEL: int = 8

# Default model checkpoints (small + fast, match what the existing CLIP
# gate code uses; can be overridden on the CLI).
DEFAULT_CLIP_MODEL: str = "openai/clip-vit-base-patch32"
DEFAULT_DINO_MODEL: str = "facebook/dinov2-small"

PROGRESS_EVERY: int = 50

# Tier 3 columns are emitted but flagged here so downstream consumers can
# split them out programmatically.
TIER3_COLUMNS: Tuple[str, ...] = (
    "dino_tta_vs_genregion_sim",
    "clip_text_genregion_sim_mean",
)


# ---------------------------------------------------------------------------
# Canonical video-id extraction (mirrors analyze_per_video_tta_gain.py)
# ---------------------------------------------------------------------------
# Provided by scripts.caption_utils.canonical_video_id (imported above).


# ---------------------------------------------------------------------------
# Video frame decoding (PyAV; matches delta_experiment/scripts/common.py)
# ---------------------------------------------------------------------------
def _resize_frame_stack_uint8(
    frames_rgb: np.ndarray,
    resize_hw: Tuple[int, int],
) -> np.ndarray:
    """Resize a ``(T, H, W, 3)`` uint8 stack via torch bilinear interpolate.

    Round-4: replaces ``cv2.resize`` in ``decode_window``.  The OOD runner
    already resizes with ``torch.nn.functional.interpolate`` (see
    ``delta_experiment/scripts/common.py::load_video_frames``); keeping
    extract on the same path avoids OpenCV-4.9.0's numpy-binding failures
    on the cluster.
    """
    import torch
    import torch.nn.functional as F

    h_t, w_t = resize_hw
    if frames_rgb.shape[1] == h_t and frames_rgb.shape[2] == w_t:
        return frames_rgb
    t = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float()  # T,3,H,W
    t = F.interpolate(t, size=(h_t, w_t), mode="bilinear", align_corners=False)
    out = t.permute(0, 2, 3, 1).clamp(0.0, 255.0).numpy().astype(np.uint8)
    return np.ascontiguousarray(out)


def decode_window(
    video_path: str,
    start_frame: int,
    num_frames: int,
    resize_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Decode a contiguous frame window as uint8 HxWx3 (RGB).

    Returns array shaped (T, H, W, 3) with len(T) == num_frames (padded by
    repeating the last frame if the source clip is too short).  Optional
    ``resize_hw=(h, w)`` resizes the decoded stack with torch bilinear
    interpolate (no OpenCV).
    """
    import av

    try:
        container = av.open(video_path)
    except Exception as exc:  # noqa: BLE001  -- av.AVError, OSError, etc.
        raise ValueError(f"Cannot open {video_path}: {exc}") from exc

    frames: List[np.ndarray] = []
    decoded = 0
    try:
        for frame in container.decode(video=0):
            if decoded < start_frame:
                decoded += 1
                continue
            if len(frames) >= num_frames:
                break
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)
            decoded += 1
    except Exception as exc:  # noqa: BLE001  -- InvalidDataError (moov atom), etc.
        raise ValueError(f"Decode failed for {video_path}: {exc}") from exc
    finally:
        container.close()

    if not frames:
        raise ValueError(f"No frames decoded from {video_path} at start={start_frame}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    stacked = np.stack(frames[:num_frames], axis=0)
    if resize_hw is not None:
        stacked = _resize_frame_stack_uint8(stacked, resize_hw)
    return stacked


# ---------------------------------------------------------------------------
# Cheap CPU features (cuts + texture + colour)
# ---------------------------------------------------------------------------
def _to_uint8_hwc_stack_for_cv2(arr) -> np.ndarray:
    """Coerce a frame stack to a contiguous uint8 ``(T, H, W, 3)`` numpy array.

    Round-3 note (commit landing this docstring update): the original
    cv2-binding-compat motivation for this helper is now obsolete --
    ``count_cuts_histogram`` and ``rgb_histogram_entropy_mean`` no longer
    call ``cv2.calcHist`` / ``cv2.normalize`` / ``cv2.compareHist`` at all
    (numpy ``histogramdd`` + manual L2/L1 normalisation + analytic
    Bhattacharyya replace them; see ``_joint_hist_3ch_np`` for the
    motivation -- OpenCV-4.9.0's Python binding rejected even
    correctly-formed numpy views in job 10739993).  The helper is kept
    for its *shape-coercion* role: the numpy histogram code path still
    needs a well-formed ``(T, H, W, 3)`` uint8 stack, and several other
    feature functions further down the file (``laplacian_variance_mean``,
    the CLIP/DINO encoders) still benefit from a single defensive
    coercion at the entry point.

    Round-1 fix (commit ``0802d5a``) coerced *dtype* and *contiguity* but
    not *shape* or *container*: an upstream producer that hands this
    script a torch tensor, a Python list of per-frame arrays, or a
    channels-first ``(T, C, H, W)`` stack still trips downstream
    consumers expecting ``(T, H, W, 3)`` uint8.

    Coercion pipeline (all steps no-ops on the happy ``np.uint8`` HWC
    path that PyAV's ``frame.to_ndarray(format="rgb24")`` produces, so
    the post-patch numerical output on well-formed input is bit-for-bit
    identical):

      * ``torch.Tensor`` (any device / dtype) -> ``detach().cpu().numpy()``
      * ``list`` / ``tuple`` of per-frame arrays -> ``np.stack`` on axis 0
      * ``(T, 3, H, W)`` channels-first ndarray -> ``np.transpose`` to HWC
      * float in ``[0, 1]`` or ``[0, 255]`` -> rescale + clip + cast uint8
      * non-contiguous view -> ``np.ascontiguousarray``

    The trailing asserts pin the output contract so any *5th* malformed
    input shape surfaces with a useful message in the .err log.
    """
    if hasattr(arr, "detach") and hasattr(arr, "cpu") and hasattr(arr, "numpy"):
        arr = arr.detach().cpu().numpy()
    if isinstance(arr, (list, tuple)):
        arr = np.stack([np.asarray(x) for x in arr], axis=0)
    arr = np.asarray(arr)
    # ``(T, 3, H, W)`` -> ``(T, H, W, 3)``. We detect channels-first by the
    # axis-size signature: C is 3, H/W are spatial and (for the 480p frames
    # this script processes) always > 3.  The ``shape[-1] != 3`` guard means
    # a well-formed HWC stack with H == 3 (impossible in practice) is NOT
    # mis-transposed.
    if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            finite_max = float(np.nanmax(arr)) if arr.size else 0.0
            if finite_max <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8, copy=False)
    arr = np.ascontiguousarray(arr)
    assert arr.ndim == 4, (
        f"Expected 4D stack (T, H, W, C), got shape {arr.shape}"
    )
    assert arr.shape[-1] == 3, (
        f"Expected 3-channel HWC, got channels-last={arr.shape[-1]} "
        f"(full shape {arr.shape})"
    )
    return arr


def _joint_hist_3ch_np(frame_hwc: np.ndarray, bins_per_channel: int) -> np.ndarray:
    """3D joint RGB histogram, numpy-only equivalent of
    ``cv2.calcHist([f], [0,1,2], None, [b]*3, [0,256]*3)``.

    Round-3 bug-fix: bypasses cv2.calcHist (which silently rejects
    correctly-formed numpy views in OpenCV-4.9.0's Python binding --
    job 10739993 reproduced this with the round-2 helper running
    successfully but cv2 still raising 'wrong type' on the per-frame
    slice).

    Returns a float32 array of shape (bins, bins, bins) that is
    bit-equivalent to cv2.calcHist's output for uint8 HWC input.
    """
    assert frame_hwc.ndim == 3 and frame_hwc.shape[-1] == 3, (
        f"Expected (H, W, 3) uint8 frame, got shape {frame_hwc.shape}"
    )
    pixels = frame_hwc.reshape(-1, 3)
    hist, _ = np.histogramdd(
        pixels,
        bins=[bins_per_channel] * 3,
        range=[[0, 256], [0, 256], [0, 256]],
    )
    return hist.astype(np.float32)


def _bhattacharyya_np(h1: np.ndarray, h2: np.ndarray) -> float:
    """Bhattacharyya distance, numpy equivalent of
    ``cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)``.

    Both histograms must be already-normalised (cv2.normalize default
    is L2; we mirror that).  Formula matches OpenCV's
    HISTCMP_BHATTACHARYYA (which is actually the Hellinger distance
    variant -- see OpenCV source).
    """
    h1 = h1.astype(np.float64)
    h2 = h2.astype(np.float64)
    s1 = float(h1.sum())
    s2 = float(h2.sum())
    if s1 <= 0.0 or s2 <= 0.0:
        return 1.0
    p1 = h1 / s1
    p2 = h2 / s2
    bc = float(np.sqrt(p1 * p2).sum())
    return float(np.sqrt(max(0.0, 1.0 - bc)))


def _l2_normalize_np(hist: np.ndarray) -> np.ndarray:
    """L2-normalise a histogram, mirroring cv2.normalize default."""
    hist = hist.astype(np.float32)
    norm = float(np.linalg.norm(hist))
    if norm > 0.0:
        hist = hist / norm
    return hist


def count_cuts_histogram(frames_rgb: np.ndarray,
                         bins_per_channel: int = HIST_BINS_PER_CHANNEL,
                         thresh: float = HIST_BHATTACHARYYA_THRESH) -> int:
    """Count consecutive frame pairs whose RGB joint-histogram Bhattacharyya
    distance exceeds ``thresh``.  Cheap, fully deterministic backup for
    PySceneDetect.

    Round-3: cv2.calcHist replaced with numpy.histogramdd to bypass an
    OpenCV-4.9.0 binding incompatibility -- see ``_joint_hist_3ch_np``.
    """
    T = frames_rgb.shape[0] if hasattr(frames_rgb, "shape") else len(frames_rgb)
    if T < 2:
        return 0
    frames_rgb = _to_uint8_hwc_stack_for_cv2(frames_rgb)
    prev_hist = None
    cuts = 0
    for t in range(T):
        hist = _joint_hist_3ch_np(frames_rgb[t], bins_per_channel)
        hist = _l2_normalize_np(hist)
        if prev_hist is not None:
            dist = _bhattacharyya_np(prev_hist, hist)
            if dist >= thresh:
                cuts += 1
        prev_hist = hist
    return cuts


def count_cuts_pyscenedetect(video_path: str, start_frame: int,
                             end_frame: int) -> Optional[int]:
    """Use PySceneDetect ContentDetector on the [start_frame, end_frame)
    range.  Returns ``None`` if PySceneDetect is not installed; caller is
    responsible for emitting NaN into the CSV in that case.
    """
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] PySceneDetect unavailable ({exc}); "
              "cut_count_pyscenedetect will be NaN", file=sys.stderr)
        return None

    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        scene_manager.detect_scenes(
            video, frame_skip=0, show_progress=False,
            duration=None, start_time=None, end_time=None,
        )
        # PySceneDetect returns scene cuts as start times of each scene
        # (in frames). The number of cuts is len(scenes) - 1 if scenes is
        # not empty.  We restrict to those falling inside [start_frame,
        # end_frame).
        scenes = scene_manager.get_scene_list()
        cut_count = 0
        for s_start, _s_end in scenes[1:]:  # skip the artificial scene-0 start
            f = s_start.get_frames()
            if start_frame <= f < end_frame:
                cut_count += 1
        return int(cut_count)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] PySceneDetect failed on {video_path}: {exc}",
              file=sys.stderr)
        return None


def laplacian_variance_mean(frames_rgb: np.ndarray) -> float:
    """Mean over frames of Var(Laplacian(gray)).  Sharpness proxy.

    Round-4: numpy-only discrete Laplacian replaces ``cv2.cvtColor`` /
    ``cv2.Laplacian`` — the remaining cv2 calls after round-3's histogram
    bypass were the actual cluster failure surface (job 10756670: 10 min,
    ``errored=1000``, header-only CSV).
    """
    frames_rgb = _to_uint8_hwc_stack_for_cv2(frames_rgb)
    vals: List[float] = []
    for t in range(frames_rgb.shape[0]):
        gray = (
            0.299 * frames_rgb[t, :, :, 0].astype(np.float64)
            + 0.587 * frames_rgb[t, :, :, 1].astype(np.float64)
            + 0.114 * frames_rgb[t, :, :, 2].astype(np.float64)
        )
        g_pad = np.pad(gray, 1, mode="edge")
        lap = (
            g_pad[:-2, 1:-1] + g_pad[2:, 1:-1]
            + g_pad[1:-1, :-2] + g_pad[1:-1, 2:]
            - 4.0 * g_pad[1:-1, 1:-1]
        )
        vals.append(float(lap.var()))
    return float(np.mean(vals)) if vals else float("nan")


def rgb_histogram_entropy_mean(frames_rgb: np.ndarray,
                               bins_per_channel: int = HIST_BINS_PER_CHANNEL,
                               ) -> float:
    """Mean over frames of Shannon entropy (bits) of the joint RGB histogram.

    A featureless monochrome frame entropy -> ~0; a richly-coloured
    natural frame approaches log2(bins_per_channel ** 3).

    Round-3: cv2.calcHist replaced with numpy.histogramdd to bypass an
    OpenCV-4.9.0 binding incompatibility -- see ``_joint_hist_3ch_np``.
    The L1 normalisation that cv2.normalize would have applied is done
    manually via ``hist / hist.sum()``.
    """
    vals: List[float] = []
    frames_rgb = _to_uint8_hwc_stack_for_cv2(frames_rgb)
    for t in range(frames_rgb.shape[0]):
        hist = _joint_hist_3ch_np(frames_rgb[t], bins_per_channel).flatten()
        total = float(hist.sum())
        if total <= 0:
            continue
        p = hist / total
        p = p[p > 0]
        vals.append(float(-(p * np.log2(p)).sum()))
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# CLIP image / text encoder (HuggingFace transformers; matches CLIP-gate path)
# ---------------------------------------------------------------------------
class _CLIPScorer:
    def __init__(self, model_name: str, device: str):
        from transformers import CLIPModel, CLIPProcessor
        import torch

        self.device = device
        self.model_name = model_name
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.torch = torch

    def encode_images(self, frames_rgb: np.ndarray, batch_size: int = 16):
        """frames_rgb: (T, H, W, 3) uint8 -> (T, D) normalised image embeds."""
        torch = self.torch
        T = frames_rgb.shape[0]
        out = []
        with torch.inference_mode():
            for s in range(0, T, batch_size):
                chunk = [frames_rgb[i] for i in range(s, min(s + batch_size, T))]
                # CLIPProcessor accepts a list of HWC uint8 numpy / PIL.
                inputs = self.processor(images=chunk, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                feats = self.model.get_image_features(pixel_values=pixel_values)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)

    def encode_text(self, text: str):
        torch = self.torch
        with torch.inference_mode():
            inputs = self.processor(text=[text], return_tensors="pt",
                                    truncation=True, padding=True, max_length=77)
            input_ids = inputs["input_ids"].to(self.device)
            attn = inputs.get("attention_mask")
            if attn is not None:
                attn = attn.to(self.device)
            feat = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attn,
            )
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feat.float().cpu().numpy()[0]


# ---------------------------------------------------------------------------
# DINOv2 image encoder
# ---------------------------------------------------------------------------
class _DINOScorer:
    def __init__(self, model_name: str, device: str):
        from transformers import AutoImageProcessor, AutoModel
        import torch

        self.device = device
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.torch = torch

    def encode_images(self, frames_rgb: np.ndarray, batch_size: int = 16):
        """frames_rgb: (T, H, W, 3) uint8 -> (T, D) L2-normalised CLS embeddings."""
        torch = self.torch
        T = frames_rgb.shape[0]
        out = []
        with torch.inference_mode():
            for s in range(0, T, batch_size):
                chunk = [frames_rgb[i] for i in range(s, min(s + batch_size, T))]
                inputs = self.processor(images=chunk, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                feats = self.model(pixel_values=pixel_values).last_hidden_state[:, 0]
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)


# ---------------------------------------------------------------------------
# Caption CSV loader — segment-aligned single string per video
# ---------------------------------------------------------------------------
def load_captions_csv(path: Path) -> Dict[str, str]:
    """Return {canonical_video_id -> resolved caption string}."""
    return load_resolved_captions_csv(path, canonical_id=_canonical_video_id)


# ---------------------------------------------------------------------------
# Video file enumeration (mirrors load_ucf101_video_list logic but file-only)
# ---------------------------------------------------------------------------
def list_video_paths(videos_dir: Path) -> List[Path]:
    """Return sorted (by canonical id) list of video files under
    ``videos_dir``.  Honours the ``videos/`` subdir layout the Panda
    datasets use; falls back to a recursive .mp4/.avi scan otherwise."""
    candidates: List[Path] = []
    subdir = videos_dir / "videos"
    if subdir.is_dir():
        for ext in ("*.mp4", "*.avi"):
            candidates.extend(subdir.glob(ext))
    if not candidates:
        for ext in ("*.mp4", "*.avi"):
            candidates.extend(videos_dir.rglob(ext))
    return sorted(candidates, key=lambda p: _canonical_video_id(p.name))


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------
def _parse_frame_range_arg(arg: str, default: Tuple[int, int]) -> Tuple[int, int]:
    """Parse a 'start:end' frame-range CLI value.

    Returns ``default`` when ``arg`` is empty / ``'auto'``; otherwise parses
    ``'A:B'`` -> ``(int(A), int(B))``.  Used for both ``--tta-visible-frames``
    and ``--gen-target-frames`` so each flag resolves to its OWN auto value
    (previously a single helper aliased to AUTO_TTA_VISIBLE_RANGE was used
    for both, which made ``--gen-target-frames auto`` collapse onto the
    visible range and silently turn the Tier-3 columns into self-similarity).
    """
    if not arg or arg.lower() == "auto":
        return default
    if ":" in arg:
        a, b = arg.split(":", 1)
        return int(a), int(b)
    raise argparse.ArgumentTypeError(
        f"frame-range arg must be 'auto' or 'A:B', got {arg!r}"
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--videos-dir", type=Path, required=True,
                    help="Dataset root containing videos/ subdir or *.mp4 directly.")
    ap.add_argument("--captions-csv", type=Path, required=True,
                    help="Panda metadata.csv (or UCF-style: filename,text).")
    ap.add_argument("--tta-visible-frames", type=str, default="auto",
                    help="'auto' (resolved from panda_1000v_standard audit: "
                         f"{AUTO_TTA_VISIBLE_RANGE[0]}:{AUTO_TTA_VISIBLE_RANGE[1]}, "
                         "i.e. [max(0, GEN_START_FRAME - TTA_TOTAL_FRAMES) : "
                         "GEN_START_FRAME]) or an explicit 'A:B' python-slice range.")
    ap.add_argument("--gen-target-frames", type=str, default="auto",
                    help="'auto' (resolves to "
                         f"{AUTO_GEN_TARGET_RANGE[0]}:{AUTO_GEN_TARGET_RANGE[1]}, "
                         "i.e. [GEN_START_FRAME : GEN_START_FRAME + "
                         "(NUM_FRAMES - NUM_COND_FRAMES)]) or an explicit 'A:B' "
                         "python-slice range.  Used ONLY by Tier-3 diagnostic "
                         "columns.  Pass 'none' to leave those columns NaN.")
    ap.add_argument("--output", type=Path, required=True,
                    help="CSV path. Idempotent — existing rows are reused "
                         "unless --force is passed.")
    ap.add_argument("--device", type=str, default="cuda",
                    help="Torch device. 'cpu' works but is much slower.")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Frames per CLIP/DINO forward pass within a video.")
    ap.add_argument("--max-videos", type=int, default=0,
                    help="0 = all. Otherwise process the first N (by canonical id).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if the row already exists in --output.")
    ap.add_argument("--clip-model", type=str, default=DEFAULT_CLIP_MODEL,
                    help=f"HuggingFace CLIP id. Default: {DEFAULT_CLIP_MODEL}.")
    ap.add_argument("--dino-model", type=str, default=DEFAULT_DINO_MODEL,
                    help=f"HuggingFace DINOv2 id. Default: {DEFAULT_DINO_MODEL}.")
    ap.add_argument("--resize-h", type=int, default=480,
                    help="Decode/resize frames to this height before featurization.")
    ap.add_argument("--resize-w", type=int, default=832,
                    help="Decode/resize frames to this width before featurization.")
    ap.add_argument("--skip-pyscenedetect", action="store_true",
                    help="Force-skip PySceneDetect even if installed "
                         "(use for environments where it's known flaky).")
    return ap.parse_args()


def _load_existing_csv(path: Path) -> Tuple[List[dict], List[str]]:
    """Return (rows, fieldnames) for an existing output CSV (idempotency)."""
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
    return rows, fieldnames


def _format_row(row: dict, fieldnames: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in fieldnames:
        v = row.get(k)
        if v is None:
            out[k] = ""
        elif isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                out[k] = ""
            else:
                out[k] = f"{v:.6f}"
        else:
            out[k] = str(v)
    return out


# ---------------------------------------------------------------------------
# Per-video extraction (one entry point so the body is easy to read)
# ---------------------------------------------------------------------------
def extract_one_video(
    video_path: Path,
    caption_raw: str,
    visible_range: Tuple[int, int],
    gen_range: Optional[Tuple[int, int]],
    *,
    clip: _CLIPScorer,
    dino: _DINOScorer,
    resize_hw: Tuple[int, int],
    batch_size: int,
    skip_pyscenedetect: bool,
    diagnostic: bool = False,
) -> dict:
    vs, ve = visible_range
    n_visible = ve - vs
    caption_text = resolve_caption_for_clip(caption_raw)

    # ---- decode TTA-visible frames once ------------------------------------
    visible_frames = decode_window(
        str(video_path), start_frame=vs, num_frames=n_visible,
        resize_hw=resize_hw,
    )
    if diagnostic:
        print(
            f"[diag] {video_path.name}: decode "
            f"shape={visible_frames.shape} dtype={visible_frames.dtype} "
            f"contig={visible_frames.flags['C_CONTIGUOUS']}",
            flush=True,
        )

    # ---- cheap CPU features (isolated so one bad feature -> NaN, not skip) --
    cut_hist: Optional[int] = None
    cut_psd: Optional[int] = None
    cut_density = float("nan")
    lap_var_mean = float("nan")
    hist_ent_mean = float("nan")

    try:
        cut_hist = count_cuts_histogram(visible_frames)
        if diagnostic:
            print(f"[diag] {video_path.name}: cut_hist={cut_hist}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {video_path.name}: cut_count_histogram failed ({exc})",
              file=sys.stderr)

    if not skip_pyscenedetect:
        cut_psd = count_cuts_pyscenedetect(str(video_path), start_frame=vs, end_frame=ve)
    cut_density = (
        (cut_psd / n_visible) if (cut_psd is not None and n_visible > 0)
        else float("nan")
    )

    try:
        lap_var_mean = laplacian_variance_mean(visible_frames)
        if diagnostic:
            print(f"[diag] {video_path.name}: lap_var={lap_var_mean:.6f}",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {video_path.name}: laplacian_variance_mean failed ({exc})",
              file=sys.stderr)

    try:
        hist_ent_mean = rgb_histogram_entropy_mean(visible_frames)
        if diagnostic:
            print(f"[diag] {video_path.name}: hist_ent={hist_ent_mean:.6f}",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {video_path.name}: rgb_histogram_entropy_mean failed ({exc})",
              file=sys.stderr)

    # ---- CLIP image embeddings (visible region) ---------------------------
    ct_mean = ct_var = ct_min = float("nan")
    clip_text_vec = None
    try:
        clip_img_visible = clip.encode_images(visible_frames, batch_size=batch_size)
        if diagnostic:
            print(
                f"[diag] {video_path.name}: clip_img "
                f"shape={clip_img_visible.shape}",
                flush=True,
            )
        clip_text_vec = (
            clip.encode_text(caption_text)
            if caption_text else None
        )
        if clip_text_vec is not None and clip_img_visible.shape[0] > 0:
            sims_visible = clip_img_visible @ clip_text_vec  # cosine since normalised
            ct_mean = float(sims_visible.mean())
            ct_var = float(sims_visible.var(ddof=0))
            ct_min = float(sims_visible.min())
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {video_path.name}: CLIP features failed ({exc})",
              file=sys.stderr)

    # ---- DINO temporal coherence on visible region ------------------------
    dino_temp_l2 = float("nan")
    dino_visible = np.zeros((0, 1), dtype=np.float32)
    try:
        dino_visible = dino.encode_images(visible_frames, batch_size=batch_size)
        if diagnostic:
            print(
                f"[diag] {video_path.name}: dino_visible "
                f"shape={dino_visible.shape}",
                flush=True,
            )
        if dino_visible.shape[0] >= 2:
            diffs = dino_visible[1:] - dino_visible[:-1]
            l2 = np.linalg.norm(diffs, axis=1)
            dino_temp_l2 = float(l2.mean())
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {video_path.name}: DINO features failed ({exc})",
              file=sys.stderr)

    # ---- Tier 3: generation-target region (diagnostic only) ---------------
    dino_tta_vs_gen = float("nan")
    clip_text_gen_mean = float("nan")
    if gen_range is not None:
        gs, ge = gen_range
        n_gen = ge - gs
        if n_gen > 0:
            try:
                gen_frames = decode_window(
                    str(video_path), start_frame=gs, num_frames=n_gen,
                    resize_hw=resize_hw,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {video_path.name}: gen-region decode failed "
                      f"({exc}); Tier-3 columns NaN", file=sys.stderr)
                gen_frames = None
            if gen_frames is not None:
                dino_gen = dino.encode_images(gen_frames, batch_size=batch_size)
                if dino_gen.shape[0] > 0 and dino_visible.shape[0] > 0:
                    mu_tta = dino_visible.mean(axis=0)
                    mu_tta = mu_tta / np.linalg.norm(mu_tta).clip(min=1e-12)
                    mu_gen = dino_gen.mean(axis=0)
                    mu_gen = mu_gen / np.linalg.norm(mu_gen).clip(min=1e-12)
                    dino_tta_vs_gen = float(mu_tta @ mu_gen)
                if clip_text_vec is not None:
                    clip_img_gen = clip.encode_images(gen_frames, batch_size=batch_size)
                    if clip_img_gen.shape[0] > 0:
                        sims_gen = clip_img_gen @ clip_text_vec
                        clip_text_gen_mean = float(sims_gen.mean())

    vid_id = _canonical_video_id(video_path.name)
    return {
        "video_id": vid_id,
        "n_frames_used": int(n_visible),
        "tta_visible_range": f"{vs}:{ve}",
        "gen_target_range": (
            f"{gen_range[0]}:{gen_range[1]}" if gen_range is not None else ""
        ),
        "caption_text": caption_text,
        # Tier 1
        "cut_count_pyscenedetect": (
            float("nan") if cut_psd is None else int(cut_psd)
        ),
        "cut_count_histogram": (
            float("nan") if cut_hist is None else int(cut_hist)
        ),
        "cut_density_per_frame": float(cut_density),
        "clip_text_image_sim_mean": float(ct_mean),
        "clip_text_image_sim_var": float(ct_var),
        "clip_text_image_sim_min": float(ct_min),
        "dino_temporal_l2_mean": float(dino_temp_l2),
        "laplacian_variance_mean": float(lap_var_mean),
        "rgb_histogram_entropy_mean": float(hist_ent_mean),
        # Tier 3 (DIAGNOSTIC ONLY)
        "dino_tta_vs_genregion_sim": float(dino_tta_vs_gen),
        "clip_text_genregion_sim_mean": float(clip_text_gen_mean),
        # Provenance
        "clip_model": clip.model_name,
        "dino_model": dino.model_name,
        "hist_bins_per_channel": int(HIST_BINS_PER_CHANNEL),
        "hist_bhattacharyya_thresh": float(HIST_BHATTACHARYYA_THRESH),
    }


def _fieldnames() -> List[str]:
    return [
        "video_id", "n_frames_used", "tta_visible_range", "gen_target_range",
        "caption_text",
        "cut_count_pyscenedetect", "cut_count_histogram",
        "cut_density_per_frame",
        "clip_text_image_sim_mean", "clip_text_image_sim_var",
        "clip_text_image_sim_min",
        "dino_temporal_l2_mean",
        "laplacian_variance_mean",
        "rgb_histogram_entropy_mean",
        "dino_tta_vs_genregion_sim",
        "clip_text_genregion_sim_mean",
        "clip_model", "dino_model",
        "hist_bins_per_channel", "hist_bhattacharyya_thresh",
    ]


def main() -> int:
    args = _parse_args()
    visible_range = _parse_frame_range_arg(
        args.tta_visible_frames, default=AUTO_TTA_VISIBLE_RANGE,
    )
    if args.gen_target_frames.lower() == "none":
        gen_range: Optional[Tuple[int, int]] = None
    else:
        gen_range = _parse_frame_range_arg(
            args.gen_target_frames, default=AUTO_GEN_TARGET_RANGE,
        )
        # Sanity: don't let gen overlap visible (would mean Tier 3 == Tier 1
        # which would silently inflate diagnostic correlations).
        if gen_range[0] < visible_range[1]:
            print(f"[warn] gen-target range {gen_range} starts inside the "
                  f"visible range {visible_range}; Tier-3 columns will be "
                  "computed but flagged as overlapping.",
                  file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Per-video feature extraction (TTA-visible scope)")
    print("=" * 70)
    print(f"Videos dir         : {args.videos_dir}")
    print(f"Captions CSV       : {args.captions_csv}")
    print(f"TTA-visible frames : {visible_range[0]}:{visible_range[1]}  "
          f"(n={visible_range[1] - visible_range[0]})")
    print(f"Gen-target frames  : "
          f"{'none' if gen_range is None else f'{gen_range[0]}:{gen_range[1]}'}")
    print(f"Output             : {args.output}")
    print(f"Device             : {args.device}")
    print(f"Batch size         : {args.batch_size}")
    print(f"CLIP model         : {args.clip_model}")
    print(f"DINO model         : {args.dino_model}")
    print(f"PySceneDetect      : {'skipped' if args.skip_pyscenedetect else 'enabled (if importable)'}")
    print("=" * 70)

    # ---- existing rows for idempotency ------------------------------------
    fieldnames = _fieldnames()
    existing_rows, existing_fields = _load_existing_csv(args.output)
    existing_by_id: Dict[str, dict] = {}
    if existing_rows and not args.force:
        for r in existing_rows:
            vid = r.get("video_id", "").strip()
            if vid:
                existing_by_id[vid] = r
        # Warn if the existing CSV's schema differs from ours -- we'll keep
        # extra historical columns by extending fieldnames.
        for extra in existing_fields:
            if extra and extra not in fieldnames:
                fieldnames.append(extra)
        print(f"[info] Loaded {len(existing_by_id)} existing rows from "
              f"{args.output}; will skip videos already covered "
              "(pass --force to recompute).")

    # ---- enumerate videos + load captions ---------------------------------
    captions_by_id = load_captions_csv(args.captions_csv)
    video_paths = list_video_paths(args.videos_dir)
    if args.max_videos > 0:
        video_paths = video_paths[: args.max_videos]
    if not video_paths:
        print(f"[error] no video files found under {args.videos_dir}",
              file=sys.stderr)
        return 2
    print(f"Videos discovered  : {len(video_paths)}")
    print(f"Captions loaded    : {len(captions_by_id)}")
    print()

    # ---- preload heavy models ---------------------------------------------
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        print(f"[error] torch not importable ({exc}); aborting", file=sys.stderr)
        return 2

    print("Loading CLIP model...")
    clip = _CLIPScorer(args.clip_model, args.device)
    print("Loading DINOv2 model...")
    dino = _DINOScorer(args.dino_model, args.device)
    resize_hw = (args.resize_h, args.resize_w)
    print()

    # ---- per-video loop ---------------------------------------------------
    n_done = 0
    n_skipped = 0
    n_errored = 0
    new_rows: List[dict] = []
    t0 = time.time()
    last_print_t = t0

    example_row: Optional[dict] = None

    for v_idx, vp in enumerate(video_paths):
        vid = _canonical_video_id(vp.name)
        if vid in existing_by_id and not args.force:
            n_skipped += 1
            continue
        cap = captions_by_id.get(vid, "")
        try:
            row = extract_one_video(
                vp, cap, visible_range, gen_range,
                clip=clip, dino=dino,
                resize_hw=resize_hw,
                batch_size=args.batch_size,
                skip_pyscenedetect=args.skip_pyscenedetect,
                diagnostic=(n_done == 0 and n_errored == 0),
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"[error] {vp.name}: {exc}", file=sys.stderr)
            traceback.print_exc()
            n_errored += 1
            continue
        new_rows.append(row)
        if example_row is None:
            example_row = row
        n_done += 1

        if (v_idx + 1) % PROGRESS_EVERY == 0 or (v_idx + 1) == len(video_paths):
            dt = time.time() - last_print_t
            last_print_t = time.time()
            print(f"  [{v_idx + 1}/{len(video_paths)}] "
                  f"new={n_done}  skip={n_skipped}  err={n_errored}  "
                  f"(+{dt:.1f}s)  vid={vid}",
                  flush=True)

    # ---- write merged CSV (existing rows preserved, new rows appended) ----
    final_rows_by_id: Dict[str, dict] = {}
    if not args.force:
        # preserve every existing row exactly as it was
        for r in existing_rows:
            v = r.get("video_id", "").strip()
            if v:
                final_rows_by_id[v] = r
    for r in new_rows:
        final_rows_by_id[r["video_id"]] = r

    sorted_rows = sorted(final_rows_by_id.values(),
                         key=lambda r: r.get("video_id", ""))
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(_format_row(row, fieldnames))

    dt_total = time.time() - t0
    print()
    print(f"Wrote {args.output}")
    print(f"  total rows now : {len(sorted_rows)}")
    print(f"  new this run   : {n_done}")
    print(f"  skipped existing: {n_skipped}")
    print(f"  errored        : {n_errored}")
    print(f"  wall time      : {dt_total:.1f}s")

    if example_row is not None:
        print()
        print("Example row (one video, post-extraction):")
        for k in fieldnames:
            v = example_row.get(k)
            if isinstance(v, float):
                v_s = f"{v:.6f}" if (v == v and not math.isinf(v)) else "NaN"
            else:
                v_s = str(v)
            if k == "caption_text" and len(v_s) > 96:
                v_s = v_s[:93] + "..."
            print(f"  {k:32s} = {v_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
