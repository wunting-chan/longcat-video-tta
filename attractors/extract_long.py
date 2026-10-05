#!/usr/bin/env python3
"""Observables for Phase 0 long rollouts: same per-frame features as extract_observables.py,
for every <model>_p<i>_s<seed>.mp4 under --videos/<model>/. Also saves 9 thumbnails per video
(start, every 20 s, end) for figures."""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

import extract_observables as E


def work(mp4):
    frames = E.load_frames(mp4)
    low = E.low_level(frames)
    h, w = frames.shape[1:3]
    s = 224 / min(h, w)
    nh, nw = round(h * s), round(w * s)
    top, left = (nh - 224) // 2, (nw - 224) // 2
    crops = np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_CUBIC)[top:top + 224, left:left + 224]
                      for f in frames])
    T = len(frames)
    idx = sorted(set([0] + list(range(0, T, 16 * 20)) + [T - 1]))
    thumbs = np.stack([cv2.resize(frames[i], (208, 120), interpolation=cv2.INTER_AREA) for i in idx])
    return mp4, low, crops, thumbs, np.array(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", source="github", trust_repo=True).to(device).eval()
    t0 = time.time()
    mp4s = sorted(glob.glob(os.path.join(a.videos, "*", "*.mp4")))
    todo = []
    for p in mp4s:
        o = Path(a.out) / Path(p).parent.name / (Path(p).stem + ".npz")
        if not o.exists():
            todo.append(p)
    with ProcessPoolExecutor(a.workers) as ex:
        for mp4, low, crops, thumbs, tidx in ex.map(work, todo):
            p = Path(mp4)
            cls = E.dino_features(model, crops, device)
            side = json.load(open(p.with_suffix(".json")))
            o = Path(a.out) / p.parent.name
            o.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(o / (p.stem + ".npz"), cls=cls, low=low, low_names=np.array(E.LOW_NAMES),
                                thumbs=thumbs, thumb_idx=tidx, prompt_index=side["prompt_index"], seed=side["seed"], stem=p.stem,
                                perturb_eps=-1.0 if side.get("perturb_eps") is None else side["perturb_eps"],
                                local_attn=-1 if side.get("local_attn") is None else side["local_attn"],
                                sink=-1 if side.get("sink") is None else side["sink"],
                                model=side["model"], fps=16)
            print(p.stem, len(cls), f"{time.time() - t0:.0f}s", flush=True)
    print("ALL_DONE")


if __name__ == "__main__":
    main()
