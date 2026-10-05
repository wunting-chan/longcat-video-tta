# API dump (verbatim, extracted from GitHub clone)

## common.py

### load_longcat_components
```python
def load_longcat_components(
    checkpoint_dir: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cp_split_hw: Optional[list] = None,
) -> Dict[str, object]:
    """Load all LongCat-Video components from a checkpoint directory.

    Returns a dict with keys: tokenizer, text_encoder, vae, scheduler, dit, pipe.
    """
    if cp_split_hw is None:
        cp_split_hw = [1, 1]

    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, subfolder="tokenizer", torch_dtype=dtype,
        use_fast=False,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint_dir, subfolder="text_encoder", torch_dtype=dtype
    )
    vae = AutoencoderKLWan.from_pretrained(
        checkpoint_dir, subfolder="vae", torch_dtype=dtype
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint_dir, subfolder="scheduler", torch_dtype=dtype
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw,
        enable_flashattn2=True, torch_dtype=dtype,
    )

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )

    # Move to device
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    dit = dit.to(device)

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "scheduler": scheduler,
        "dit": dit,
        "pipe": pipe,
    }

```

### load_video_frames
```python
def load_video_frames(
    video_path: str,
    num_frames: int,
    height: int = 480,
    width: int = 832,
    start_frame: int = 0,
) -> torch.Tensor:
    """Load video frames as a tensor [1, C, T, H, W] in [-1, 1].

    Args:
        video_path: path to video file.
        num_frames: how many frames to return.
        height, width: target spatial resolution.
        start_frame: skip this many decoded frames before collecting.
                     Useful for the anchor-based scheme where conditioning
                     starts at ``anchor - num_cond`` instead of frame 0.
    """
    import av
    container = av.open(video_path)
    frames = []
    decoded = 0
    for frame in container.decode(video=0):
        if decoded < start_frame:
            decoded += 1
            continue
        if len(frames) >= num_frames:
            break
        img = frame.to_ndarray(format="rgb24")
        frames.append(img)
        decoded += 1
    container.close()

    if len(frames) == 0:
        raise ValueError(f"No frames decoded from {video_path}")
    # Pad with last frame if not enough
    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames_np = np.stack(frames[:num_frames], axis=0)  # (T, H, W, 3)
    frames_t = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()  # (3, T, H, W)
    frames_t = frames_t / 255.0

    # Resize
    frames_t = F.interpolate(
        frames_t.unsqueeze(0),
        size=(frames_t.shape[1], height, width),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)

    # Normalize to [-1, 1]
    frames_t = frames_t * 2.0 - 1.0
    return frames_t.unsqueeze(0)  # (1, 3, T, H, W)

```

### encode_video
```python
def encode_video(
    vae: AutoencoderKLWan,
    pixel_frames: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Encode pixel frames [B, C, T, H, W] into VAE latents.

    If *normalize* is True, applies the VAE's latent normalization
    (mean/std shift) that LongCat-Video expects.
    """
    from longcat_video.pipeline_longcat_video import retrieve_latents

    with torch.no_grad():
        posterior = vae.encode(pixel_frames)
        latents = retrieve_latents(posterior)

    if normalize:
        latents = normalize_latents(vae, latents)
    return latents

```

### decode_latents
```python
def decode_latents(
    vae: AutoencoderKLWan,
    latents: torch.Tensor,
    denorm: bool = True,
) -> torch.Tensor:
    """Decode latents to pixel frames [B, C, T, H, W] in [0, 1]."""
    if denorm:
        latents = denormalize_latents(vae, latents)
    with torch.no_grad():
        video = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
    # video is in [-1, 1]
    video = (video + 1.0) / 2.0
    return video.clamp(0, 1)

```

### encode_prompt
```python
def encode_prompt(
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    prompt: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_length: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a text prompt. Returns (prompt_embeds, attention_mask)."""
    inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        embeds = text_encoder(input_ids, mask).last_hidden_state

    embeds = embeds.to(dtype=dtype, device=device)
    # Shape for LongCat-Video: [B, 1, N, C]
    embeds = embeds.unsqueeze(1)
    return embeds, mask


```

### generate_video_continuation
```python
def generate_video_continuation(
    pipe: Any,
    video_frames: list,
    prompt: str,
    num_cond_frames: int = 13,
    num_frames: int = 93,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    seed: int = 42,
    resolution: str = "480p",
    device: str = "cuda",
    use_kv_cache: bool = True,
) -> np.ndarray:
    """Generate video continuation using the pipeline.

    Parameters
    ----------
    video_frames : list of PIL Images (conditioning frames)
    prompt : text prompt
    Returns np.ndarray of shape [N, H, W, 3] in [0, 1].
    """
    from PIL import Image

    vae_temporal_factor = 4
    num_frames_valid = (
        ((num_frames - 1 + vae_temporal_factor - 1) // vae_temporal_factor)
        * vae_temporal_factor + 1
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    output = pipe.generate_vc(
        video=video_frames,
        prompt=prompt,
        resolution=resolution,
        num_frames=num_frames_valid,
        num_cond_frames=num_cond_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        use_kv_cache=use_kv_cache,
        offload_kv_cache=False,
    )[0]

    return output  # np array [N, H, W, 3]
```

### evaluate_generation_metrics
```python
def evaluate_generation_metrics(
    gen_output: np.ndarray,
    video_path: str,
    num_cond_frames: int,
    num_gen_frames: int,
    gen_start_frame: int,
    device: str = "cuda",
    return_gt_frames: bool = False,
) -> Dict[str, Any]:
    """Compute PSNR, SSIM, LPIPS between generated and ground truth frames.

    Parameters
    ----------
    gen_output : np.ndarray
        Full pipeline output of shape [N, H, W, 3] in [0, 1], where
        N = num_cond_frames + num_generated.
    video_path : str
        Path to the source video for loading GT frames.
    num_cond_frames : int
        Number of conditioning frames at the start of gen_output.
    num_gen_frames : int
        Number of generated frames to evaluate (may be less than total gen).
    gen_start_frame : int
        Anchor frame index — GT starts at this frame in the source video.
    device : str
        Device for LPIPS computation.
    return_gt_frames : bool
        If True, include ``"gt_frames_hwc"`` (np.ndarray [T,H,W,3] in [0,1])
        in the returned dict so callers can reuse decoded GT without
        re-opening the video.

    Returns
    -------
    dict with keys: psnr, ssim, lpips, and optionally gt_frames_hwc
    """
    from PIL import Image
    import av

    gen_frames = gen_output[num_cond_frames:num_cond_frames + num_gen_frames]
    out_h, out_w = gen_frames.shape[1], gen_frames.shape[2]

    container = av.open(video_path)
    gt_pil = []
    decoded = 0
    for frame in container.decode(video=0):
        if decoded < gen_start_frame:
            decoded += 1
            continue
        if len(gt_pil) >= num_gen_frames:
            break
        gt_pil.append(frame.to_image())
        decoded += 1
    container.close()

    n_compare = min(len(gen_frames), len(gt_pil))
    if n_compare == 0:
        result: Dict[str, Any] = {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan")}
        if return_gt_frames:
            result["gt_frames_hwc"] = None
        return result

    gt_np = np.stack([
        np.array(img.resize((out_w, out_h), Image.LANCZOS)) / 255.0
        for img in gt_pil[:n_compare]
    ], axis=0).astype(np.float32)  # [T, H, W, 3]
    gen_np = gen_frames[:n_compare].astype(np.float32)

    # PSNR (per-frame, then average)
    psnr_vals = []
    for i in range(n_compare):
        mse = np.mean((gen_np[i] - gt_np[i]) ** 2)
        if mse < 1e-10:
            psnr_vals.append(50.0)
        else:
            psnr_vals.append(float(10.0 * np.log10(1.0 / mse)))
    psnr = float(np.mean(psnr_vals))

    # SSIM (per-frame, then average)
    ssim_vals = []
    for i in range(n_compare):
        p = torch.from_numpy(gen_np[i]).permute(2, 0, 1).unsqueeze(0).float()
        g = torch.from_numpy(gt_np[i]).permute(2, 0, 1).unsqueeze(0).float()
        ssim_vals.append(_ssim_single(p, g))
    ssim = float(np.mean(ssim_vals))

    # LPIPS (uses cached model)
    try:
        loss_fn = _get_lpips_model(device)
        lpips_vals = []
        for i in range(n_compare):
            p = torch.from_numpy(gen_np[i]).permute(2, 0, 1).unsqueeze(0).float().to(device)
            g = torch.from_numpy(gt_np[i]).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                s = loss_fn(p * 2 - 1, g * 2 - 1)
            lpips_vals.append(s.item())
        lpips_val = float(np.mean(lpips_vals))
    except ImportError:
        lpips_val = float("nan")

    result = {"psnr": psnr, "ssim": ssim, "lpips": lpips_val}
    if return_gt_frames:
        result["gt_frames_hwc"] = gt_np
    return result

```

### load_ucf101_video_list
```python
def load_ucf101_video_list(
    data_dir: str,
    max_videos: int = 100,
    seed: int = 42,
    stratified: bool = True,
    validate_decodable: bool = False,
) -> List[Dict]:
    """Load a list of video entries from a dataset directory.

    Reads metadata.csv if present (columns: filename, caption, category).
    Falls back to scanning for video files and using directory names.

    Returns list of dicts with keys: video_path, caption, class_name.
    """
    import csv as _csv

    data_dir = Path(data_dir)
    video_entries = []

    meta_path = data_dir / "metadata.csv"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", row.get("video_path", ""))
                vp = data_dir / "videos" / fname
                if not vp.exists():
                    vp = data_dir / fname
                if not vp.exists():
                    continue
                caption = resolve_caption_from_row(row)
                class_name = row.get("category", row.get("class_name", "unknown"))
                video_entries.append({
                    "video_path": str(vp),
                    "caption": caption,
                    "class_name": class_name,
                })
        if video_entries:
            print(f"  Loaded {len(video_entries)} videos from {meta_path}")
    
    if not video_entries:
        for ext in ("*.mp4", "*.avi"):
            for vp in sorted(data_dir.rglob(ext)):
                class_name = vp.parent.name if vp.parent != data_dir else "unknown"
                caption = class_name.replace("_", " ")
                video_entries.append({
                    "video_path": str(vp),
                    "caption": caption,
                    "class_name": class_name,
                })

    if not video_entries:
        raise FileNotFoundError(f"No video files found in {data_dir}")

    if validate_decodable:
        valid_entries = []
        bad_examples: List[str] = []
        for entry in video_entries:
            vp = entry.get("video_path")
            ok = True
            try:
                import av
                container = av.open(str(vp))
                try:
                    next(container.decode(video=0))
                except StopIteration:
                    ok = False
                finally:
                    container.close()
            except Exception:
                ok = False
            if ok:
                valid_entries.append(entry)
            elif len(bad_examples) < 5:
                bad_examples.append(str(vp))

        dropped = len(video_entries) - len(valid_entries)
        if dropped > 0:
            print(f"  Dropped {dropped} undecodable videos during dataset load.")
            for ex in bad_examples:
                print(f"    bad_video: {ex}")
        video_entries = valid_entries

    if not video_entries:
        raise FileNotFoundError(f"No decodable video files found in {data_dir}")

    rng = np.random.RandomState(seed)
    # Panda datasets should use plain random sampling to honor max_videos.
    data_dir_lower = str(data_dir).lower()
    if "panda" in data_dir_lower and stratified:
        print("  Stratified sampling disabled for Panda dataset path.")
        stratified = False

    if stratified:
        from collections import defaultdict
        from collections import Counter
        by_class = defaultdict(list)
        for entry in video_entries:
            by_class[entry["class_name"]].append(entry)

        n_classes = len(by_class)
        class_sizes = Counter(entry["class_name"] for entry in video_entries)
        singleton_ratio = (
            sum(1 for _, c in class_sizes.items() if c == 1) / max(n_classes, 1)
        )
        # Panda metadata often has many singleton "classes", which collapses
        # sampling to ~number of unique labels (e.g., 86). Fall back to random
        # sampling in that case so max_videos is respected.
        if singleton_ratio > 0.5 and n_classes > max_videos // 2:
            print(
                "  Stratified sampling disabled: many singleton classes "
                f"(classes={n_classes}, singleton_ratio={singleton_ratio:.2f})."
            )
            rng.shuffle(video_entries)
            return video_entries[:max_videos]

        per_class = max(1, max_videos // n_classes)
        selected = []
        leftover = []
        for cls in sorted(by_class.keys()):
            entries = by_class[cls]
            rng.shuffle(entries)
            selected.extend(entries[:per_class])
            leftover.extend(entries[per_class:])

        if len(selected) < max_videos and leftover:
            rng.shuffle(leftover)
            selected.extend(leftover[: max_videos - len(selected)])

        rng.shuffle(selected)
        return selected[:max_videos]
    else:
        rng.shuffle(video_entries)
        return video_entries[:max_videos]

```

### apply_fixed_caption
```python
def apply_fixed_caption(
    video_entries: List[Dict[str, Any]],
    fixed_caption: Optional[str],
    *,
    context: str = "eval",
) -> List[Dict[str, Any]]:
    """Override all captions with one fixed caption string."""
    if fixed_caption is None:
        return video_entries
    cap = str(fixed_caption).strip()
    # If callers accidentally pass shell-quoted literals (e.g. '"videos"'),
    # normalize back to plain text to avoid silent caption drift.
    if len(cap) >= 2 and cap[0] == cap[-1] and cap[0] in ("'", '"'):
        cap = cap[1:-1]
    for row in video_entries:
        row["caption"] = cap
    print(f"[caption_override:{context}] applied fixed caption to {len(video_entries)} videos: {cap!r}")
    return video_entries

```

### torch_gc (definition 1, line 2354)
```python
def torch_gc():
    """Release GPU memory aggressively."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
```

### torch_gc (definition 2, line 2452 -- this one wins at import)
```python
def torch_gc():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
```

### save_results
```python
def save_results(results: dict, output_path: str):
    """Save experiment results to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
```

### load_checkpoint
```python
def load_checkpoint(checkpoint_path: str) -> Optional[dict]:
    """Load a checkpoint JSON if it exists."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return None
```

### save_checkpoint
```python
def save_checkpoint(checkpoint: dict, checkpoint_path: str):
    """Save checkpoint JSON."""
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)

```

### aggregate_quality_metrics
```python
def aggregate_quality_metrics(summary: dict):
    """Compute avg PSNR, SSIM, LPIPS from per-video results and merge into summary."""
    successful = [r for r in summary.get("results", []) if r.get("success")]
    for key in ("psnr", "ssim", "lpips"):
        values = [r[key] for r in successful if r.get(key) is not None]
        summary[key] = round(float(np.mean(values)), 6) if values else None

```

### add_online_eval_args
```python
def add_online_eval_args(parser: "argparse.ArgumentParser"):
    """Add --compute-fvd, --compute-fid, --compute-vbench, --min-fvd-videos, --gt-features-cache."""
    grp = parser.add_argument_group("Online distributional metrics")
    grp.add_argument("--compute-fvd", action="store_true",
                     help="Compute FVD online (no saved videos needed)")
    grp.add_argument("--compute-fid", action="store_true",
                     help="Compute per-frame FID online (requires --compute-fvd)")
    grp.add_argument("--compute-vbench", action="store_true",
                     help="Run VBench++ at end of job (requires saved videos)")
    grp.add_argument("--min-fvd-videos", type=int,
                     default=_DEFAULT_MIN_FVD_VIDEOS,
                     help="Minimum videos before FVD is considered reliable "
                          f"(default: {_DEFAULT_MIN_FVD_VIDEOS})")
    grp.add_argument("--gt-features-cache", type=str, default=None,
                     help="Path to pre-computed GT features .npz from "
                          "precompute_gt_features.py. When set, the GT "
                          "distribution is loaded once and never re-extracted.")

```

### finalize_online_eval
```python
def finalize_online_eval(
    accumulator: Optional[OnlineFrechetAccumulator],
    summary: dict,
    videos_dir: str,
    args,
):
    """Compute FVD/FID from the accumulator and optionally run VBench++.

    Merges results into *summary* in-place so they are saved alongside
    existing PSNR / SSIM / LPIPS metrics.
    """
    if accumulator is not None:
        print("\n[Online eval] Computing FVD/FID from accumulated features...")
        fvd_results = accumulator.compute()
        summary.update(fvd_results)
        for k, v in fvd_results.items():
            print(f"  {k}: {v}")

        stats_path = os.path.join(os.path.dirname(videos_dir), "fvd_fid_stats.npz")
        np.savez(stats_path, **accumulator.export_stats())
        print(f"  Saved sufficient statistics to {stats_path}")

    vbench_skipped = True
    if getattr(args, "compute_vbench", False):
        mp4s = sorted(Path(videos_dir).glob("*.mp4")) if os.path.isdir(videos_dir) else []
        if mp4s:
            print(f"\n[VBench++] Running on {len(mp4s)} videos in {videos_dir}...")
            try:
                from vbench import VBench
                import vbench as _vbench_pkg

                _VBENCH_DIMS = [
                    "subject_consistency",
                    "background_consistency",
                    "motion_smoothness",
                    "dynamic_degree",
                    "aesthetic_quality",
                    "imaging_quality",
                ]

                pkg_dir = os.path.dirname(_vbench_pkg.__file__)
                full_info_json = os.path.join(pkg_dir, "VBench_full_info.json")
                if not os.path.exists(full_info_json):
                    full_info_json = os.path.join(
                        os.path.dirname(pkg_dir), "vbench", "VBench_full_info.json"
                    )

                vbench_output = os.path.join(
                    os.path.dirname(videos_dir), "vbench_results"
                )
                os.makedirs(vbench_output, exist_ok=True)

                vb = VBench(torch.device("cuda"), full_info_json, vbench_output)

                vbench_scores: Dict[str, Any] = {}
                for dim in _VBENCH_DIMS:
                    try:
                        print(f"  Evaluating {dim}...")
                        vb.evaluate(
                            videos_path=videos_dir,
                            name=f"vbench_{dim}",
                            dimension_list=[dim],
                            mode="custom_input",
                        )
                        result_file = os.path.join(
                            vbench_output, f"vbench_{dim}_eval_results.json"
                        )
                        if os.path.exists(result_file):
                            import json as _json
                            with open(result_file) as _f:
                                dim_results = _json.load(_f)
                            overall = _extract_vbench_score(dim, dim_results)
                            if overall is not None:
                                vbench_scores[dim] = overall
                            else:
                                vbench_scores[dim] = dim_results
                        print(f"    {dim}: {vbench_scores.get(dim, 'N/A')}")
                    except Exception as exc:
                        print(f"  WARNING: VBench++ {dim} failed: {exc}",
                              file=sys.stderr)
                        vbench_scores[dim] = None
                summary["vbench"] = vbench_scores
                vbench_skipped = False
            except ImportError:
                print("  WARNING: vbench not installed, skipping VBench++. "
                      "Install with: pip install vbench",
                      file=sys.stderr)
        else:
            print("[VBench++] No saved videos found; skipping "
                  "(requires NO_SAVE_VIDEOS=0).")

    summary["vbench_skipped"] = vbench_skipped
```

### OnlineFrechetAccumulator (class docstring + __init__)
```python
class OnlineFrechetAccumulator:
    """Incrementally accumulate I3D (and optionally InceptionV3) features
    for online FVD / FID computation.  No video files on disk required.

    When *gt_cache_path* is provided (an .npz from ``precompute_gt_features.py``),
    the reference distribution is loaded once at init and never re-extracted.
    Only generated features are accumulated per video, ensuring the GT
    distribution is identical across all runs.
    """

    def __init__(
        self,
        device: str = "cuda",
        compute_fid: bool = False,
        min_videos: int = _DEFAULT_MIN_FVD_VIDEOS,
        gt_cache_path: Optional[str] = None,
    ):
        self.device = device
        self.compute_fid = compute_fid
        self.min_videos = min_videos
        self._gt_cached = False

        self._i3d: Optional["torch.jit.ScriptModule"] = None
        self._inception: Optional[nn.Module] = None

        d = _I3D_FEATURE_DIM
        self._gen_sum = np.zeros(d, dtype=np.float64)
        self._gen_cov = np.zeros((d, d), dtype=np.float64)
        self._ref_sum = np.zeros(d, dtype=np.float64)
        self._ref_cov = np.zeros((d, d), dtype=np.float64)
        self._gen_count = 0
        self._ref_count = 0

        if compute_fid:
            fd = _FID_FEATURE_DIM
            self._fid_gen_sum = np.zeros(fd, dtype=np.float64)
            self._fid_gen_cov = np.zeros((fd, fd), dtype=np.float64)
            self._fid_ref_sum = np.zeros(fd, dtype=np.float64)
            self._fid_ref_cov = np.zeros((fd, fd), dtype=np.float64)
            self._fid_gen_frames = 0
            self._fid_ref_frames = 0

        if gt_cache_path is not None:
            self._load_gt_cache(gt_cache_path)

```

### OnlineFrechetAccumulator.update (signature)
```python
    def update(
        self,
        gen_output: np.ndarray,
        video_path: str,
        num_cond_frames: int,
        num_gen_frames: int,
        gen_start_frame: int,
        gt_frames_hwc: Optional[np.ndarray] = None,
    ):
```

### OnlineFrechetAccumulator.compute (called by finalize_online_eval; full body)
```python
    def compute(self) -> Dict[str, Any]:
        """Return FVD (and FID) metrics from accumulated statistics."""
        result: Dict[str, Any] = {}

        if self._gen_count < 2:
            result["fvd"] = None
            result["fvd_num_videos"] = self._gen_count
            result["fvd_error"] = "Need at least 2 videos for FVD"
            return result

        fvd = _compute_frechet_distance(
            self._gen_sum, self._gen_cov, self._gen_count,
            self._ref_sum, self._ref_cov, self._ref_count,
        )
        result["fvd"] = round(fvd, 6)
        result["fvd_num_videos"] = self._gen_count
        result["fvd_num_ref_videos"] = self._ref_count
        result["fvd_gt_cached"] = self._gt_cached
        result["fvd_feature_extractor"] = "i3d_kinetics400_torchscript"
        result["fvd_feature_dim"] = _I3D_FEATURE_DIM

        if self._gen_count < self.min_videos:
            result["fvd_sample_size_warning"] = (
                f"FVD computed with {self._gen_count} videos "
                f"(recommended >= {self.min_videos}). "
                f"Covariance estimate may be unreliable."
            )

        if self.compute_fid and self._fid_gen_frames >= 2:
            fid = _compute_frechet_distance(
                self._fid_gen_sum, self._fid_gen_cov, self._fid_gen_frames,
                self._fid_ref_sum, self._fid_ref_cov, self._fid_ref_frames,
            )
            result["fid"] = round(fid, 6)
            result["fid_num_frames_gen"] = self._fid_gen_frames
            result["fid_num_frames_ref"] = self._fid_ref_frames
            result["fid_feature_extractor"] = "inception_v3_imagenet"
            result["fid_feature_dim"] = _FID_FEATURE_DIM

        return result

```

### OnlineFrechetAccumulator.export_stats (called by finalize_online_eval; signature)
```python
    def export_stats(self) -> Dict[str, np.ndarray]:
        """Export sufficient statistics for merging across chunked runs.

```

### OnlineFrechetAccumulator.save_stats / load_stats (signatures + bodies)
```python
    def save_stats(self, path: str):
        """Save accumulator state to .npz for checkpoint resume."""
        np.savez(path, **self.export_stats())

    def load_stats(self, path: str):
        """Load accumulator state from .npz (checkpoint resume)."""
        if os.path.exists(path):
            stats = dict(np.load(path, allow_pickle=True))
            self.import_stats(stats)
            print(f"[FVD/FID] Restored accumulator state: "
                  f"{self._gen_count} gen, {self._ref_count} ref videos")
```

## savi_dno_longcat.py

### SAViDNO_LongCat.__init__
```python
    def __init__(
        self,
        dit,
        vae,
        scheduler,
        tokenizer,
        text_encoder,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_inference_steps: int = 10,
        guidance_scale: float = 4.0,
        lr: float = 0.01,
        lam: float = 0.0012,
        p: float = 0.9,
        feature_model: Optional[nn.Module] = None,
        gradient_checkpointing: bool = True,
        latent_loss: bool = False,
        max_grad_norm: float = 1.0,
        regularizer: str = "none",
        reg_weight: float = 0.0,
        noise_interp: bool = True,
        generation_use_cfg: bool = False,
        generation_steps: Optional[int] = None,
        generation_enhance_hf: bool = False,
        generation_apg: bool = False,
        tango_guidance: bool = False,
        tango_lambda: float = 0.0,
        tango_sigma_hi: float = 0.9,
        tango_sigma_lo: float = 0.0,
        tango_kurtosis: float = 0.0,
    ):
        self.device = device
        self.dtype = dtype
        self.dit = dit
        self.vae = vae
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        # LongCat's real inference (generate_vc, ~19 dB) uses CFG on (guidance
        # 4.0) and 50 Euler steps; the SAVi-DNO PVDM recipe used no-CFG / 10
        # steps, which on a LongCat backbone yields near-garbage (SSIM~0.05).
        # These control the (no-grad) PREDICTION passes only; the differentiable
        # optimization inner loop still uses num_inference_steps to stay cheap.
        self.generation_use_cfg = bool(generation_use_cfg)
        self.generation_steps = int(generation_steps) if generation_steps else int(num_inference_steps)
        # generate_vc's enhance_hf reshapes the low-noise tail (uniform 500->0)
        # and optimized_scale/APG rescales guidance. Both are OPT-IN: on the 2-video
        # diagnostic they net-HURT this reimplementation (enhance_hf tanked the
        # near-static video1 20.2 -> 15.0 dB), so the plain sampler (~20-23 dB) is
        # the default. Flags kept to A/B each one independently.
        self.generation_enhance_hf = bool(generation_enhance_hf)
        self.generation_apg = bool(generation_apg)

        # --- EXP3: TANGO-style predicted-noise-gaussianity guidance ------------
        # Per-step, training-free intervention applied DURING sampling (targets
        # FVD, the distribution-level metric, unlike AdaSteer's per-video delta).
        # In LongCat's rectified flow the code's DiT output v_pred = x0 - eps
        # (the -dt*v_pred Euler update negates the raw velocity), so at sigma the
        # implied clean/noise estimates are:
        #     x0_hat  = x_t + sigma      * v_pred
        #     eps_hat = x_t - (1 - sigma)* v_pred
        # TANGO nudges the trajectory so eps_hat stays close to N(0, I) (zero
        # mean / unit variance, optional low excess-kurtosis). Moving eps_hat by
        # -lambda*grad(G) is equivalent to  v_pred += lambda*grad(G)  (since
        # eps_hat = x_t - (1-sigma)v_pred), so no extra DiT backward is needed
        # (cf. TTC's per-step velocity rewrite). Applied only for sigma in
        # [sigma_lo, sigma_hi] to avoid the 1/(1-sigma) blow-up near pure noise.
        self.tango_guidance = bool(tango_guidance)
        self.tango_lambda = float(tango_lambda)
        self.tango_sigma_hi = float(tango_sigma_hi)
        self.tango_sigma_lo = float(tango_sigma_lo)
        self.tango_kurtosis = float(tango_kurtosis)
        self.lr = lr
        self.lam = lam
        self.p = p
        self.gradient_checkpointing = gradient_checkpointing
        self.latent_loss = latent_loss
        self.max_grad_norm = max_grad_norm

        # In-distribution noise regularizer for published noise-opt methods.
        # regularizer="none" + noise_interp=True reproduces SAVi-DNO exactly.
        if regularizer not in _REGULARIZERS:
            raise ValueError(
                "regularizer must be one of %s (got %r)"
                % (list(_REGULARIZERS), regularizer)
            )
        self.regularizer = regularizer
        self.reg_fn = _REGULARIZERS[regularizer]
        self.reg_weight = float(reg_weight)
        self.noise_interp = bool(noise_interp)

        self.feature_model = None
        if feature_model is not None and not latent_loss:
            self.feature_model = feature_model
            for param in self.feature_model.parameters():
                param.requires_grad = False

        self.eps_optimized = None
        self.optimizer = None

        # Cache null prompt embeddings for CFG
        self._null_embeds = None
        self._null_mask = None

```

### _build_sigmas
```python
    def _build_sigmas(self, n_steps: Optional[int] = None, enhance_hf: bool = False):
        """Build LongCat's flow-matching sigma schedule (from ~1.0 to 0.0).

        Matches pipeline_longcat_video.generate_vc: an EXPLICIT
        linspace(1, 0.001, steps) schedule passed to the scheduler, NOT the
        scheduler's DEFAULT (resolution-shifted) sigmas. Using the default
        sigmas gave the DiT the wrong per-token timestep conditioning.

        Returns scheduler.sigmas (N+1 values including terminal 0.0),
        NOT scheduler.timesteps (which are sigma*1000).
        """
        steps = int(n_steps) if n_steps else self.num_inference_steps
        sigmas = torch.linspace(1, 0.001, steps).to(torch.float32)
        self.scheduler.set_timesteps(steps, sigmas=sigmas, device=self.device)
        if not enhance_hf:
            return self.scheduler.sigmas
        # enhance_hf (generate_vc): keep timesteps > 500 from the linspace, then
        # append 10 uniform steps 500 -> 50 (np.linspace(500,0,10,endpoint=False)),
        # then terminal sigma 0. Reshapes the low-noise tail for sharper detail.
        timesteps = self.scheduler.timesteps
        tail = (500.0 - 50.0 * torch.arange(
            10, device=timesteps.device, dtype=torch.float32
        ))  # [500, 450, ..., 50]
        filtered = timesteps[timesteps > 500].to(torch.float32)
        new_ts = torch.cat([filtered, tail])
        self.scheduler.timesteps = new_ts
        self.scheduler.sigmas = torch.cat(
            [new_ts / 1000.0, torch.zeros(1, device=new_ts.device, dtype=torch.float32)]
        )
        return self.scheduler.sigmas

```

### _dit_forward_step
```python
    def _dit_forward_step(
        self,
        x_t: torch.Tensor,
        cond_latents: torch.Tensor,
        t_value: float,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Single DiT forward pass with conditioning.

        Concatenates [cond_clean, x_t] along temporal dim, builds per-token
        timesteps (cond=0, target=t*1000), and runs the DiT.

        Returns the velocity prediction for the TARGET portion only.
        """
        cfg = _get_model_config(self.dit)
        patch_t = cfg.patch_size[0]

        B, C, T_target, H_lat, W_lat = x_t.shape
        T_cond = cond_latents.shape[2]
        T_total = T_cond + T_target
        N_cond = T_cond // patch_t
        N_target = T_target // patch_t
        N_total = N_cond + N_target

        hidden_states = torch.cat([cond_latents, x_t], dim=2).to(self.dtype)

        timestep = torch.zeros(B, N_total, device=self.device, dtype=self.dtype)
        timestep[:, N_cond:] = t_value * 1000.0

        pred = self.dit(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_mask,
            num_cond_latents=N_cond,
        )

        return pred[:, :, T_cond:]

```

### _dit_forward_step_cfg
```python
    def _dit_forward_step_cfg(
        self,
        x_t: torch.Tensor,
        cond_latents: torch.Tensor,
        t_value: float,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """DiT forward with classifier-free guidance."""
        null_embeds, null_mask = self._get_null_embeds()

        v_cond = self._dit_forward_step(
            x_t, cond_latents, t_value, prompt_embeds, prompt_mask,
        )
        v_uncond = self._dit_forward_step(
            x_t, cond_latents, t_value, null_embeds, null_mask,
        )

        if not self.generation_apg:
            # Vanilla CFG (default): on the 2-video diagnostic this + plain schedule
            # gave the best faithful reimpl (~20-23 dB).
            return v_uncond + self.guidance_scale * (v_cond - v_uncond)

        # Adaptive projected guidance (opt-in, matches generate_vc's optimized_scale):
        # rescale the uncond branch by its projection onto the cond branch before
        # the guidance push, which avoids the over-saturation of vanilla CFG.
        B = v_cond.shape[0]
        pos = v_cond.reshape(B, -1).float()
        neg = v_uncond.reshape(B, -1).float()
        st = (pos * neg).sum(dim=1) / (neg.pow(2).sum(dim=1) + 1e-8)
        st = st.view(B, *([1] * (v_cond.dim() - 1))).to(v_cond.dtype)
        return v_uncond * st + self.guidance_scale * (v_cond - v_uncond * st)

```

## run_savi_dno_longcat.sbatch
```bash
#!/bin/bash
#SBATCH --job-name=savi_dno_lc
# ONE H200 by default.  The old 2-GPU naive pipeline split (blocks 0-23 on
# cuda:0, 24-47 on cuda:1) leaves each GPU idle ~half the time (~50% util),
# which trips Torch's aggressive Low-GPU-Utilization policy on gh* nodes
# (auto-cancel below 60%) -> that is what killed jobs 15044153/54 at ~2h, NOT
# the 48h wall clock.  A single GPU runs all blocks sequentially at ~100% util
# and (since the pipeline was not microbatched) is not slower.  A LongCat DiT
# forward with gradient checkpointing fits well within one H200's 140 GiB.
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_36_mren
#SBATCH --requeue
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --output=comparison_methods/slurm_log/savi_dno_longcat_%j.out
#SBATCH --error=comparison_methods/slurm_log/savi_dno_longcat_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1
# Single-GPU differentiable DiT backprop used to OOM by ~200 MiB because the
# stock DiT forward ran its 48-block loop un-checkpointed (~139 GiB live). The
# script now enables PER-BLOCK gradient checkpointing on 1 GPU (caps live
# activation at ~1 block), so it fits one H200 with large headroom. Keep
# expandable_segments to further reduce allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=============================================================================="
echo "SAVi-DNO Evaluation (Noise Optimization on LongCat Backbone)"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Euler steps: ${SAVI_EULER_STEPS:-10}"
echo "Num GPUs: ${NUM_GPUS:-1}"
echo "LR: ${SAVI_LR:-1e-4}"
echo "Lambda: ${SAVI_LAM:-0.0012}"
echo "p: ${SAVI_P:-0.7}"
echo "Guidance: ${SAVI_GUIDANCE:-4.0}"
echo "Loss mode: ${SAVI_LOSS_MODE:-latent} (default: latent loss for LongCat)"
echo "Rollout steps: ${SAVI_ROLLOUT_STEPS:-10}"
echo "Max grad norm: ${SAVI_MAX_GRAD_NORM:-1.0}"
echo "No-optimize: ${SAVI_NO_OPTIMIZE:-}"
echo "Oracle-leak: ${SAVI_ORACLE_LEAK:-0} (1=leaky oracle upper bound; default 0=fair leakage-free)"
echo "Method: ${NOISE_OPT_METHOD:-savi_dno} (savi_dno=default; dno=Karunratanakul CVPR24; direct_noise_opt=Tang ICML25)"
echo "Reg weight: ${NOISE_OPT_REG_WEIGHT:--1} (<0 = method default)"
echo "Geometry: cond=${SAVI_NUM_COND_FRAMES:-14} total=${SAVI_NUM_FRAMES:-28} gen_start=${SAVI_GEN_START_FRAME:-48}"
echo "=============================================================================="

NOISE_OPT_METHOD="${NOISE_OPT_METHOD:-savi_dno}"

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${SAVI_LC_DATA_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
# Default output dir is keyed on Euler steps AND the variant (optimized / no-opt
# / oracle) so distinct runs can NEVER silently overwrite each other's
# summary.json when SAVI_LC_OUTPUT_DIR is not set explicitly.
_SAVI_VARIANT=""
if [ "${SAVI_NO_OPTIMIZE:-}" = "1" ]; then
    _SAVI_VARIANT="_noopt"
elif [ "${SAVI_ORACLE_LEAK:-}" = "1" ]; then
    _SAVI_VARIANT="_oracle"
fi
# Output dir is also keyed on the noise-opt method so dno / direct_noise_opt
# never collide with savi_dno. Default (savi_dno) keeps the original path.
case "${NOISE_OPT_METHOD}" in
    dno)              _METHOD_BASE="dno_longcat" ;;
    direct_noise_opt) _METHOD_BASE="direct_noise_opt_longcat" ;;
    *)                _METHOD_BASE="savi_dno_longcat" ;;
esac
OUTPUT_DIR="${SAVI_LC_OUTPUT_DIR:-${PROJECT_ROOT}/comparison_methods/results/${_METHOD_BASE}_s${SAVI_EULER_STEPS:-10}${_SAVI_VARIANT}}"
MAX_VIDEOS="${SAVI_LC_MAX_VIDEOS:-1000}"

# ============================================================================
# Environment Setup (LongCat conda env)
# ============================================================================
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
if [ -d "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
    echo "Activated conda: $CONDA_ENV"
else
    echo "ERROR: Conda environment not found at $CONDA_ENV" >&2
    exit 1
fi

unset PYTHONHOME
unset PYTHONPATH

PYTHON="${CONDA_ENV}/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version 2>&1))"

"$PYTHON" -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')" || {
    echo "ERROR: torch import failed" >&2
    exit 1
}

cd "${PROJECT_ROOT}"

# ============================================================================
# Build flags
# ============================================================================
GT_CACHE_FLAG=""
if [ -n "${GT_FEATURES_CACHE:-}" ] && [ -f "${GT_FEATURES_CACHE}" ]; then
    GT_CACHE_FLAG="--gt-features-cache ${GT_FEATURES_CACHE}"
fi
SAVE_LIST_FLAG=""
if [ -n "${SAVE_ONLY_LIST:-}" ] && [ -f "${SAVE_ONLY_LIST}" ]; then
    SAVE_LIST_FLAG="--save-only-list ${SAVE_ONLY_LIST}"
fi
NO_OPT_FLAG=""
if [ -n "${SAVI_NO_OPTIMIZE:-}" ] && [ "${SAVI_NO_OPTIMIZE}" = "1" ]; then
    NO_OPT_FLAG="--no-optimize"
fi
LOSS_FLAG=""
if [ "${SAVI_LOSS_MODE:-latent}" = "pixel" ]; then
    LOSS_FLAG="--pixel-loss"
fi
# Fair protocol is the DEFAULT (adapt noise on observed history, predict unseen
# future). Set SAVI_ORACLE_LEAK=1 only to produce a labelled oracle upper bound
# (optimizes noise against the scored future = ground-truth leakage).
ORACLE_FLAG=""
if [ -n "${SAVI_ORACLE_LEAK:-}" ] && [ "${SAVI_ORACLE_LEAK}" = "1" ]; then
    ORACLE_FLAG="--oracle-leak"
fi
# Save every generated mp4 by default (bank videos for the figure bank / future
# experiments). Set SAVI_NO_SAVE_VIDEOS=1 for a metrics-only run.
SAVE_FLAG="--save-videos"
if [ "${SAVI_NO_SAVE_VIDEOS:-0}" = "1" ]; then
    SAVE_FLAG=""
fi
# VBench++ (7 dims) on the saved mp4s so SAVi-DNO matches AdaSteer's metric set.
# On by default; needs saved videos. Set SAVI_COMPUTE_VBENCH=0 to skip.
VBENCH_FLAG="--compute-vbench"
if [ "${SAVI_COMPUTE_VBENCH:-1}" != "1" ] || [ -z "${SAVE_FLAG}${SAVE_LIST_FLAG}" ]; then
    VBENCH_FLAG=""
fi
echo "Save videos  : ${SAVE_FLAG:-NO (metrics only)}"
echo "VBench       : ${VBENCH_FLAG:-OFF}"

# LongCat's real inference uses CFG + 50 Euler steps; the SAVi-DNO PVDM recipe
# (no CFG, 10 steps) yields near-garbage (SSIM~0.05) on a LongCat backbone.
# Set SAVI_GENERATION_CFG=1 and SAVI_GENERATION_STEPS=50 for a faithful LongCat
# prediction. These affect only the no-grad prediction passes, not the
# differentiable optimization loop (which stays at SAVI_EULER_STEPS).
GEN_CFG_FLAG=""
if [ "${SAVI_GENERATION_CFG:-0}" = "1" ]; then
    GEN_CFG_FLAG="--generation-cfg"
fi
GEN_STEPS_FLAG=""
if [ -n "${SAVI_GENERATION_STEPS:-}" ]; then
    GEN_STEPS_FLAG="--generation-steps ${SAVI_GENERATION_STEPS}"
fi
echo "Gen CFG      : ${GEN_CFG_FLAG:-OFF}   Gen steps: ${SAVI_GENERATION_STEPS:-<same as euler>}"

# EXP3: TANGO predicted-noise-gaussianity guidance during sampling. Pair with
# SAVI_NO_OPTIMIZE=1 to isolate the guidance (control = no-opt, no TANGO).
TANGO_FLAGS=""
if [ "${SAVI_TANGO_GUIDANCE:-0}" = "1" ]; then
    TANGO_FLAGS="--tango-guidance --tango-lambda ${SAVI_TANGO_LAMBDA:-0.05}"
    TANGO_FLAGS="${TANGO_FLAGS} --tango-sigma-hi ${SAVI_TANGO_SIGMA_HI:-0.9}"
    TANGO_FLAGS="${TANGO_FLAGS} --tango-sigma-lo ${SAVI_TANGO_SIGMA_LO:-0.0}"
    TANGO_FLAGS="${TANGO_FLAGS} --tango-kurtosis ${SAVI_TANGO_KURTOSIS:-0.0}"
fi
echo "TANGO guide  : ${TANGO_FLAGS:-OFF}"

# ============================================================================
# Run SAVi-DNO with LongCat backbone
# ============================================================================
"$PYTHON" comparison_methods/scripts/savi_dno_longcat.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-videos "${MAX_VIDEOS}" \
    --num-inference-steps "${SAVI_EULER_STEPS:-10}" \
    --num-gpus "${NUM_GPUS:-1}" \
    --guidance-scale "${SAVI_GUIDANCE:-4.0}" \
    --lr "${SAVI_LR:-1e-4}" \
    --lam "${SAVI_LAM:-0.0012}" \
    --p "${SAVI_P:-0.7}" \
    --rollout-steps "${SAVI_ROLLOUT_STEPS:-10}" \
    --max-grad-norm "${SAVI_MAX_GRAD_NORM:-1.0}" \
    --num-cond-frames "${SAVI_NUM_COND_FRAMES:-14}" \
    --num-frames "${SAVI_NUM_FRAMES:-28}" \
    --gen-start-frame "${SAVI_GEN_START_FRAME:-48}" \
    --method "${NOISE_OPT_METHOD}" \
    --reg-weight "${NOISE_OPT_REG_WEIGHT:--1}" \
    --seed 42 \
    ${GEN_CFG_FLAG} \
    ${GEN_STEPS_FLAG} \
    ${NO_OPT_FLAG} \
    ${LOSS_FLAG} \
    ${ORACLE_FLAG} \
    ${SAVE_FLAG} \
    ${VBENCH_FLAG} \
    ${GT_CACHE_FLAG} \
    ${SAVE_LIST_FLAG} \
    ${TANGO_FLAGS}

echo ""
echo "=============================================================================="
echo "SAVi-DNO LongCat Complete"
echo "End time: $(date)"
echo "=============================================================================="
```
