# Multi-model attractor study: plan (2026-10-08)

Status: **plan only. No GPU spend until Wun Ting Chan approves Phase A.**

## Question

Do current long-horizon video generators, including recent methods designed to correct
long-horizon drift, all fall into attracting regions? If they do:
- Do they collapse in different ways?
- Where does each attracting region sit relative to the distribution of real video (the "origin")?

The current results come from three models on one backbone (Wan2.1-1.3B). Only Self Forcing shows a shared end region:
- mean cross-prompt end-state distance: Self Forcing 0.61, Rolling Forcing 0.93, LongLive 0.91;
- end-state clustering gives a continuum, not discrete basins.

This plan tests whether that picture generalises across distillation recipes, model scale, generation paradigm and backbone.

## Candidates (verified 2026-10-08; U = unverified)

Selection rule:
- open weights and runnable inference code;
- autoregressive or streaming generation;
- can be extended to 180 s or more with the released code;
- a documented or likely failure beyond the trained horizon;
- base models mixed, so the study is not only Wan2.1-1.3B.

| Tier | Method | Base | Paradigm | Why it is in |
|---|---|---|---|---|
| have | Self Forcing (2506.08009) | Wan2.1-1.3B | chunk AR + KV cache, DMD | reference attractor |
| have | Rolling Forcing (2509.25161) | Wan2.1-1.3B | rolling-window joint denoising + sink | corrector, less drift |
| have | LongLive (2509.22622) | Wan2.1-1.3B | frame-level AR, frame sink, KV recache | corrector, less drift |
| 0 | Infinity-RoPE (2511.20649), training-free | Self Forcing ckpt | block-relativistic RoPE, KV flush | same weights; separates position extrapolation from the attractor |
| 0 | Deep Forcing (2512.05081), training-free | Self Forcing ckpt | deep sink + KV compression | same weights; tests sink and compression correctors |
| 1 | CausVid (2412.07772) | Wan2.1-1.3B | chunk AR, DMD, trained without self-rollout | separates the distillation recipe from self forcing |
| 1 | SkyReels-V2-DF 1.3B (2504.13074) | Wan2.1-1.3B | diffusion forcing | different paradigm, same base (custom licence) |
| 1 | Helios-Distilled (2603.04379) | Wan2.1-14B | 33-frame chunk AR, drift simulated in training | 10× scale; shown only to ~60 s |
| 1 | Krea Realtime 14B | Wan2.1-14B | Self Forcing recipe at 14B | 1.3B vs 14B with the recipe held fixed |
| 2 | MAGI-1 4.5B-distill (2505.13211) | own architecture | 24-frame chunk AR, monotone noise | native AR model, not Wan |
| 2 | LongCat-Video (2510.22200 U) | own 13.6B | continuation-pretrained, not distilled | claims no colour drift over minutes; testable |
| 2 | FramePack-F1 (2504.12626) | HunyuanVideo-13B | next-frame-section packing, I2V | cleanest non-Wan backbone; freezing is the likely failure (U) |
| 2 | SANA-Video / LongSANA (2509.24695) | own linear DiT | block AR, constant-memory state | linear attention instead of a KV cache; already planned |

Excluded for now:
- no code: LoL, StreamDiT;
- code or weights not released or unverified: FreqForcing, Steady-Forcing;
- no text input: DFoT;
- needs action inputs: Matrix-Game, Yume;
- still Wan-1.3B with nothing new: Reward Forcing, Self-Forcing++ (weights U).

Optional later: SVI (Wan2.1-I2V-14B, clip chaining with error recycling, claims 20 min).

Caveat: Wan-based models hit a RoPE limit at ~1024 latents (~4 min 15 s). The protocol stays at 180 s, below it.

## Protocol (identical for every model)

**Prompts and seeds.**
- 16 MovieGenBench prompts: 0–7 for estimation, 8–15 held out.
- 2 seeds each, with **independent noise per prompt and seed**. This fixes the shared-noise limitation of the existing runs.
- I2V-only models (FramePack-F1): start from the first frame of Self Forcing's video for the same prompt and seed, so the starting content is matched.

**Length and resolution.**
- 180 s per video, each model at its native resolution and frame rate, resampled to 16 fps for analysis.

**Features.**
- DINOv2 ViT-B/14 [CLS] (224 short side, centre crop, averaged per second) as now.
- CLIP ViT-L/14 as the second representation.
- Low-level features: HSV histogram, frequency bands, frame difference.

**Dynamical measures** (same definitions as phase 0):
- prompt half-life;
- relaxation time constant: velocity vs position along the model's own drift axis;
- finite-time divergence of twin runs (8 prompts × 1 seed, small noise perturbation at block 10, where the code allows);
- RQA determinism gap.

**Collapse-mode taxonomy.** Automatic detectors, each with a threshold set on the existing Self Forcing, Rolling Forcing and LongLive runs, then checked against frame grids:
1. colour or saturation drift (HSV Bhattacharyya distance from the opening);
2. glare or over-exposure (fraction of pixels near clipping);
3. stripe or texture attractor (energy in the horizontal-stripe and high-frequency bands);
4. motion freeze (median frame difference, and optical flow on a subset);
5. prompt erasure (CLIP text–frame score, and DINOv2 prompt identification);
6. cycling or sink resets (recurrence of the opening state; the LoL "sink-collapse").

## The real-video "origin"

**Reference set.**
- 2,000 real clips of 10 s, from a held-out slice of an open, licence-compatible video dataset (OpenVid-1M or Panda-70M test split; final choice after checking access).
- Encoded with the same per-second DINOv2 and CLIP pipeline.
- A second, matched reference is each model's own first 10 s, which shows what the model produces before drift.

**Per model, at the end state (last 20 s):**
- distance from the real-video centroid;
- Fréchet distance and kernel MMD (DINOv2) to the real reference, with bootstrap intervals;
- ensemble spread: trace of the covariance and the Vendi score;
- displacement from the model's own opening: length and direction;
- cross-model end-state distances: do different models fall into the **same** region?

**Figure 1 (FIGURE_STYLE.md, scatter style).**
- One 2-D projection, fitted on the real reference plus all openings only (end states never shape the axes). Candidates: PCA, or real-vs-all-end-states LDA plus PCA.
- Real-video density as a contour, one marker per model at its end-state centroid with a 1-SD ellipse, and an arrow from the model's opening centroid.
- A companion panel plots distance from real against time for every model.

## Phases and budget (A100-hours unless stated; estimates before any throughput probe)

**A. Setup and throughput probes. Requested now.**
- One environment per code base under `/scratch/wc3013/conda-envs/` (never `longcat`).
- Weights into `/scratch/wc3013` HF cache: ~250 GB in total, 14B models included.
- Per model: one 30-s video, a check that rollouts extend to 180 s, and seconds of generation per second of video.
- The two training-free Self Forcing correctors are hook-level ports; I'll check them against their papers' 60-s examples.
- Cost: ~0.5 GPU-h × 10 models ≈ **5–8 GPU-h**, plus CPU time for the real-video reference.

**B. Tier 0 + Tier 1 rollouts.**
- 6 models × 32 videos × 180 s, plus twins.
- 1.3B models ≈ 5 h each; 14B real-time models ≈ 6–10 H200-h each.
- Total ≈ **40–60 GPU-h**.

**C. Tier 2 rollouts.**
- MAGI-1 4.5B and LongSANA: ≈ 10–15 h each.
- FramePack-F1 and LongCat-Video (not distilled, many steps) could be **50–150 H200-h each**. The Phase A probe sets this.
- If one is too slow, run it on the 8 held-out prompts × 1 seed only and say so.

**D. Analysis.**
- Zero GPU, apart from feature extraction (~1 GPU-h per model).

Approval points:
- A now;
- B after the A probes report real throughput;
- C per model, once its cost is known.

Everything respects the 2-way H200 cap.

## What would change the story

- **Every model collapses, even the correctors:** the attractor is a general property of streaming video generation. Correctors move it, delay it or reshape it, and the map shows where.
- **Correctors don't collapse, but CausVid or Krea does:** the attractor belongs to the DMD / self-rollout recipe, not to autoregression.
- **Different models fall into the same region:** a shared bias of the data or the base model, not just of the recipe. The cross-model distance measures this directly.
- **Non-Wan models collapse differently** (freezing for FramePack, prompt erasure without colour drift): the mode depends on the backbone, and the taxonomy is the result.

## Revision 2026-10-10: Phase A approved; model set chosen by baseline frequency

Wun Ting Chan approved Phase A and asked for the models other papers use as baselines, kept cheap
(not LongCat-Video). I counted the baselines in the long-video tables (≥30 s) of 16 recent papers:
LongLive, Rolling Forcing, Self-Forcing++, Infinity-RoPE, Reward Forcing, Deep Forcing, LoL,
Relax Forcing, MemRoPE, Helios, PackForcing, Steady-Forcing, FreqForcing, Recency Forcing,
ID-Forcing and BlockVid.

| Model | Long tables it appears in | Status |
|---|---|---|
| Self Forcing | 16 | have |
| LongLive | 12 | have |
| CausVid | 10 | **Phase A** |
| Rolling Forcing | 9 | have |
| SkyReels-V2-DF | 8 | **Phase A** (1.3B) |
| Deep Forcing | 7 | **Phase A** (training-free on the Self Forcing checkpoint) |
| MAGI-1 | 6 | **Phase A** (4.5B-distill; the only non-Wan T2V chunk-AR model that fits one GPU) |
| Infinity-RoPE | 5 | **Phase A** (training-free on the Self Forcing checkpoint) |
| NOVA | 4 | **Phase A** (0.6B, non-Wan, non-quantized AR) |
| Causal Forcing | 3 | optional later (cheap, Wan-1.3B) |
| Reward Forcing, Infinite Forcing, FramePack, Self-Forcing++ | 2 | not now |
| Pyramid Flow, Krea 14B, LongCat-Video, InfinityStar, SANA-Video | 1 | dropped |

Dropped from the 2026-10-08 shortlist:
- Helios-14B, Krea-14B and LongCat-Video: each is used once and costs 10× more.
- FramePack: image-to-video only.
- LongSANA: used once.
- Pyramid Flow: capped at 10 s by its code, and fails at 50 s in Self-Forcing++.
- Self-Forcing++: no weights or code released.

The set now covers four things:
- the forcing family on Wan-1.3B: Self Forcing, CausVid, Rolling Forcing, LongLive;
- the two most-cited training-free correctors: Deep Forcing, Infinity-RoPE;
- diffusion forcing: SkyReels-V2-DF;
- two non-Wan backbones: MAGI-1, NOVA.

**Phase A mechanics** (`attractors/multimodel/`):
- **Environments.**
  - `va_wan`: torch 2.5.1, cu124, prebuilt flash-attn 2.7.4. Used for CausVid, Deep Forcing, Infinity-RoPE, SkyReels-V2 and NOVA.
  - `va_magi`: torch 2.4.0 with flash-attn 2.6.3. MagiAttention is not installed; the README marks it optional off Hopper.
- **Probe.** `probe.sbatch` runs each repo's own inference script, unmodified, on held-out prompt 8 with seed 0, on one A100. It records wall time, peak GPU memory and the actual video length.
- **Exceptions to "unmodified":**
  - CausVid's script does not seed, so `seeded_run.py` seeds it.
  - MAGI-1 runs at 480×832 to match the Wan models; its config default is 720×720.
