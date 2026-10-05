# AdaSteer Paper Draft Packet

**Working title:** AdaSteer: Adaptive Shared Timestep Embedding Efficient Residuals for Test-Time Adaptation of Video Diffusion Transformers

**Status:** Planning / decision-log document. As of May 23, 2026, the **paper prose source of truth lives in LaTeX** at `paper/` (Overleaf-ready; entry point `paper/main.tex`). This file remains the high-level skeleton, claim set, figure plan, batching placeholder, and "Open Slots For Final Data" registry. When draft prose is written or revised, edit the corresponding `paper/sections/*.tex` file, not this Markdown. Final numbers from the corrected 1000-video analysis should replace the provisional result language below (and in the `\todo{}` blocks of the LaTeX sections) before submission.

> **Note on the LaTeX project location.** `paper/` is intentionally git-ignored (`.gitignore` line 36: "Paper (private, do not push)"). The LaTeX source is therefore a local-only artifact unless the user manually syncs it to Overleaf. BibTeX entries are duplicated at `sweep_experiment/reports/refs.bib` (tracked) and `paper/refs.bib` (the copy Overleaf uses); keep them in sync.

## Core Narrative

Large video diffusion transformers are powerful but instance-agnostic at inference time. In video continuation, the observed conditioning frames contain test-specific motion, scene layout, textures, and camera dynamics, yet the frozen model uses the same denoising behavior for every test clip. Test-time adaptation should exploit this signal, but the usual tools are poorly matched to a 13.6B-parameter DiT: full-model tuning is too high-capacity for a single clip, and even LoRA introduces enough free parameters to overfit or fail to move the model in a useful direction.

AdaSteer is the lightweight alternative. It adapts the denoising behavior by learning a small additive residual in the timestep embedding pathway, shared across transformer blocks. The key framing is structured weight tying: a compact trainable residual is shared globally, while the frozen adaLN projections inside each block transform it into block-specific shift, scale, and gate perturbations. This gives per-block specialization without learning per-block weights.

The paper should pitch AdaSteer as a test-time steering method rather than a weight-update method. The method asks: can the model's own conditioning and modulation pathways be used as an adaptation substrate, so that test-time learning only selects a direction of behavior rather than relearning a subspace?

## Abstract Draft

Video diffusion transformers generate high-quality videos, but inference is typically instance-agnostic: the same frozen denoising process is used regardless of the motion and appearance statistics visible in the conditioning frames. Test-time adaptation offers a natural way to specialize generation to each test video, but conventional adaptation mechanisms are mismatched to modern video DiTs. Full-model tuning exposes billions of parameters to a single short clip, while LoRA-style adapters introduce thousands to millions of free parameters and can overfit within a few gradient steps.

We introduce AdaSteer, Adaptive Shared Timestep Embedding Efficient Residuals, an ultra-lightweight test-time adaptation method for video diffusion transformers. AdaSteer learns a compact residual in the timestep embedding pathway and shares it across all transformer blocks. Each block's frozen adaptive layer-normalization projection maps this shared residual into block-specific modulation vectors, yielding a structured weight-tying mechanism that adapts denoising behavior with only a tiny number of trainable parameters and no architectural changes. The residual is optimized only on observed conditioning frames using the standard denoising objective, applied during generation, and discarded afterward.

On LongCat-Video, a 13.6B-parameter video DiT, AdaSteer matches, but does not improve on, no-TTA on standard-horizon Panda-70M and UCF-101 settings at 932–999 videos. At 999 videos, FVD on the standard-horizon Panda setting is unchanged within noise (No-TTA 154.7, AdaSteer 153.4); the long-horizon Panda 1000-video evaluation shows a small distributional regression that we report honestly as an open case rather than a success. These results suggest that pretrained modulation pathways provide a strong substrate for efficient per-video adaptation, while leaving long-horizon coherence as a clear next target.

## Main Contributions

- We formulate test-time adaptation for large video DiTs as timestep-pathway steering rather than direct weight adaptation.
- We introduce AdaSteer, a shared residual applied to the timestep embedding before per-block adaLN projections.
- We identify the structured weight-tying mechanism: one compact residual is shared globally, while frozen per-block projections provide learned de-tying.
- We compare AdaSteer against no-TTA, LoRA, TinyLoRA-style SVD adapters, and full-model adaptation regimes.
- We show that at 932–999 videos AdaSteer leaves FVD and pixel-space metrics unchanged within noise across datasets and horizons (Panda standard 154.7 → 153.4; UCF 85.7 → 88.3; source: `paper_tables/2026-06-08_headline_1000v.md`).
- We document negative results and failure modes: LoRA often matches or trails baseline, aggressive adaptation overfits, and several auxiliary tricks do not improve the Pareto frontier.

## Method Section Skeleton

### Problem Setup

We study video continuation with a pretrained video diffusion transformer. Given observed frames, the model generates future frames at 480p using classifier-free guidance and a fixed denoising schedule. Test-time adaptation is allowed to optimize only on the observed conditioning window; no future frames are used during adaptation.

The primary backbone is LongCat-Video, a 13.6B-parameter DiT with 48 single-stream transformer blocks and hidden dimension 4096. The timestep embedding dimension used by the adaLN pathway is 512. Each block maps the timestep representation into shift, scale, and gate vectors for self-attention and MLP modulation.

### AdaSteer Formulation

Let the timestep embedding be `t in R^512`. AdaSteer learns an additive residual `delta` and uses:

```tex
t' = t + \delta
```

Each transformer block then applies its frozen adaLN projection:

```tex
t' \xrightarrow{\mathrm{adaLN}_i}
[\gamma_i^{msa}, \beta_i^{msa}, \alpha_i^{msa},
 \gamma_i^{mlp}, \beta_i^{mlp}, \alpha_i^{mlp}]
\in R^{6 \times 4096}.
```

The same residual is shared across all blocks, but because each block has its own frozen adaLN projection, the induced modulation is block-specific. This is the central mechanism and should be explained visually.

### Optimization

AdaSteer is optimized with the standard denoising or flow-matching objective on the conditioning frames. The adaptation loop samples noise levels, corrupts latent encodings of the observed frames, predicts the denoising target, and updates only the AdaSteer residual. The adapted residual is applied during future-frame generation and discarded after the video is generated.

The default strong configuration from the current logs is 10 steps with learning rate 5e-3. Earlier sweeps show that the useful region is small: moderate learning rates and 5-10 steps preserve PSNR while improving FVD, while higher learning rates or longer adaptation can overfit.

### TinyLoRA / SVD Variant

The paper can include TinyLoRA as a related adaptation variant or ablation rather than the main method, depending on final results. The useful formulation is:

```tex
y = Wx + \frac{\alpha}{r} U_r \left(v \odot (V_r^T x)\right).
```

For LongCat attention targets with 48 blocks and two modules per block, rank-2 untied TinyLoRA trains 192 scalars. With full tying across blocks, this can reduce to 4 scalars. This supports the broader argument that highly constrained adaptation subspaces are better matched to single-video TTA than free LoRA factors.

## Experimental Story

### Primary Claim

The cleanest current claim is not large PSNR improvement. The honest claim is narrower: AdaSteer does not measurably change distributional video quality on the standard 28-frame Panda setting at 999 videos (FVD 154.7 → 153.4, within noise), and long-horizon generation remains unresolved. Per-frame metrics are usually flat or slightly positive; full-scale long-context Panda improves PSNR/SSIM/LPIPS/FID slightly but does not improve global FVD.

### Sample-Size Policy (locked May 23, 2026)

The paper reports **only 1000-video evaluation numbers** in tables and headline claims. Smaller subsets are distributionally biased and their absolute FVD/FID values are not reliable as single-number summaries, so they are excluded from the main paper. The 200-video deterministic subsets are described as the discovery/sweep methodology (used to select hyperparameter ranges and strong configs per method); 50-video runs and 100-video ablations are no longer reported in the paper. The 1000-video evaluations are drawn from pools of 1000 videos; in practice one video may be corrupted on disk, but this is part of the evaluation process and not surfaced in the paper text.

### Current Results To Use (1000-video only)

From `EXPERIMENT_RESULTS.md`, the paper-facing results are:

- Standard Panda-70M, 28 frames, 999 videos: No-TTA FVD 154.7; AdaSteer FVD 153.4 (−0.8%, within noise). PSNR and SSIM are essentially flat. Source: `paper_tables/2026-06-08_headline_1000v.md`. Caveat: these adapted runs used the pre-14-July split that held out 25% of target latents (2 of 8); see `docs/INTERVENTION_ATLAS.md` and the hold-out audit. The earlier 150.09 → 142.32 (5.2%) figure must not be reused.
- Long-context Panda-70M, 93 frames, 1000 videos: No-TTA FVD 278.7; AdaSteer FVD 284.1. AdaSteer improves PSNR (12.769 -> 12.787), SSIM (0.4744 -> 0.4762), LPIPS (0.5469 -> 0.5436), and FID (29.9 -> 29.5), but worsens global FVD by +5.4. LoRA also worsens FVD; TinyLoRA is essentially tied with No-TTA. Treated as an honest open case rather than a success in the paper.
- Long-context UCF-101, 61 frames, 1000 videos: **pending 1000-video validation run**; main-results table row is a `\todo{}` until completion. The earlier 50-video numbers are not reported.

Earlier 50-video and 100-video runs are not paper-facing. They remain in `experiment_metrics_log.md` as the audit trail for discovery decisions, but they do not appear in the abstract, main-results table, or ablations table.

### Next Discovery Experiments

Before adding new method variants, the next experiments use deterministic 200-video subsets from the existing 1000-video pools to select hyperparameter ranges. These 200-video runs are paper-facing only as methodology (Section "Discovery and Sweep Methodology"); their numbers do not appear in result tables. The first tuning pass stays in the standard 28-frame setting, where AdaSteer has the strongest current 1000-video evidence and where compute is cheaper than long-context Panda. Long-horizon experiments follow only after the short-horizon sweep finds configs that improve FVD without pointwise metric regression, and any reported numbers are re-run at the 1000-video scale before being published.

Gating and horizon-aware objectives remain planned method extensions, but their implementation should be discussed after the initial 200-video subset selection and the submission setup is in place.

### Important Caveats

Do not reuse the old February claim of +7.6 dB PSNR. That was caused by comparing a pre-fix no-TTA baseline against post-fix TTA runs. The corrected old improvement was about +0.5 dB on the old `panda_100_480p` subset, and the current harder `panda_1000_480p` setting shows near-zero short-horizon PSNR gain and no FVD gain at scale.

Per the sample-size policy above, the paper does not mix dataset/horizon/sample-size combinations; everything in the main paper is 1000-video. The earlier `panda_100_480p`, 100-video, and 50-video runs are kept in the experiment logs but are not surfaced in paper text.

## Figure Plan

1. Method diagram: show timestep embedding plus shared residual, then fan out through frozen per-block adaLN projections into block-specific modulation.
2. Main results chart: standard Panda 1000-video FVD null plus long-context Panda 1000-video open case.
3. Runtime/parameter chart: No-TTA, LoRA, AdaSteer, and TinyLoRA if final results justify it.
4. Qualitative filmstrip: GT, No-TTA, AdaSteer, with selected examples that show visible temporal or structural improvement.
5. Ablation chart: bare AdaSteer versus early stopping, augmentation, CLIP gating, gradient accumulation.
6. Failure-mode chart or appendix figure: LoRA rank/LR sensitivity and overfitting.
7. Batching chart: H200 train seconds/video and peak memory versus independent TTA batch size, with AdaSteer batched and LoRA/TinyLoRA serial baselines.

## Batching Experiment Placeholder

Two batching experiments are planned for the paper.

**Retrieval-augmented batch-level TTA** measures quality effects from training one shared update on an eval video plus retrieved neighbours. This uses paper-aligned configs for standard Panda, long-context Panda, and long-context UCF, sweeping `K={1,5,10}` for AdaSteer and LoRA. This is a regularization/quality experiment, not a throughput claim, because the current shared-batch implementation cycles one video per optimizer step.

**Batched independent TTA throughput** measures the deployment claim. AdaSteer is benchmarked with independent per-video residuals in a single batched forward/backward pass, using `delta` with shape `[B,512]`. LoRA and TinyLoRA are first measured as serial independent-adapter baselines through the same benchmark harness. The target result table is:

| Setting | Method | Requested B | Max OK B | Train sec/video | Peak H200 GB | Notes |
|---------|--------|-------------|----------|-----------------|--------------|-------|
| Panda 28f | AdaSteer batched | TBD | TBD | TBD | TBD | true independent deltas |
| Panda 28f | LoRA serial | 1 | 1 | TBD | TBD | independent adapter per video |
| Panda 28f | TinyLoRA serial | 1 | 1 | TBD | TBD | independent v vectors per video |
| Panda 93f | AdaSteer batched | TBD | TBD | TBD | TBD | long horizon |
| UCF 61f | AdaSteer batched | TBD | TBD | TBD | TBD | long horizon |

## Paper Outline

1. Introduction
   - Motivation: inference is instance-agnostic, but conditioning frames contain useful instance-specific signal.
   - Challenge: model size versus tiny adaptation signal.
   - Idea: steer the denoising pathway through shared timestep residuals.
   - Contributions and summary results.

2. Related Work
   - Full draft lives in `sweep_experiment/reports/related_works.md` (May 23, 2026). Uses a 6-subsection narrative-aligned organization: video DiTs and long-horizon continuation, test-time adaptation in vision, adaptation and personalization of pretrained diffusion models, parameter-efficient fine-tuning, modulation pathways and adaptive normalization, and evaluation of video generation.
   - Three organizational proposals (A: method-axis, B: problem-axis, C: hybrid narrative-aligned) are recorded at the top of `related_works.md`; the draft uses Proposal C pending sign-off.

3. Method
   - Problem setup and no-leakage adaptation objective.
   - AdaSteer formulation.
   - Structured weight tying through frozen adaLN de-tying.
   - Optimization and implementation details.
   - Optional SVD/TinyLoRA variant.

4. Experiments
   - Datasets and horizons.
   - Baselines: No-TTA, LoRA, full-model, TinyLoRA/SAVi-DNO if included.
   - Metrics: FVD/FID for distributional quality, PSNR/SSIM/LPIPS for per-frame fidelity.
   - Main results.
   - Ablations and sensitivity.
   - Qualitative analysis.

5. Discussion
   - Why neither FVD nor short-horizon PSNR moves at scale.
   - Why LoRA is poorly matched to single-video TTA on a 13.6B DiT.
   - Compute tradeoffs and deployment implications.
   - Limitations: metric variance, sample counts, dependency on conditioning quality, no future supervision.

6. Conclusion
   - AdaSteer shows that frozen modulation pathways are a useful substrate for efficient video TTA.
   - The broader lesson is to adapt behavior through pretrained conditioning interfaces rather than adding large free adapter subspaces.

## Open Slots For Final Data

- Final decision on whether long-context Panda is a negative result, appendix result, or motivation for a new method variant.
- Final LoRA R8 and TinyLoRA framing on the same exact split and horizon.
- SAVi-DNO corrected baseline status and whether it belongs in the main paper or appendix.
- Statistical uncertainty or chunk-level variance for FVD/FID.
- Final qualitative examples and captions.
- Exact parameter count for the AdaSteer configuration used in the final main table.
- Final runtime breakdown with consistent hardware and batching assumptions.
- Retrieval-batch results for `K={1,5,10}` on paper-aligned standard and long-context settings.
- Independent TTA throughput results: max AdaSteer batch size on one H200, seconds/video, and peak memory versus serial LoRA/TinyLoRA.

