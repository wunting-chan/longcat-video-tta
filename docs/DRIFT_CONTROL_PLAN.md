# Drift control: steering Self Forcing away from its shared attractor

Plan fixed 2026-10-06, before any pilot results. It merges our latent-projection design with a
second agent's closed-loop proposal: the matched random baseline, toward-attractor and one-shot
ablations, and the separate anchor-only and avoid-only arms. Code: `attractors/drift.py`
(tests in `test_drift.py`) and the flags on `attractors/long_rollout.py`.

## Model

Generation is a controlled process, `X_{t+1} = F(X_t, c, xi_t, u_t)`. The control `u_t` is
applied to each freshly denoised latent block, after Step 3.2 and before Step 3.3. It therefore
changes both the saved video and the KV cache that later chunks attend to.

- **Control space: the generator's latent space.** In the existing runs, cross-prompt,
  cross-seed latent distance falls from 0.92 to 0.43 over 3 minutes, so the attractor is visible
  there.
- **Evaluation space: DINOv2 on decoded frames.** The controller never optimises the metric it
  is judged by.
- **Prompt-specific reference** `o`: the mean latent of the first 3 blocks (about 2.25 s). Those
  blocks are never corrected.
- **Shared attractor** `c`: the mean latent of the last ~20 s of the 16 existing Self Forcing
  rollouts of prompts 0–7 (`attractor_sf_p0-7.pt`).
- **Test prompts:** MovieGen prompts 8–15, which `c` never saw.

## Arms (Self Forcing, prompts 8–15, seeds 0 and 1, 120 s)

| Arm | What it does | Question it answers |
|---|---|---|
| baseline | nothing | reference |
| repel 1.0 | remove the displacement toward `c` (u = (c − o)/‖c − o‖, only p > 0) | does avoiding the attractor help? |
| repel 0.5 | half strength | dose response |
| random 1.0 | same correction norm as repel 1.0, random direction | is direction what matters, or is it just the perturbation? |
| repel −0.5 | push *toward* `c` | causal test: does collapse speed up? |
| anchor 1.0 | pull per-channel mean and std to the opening's | does plain prompt anchoring do as well? |
| both 1.0 | anchor, then repel | does knowing the attractor add to anchoring? |

One-shot ablation (seed 0, prompts 8–15, 120 s): repel 1.0, repel −1.0 and random 1.0, applied
only for 30–35 s. It tests whether trajectories relax back after a displacement, which would be a
restoring tendency.

Stage 2, only if direction matters in the pilot: candidate-based steering. Generate K = 4
candidate chunks per step and keep the one whose latent is farthest from `c` relative to `o`.
This is the existing always-search with a new score.

## Measurements

Per second, cross-seed, as on slide 13:

- **Prompt separation:** mean 1 − cos of DINOv2 [CLS] vectors over pairs that differ in prompt
  and seed.
- **Prompt recoverability:** nearest-neighbour matching against the other seed, 1-in-8 chance.
- **Distances in both spaces:** to the attractor and to the opening, in latent space and in
  DINOv2 space (the DINOv2 attractor centroid also comes from prompts 0–7).
- **Look and quality:** saturation, brightness, sharpness and optical flow, plus VBench imaging
  quality and temporal flickering if the scorer is available.
- **Intervention size:** mean correction norm per block, from the `drift_log` stored in each
  rollout's JSON.

## Decision rules (written before the results)

- **Direction matters** if repel beats random on prompt separation at 120 s by more than the
  95% bootstrap interval over prompts, with image quality within 1 VBench point of baseline.
- **If random ≈ repel:** the effect is sensitivity, not directional control. The attractor
  geometry is not yet a useful control signal.
- **If anchor ≈ both ≈ repel:** knowing the attractor adds little beyond plain anchoring, which
  weakens the novelty claim.
- **If repel −0.5 collapses faster and repel 1.0 slower:** a causal test that the measured
  geometry is meaningful.
- **One-shot:** if displaced trajectories return toward `c` within ~30 s, that is a restoring
  tendency, and it motivates continuous feedback.
- **Image quality:** an arm that loses more than 1 VBench point of image quality, or adds
  visible flicker, fails regardless of separation. Most earlier editing methods failed this way.

Negative outcomes are reported as results. Strengths are not tuned after seeing them.

## Cost

About 112 continuous rollouts plus 24 one-shot rollouts at about 3.5 A100-minutes per 120 s
rollout: about 8 A100-hours. Features take about 0.5 L40S-hour.

## Revision, 2026-10-07: the method is a memory correction; selection is a baseline

User decision: best-of-K selection is well established (our always-search, CachedSearch,
EvoSearch and others), so it is no longer the proposed method. It remains only as a
compute-heavy baseline.

**Evidence from the one-chunk kicks** (held-out prompts 8–15, seed 0):

- A kick toward the attractor sticks, and more so the bigger the kick. At 4×, the latent shift
  at 60–118 s is +0.93 units [0.40, 1.50]. In DINOv2, distance to the attractor falls by 0.09 and
  prompt identity drops from 87% to 59%.
- Kicks away from the attractor, and random kicks, are undone within one attention window
  (half-life 2.7–4.5 s).
- The drift rate after a kick is unchanged, so a kick shifts the video along a fixed track.
- Baseline drift is a small push at every chunk, concentrated in the first minute. About 56% of
  its latent energy is a global colour cast.

The fast restoring pull acts through the model's memory, which motivates the method.

**Method: attractor-aware memory correction.** At every chunk, remove the component of drift
toward the attractor (latent axis u = (c − o)/‖c − o‖, the part with p > 0) from the block
written to the KV cache only. The displayed block is the model's own output
(`--hook-target memory`). No extra samples and no training.

**Stage A arms** (prompts 8–15 × seeds 0 and 1, 120 s):

1. baseline
2. memory correction, continuous (α = 1)
3. memory correction, early only (2.25–60 s, then off)
4. memory correction with a random direction of matched norm
5. the same correction applied to the displayed output (tests whether memory specifically matters)
6. best-of-2 selection with the attractor score (baseline)
7. best-of-2 selection, random pick (matched-compute control)

**Deferred:** tolerance band, colour-only correction, correction before the last denoising step,
front-loaded schedules, stripe filter. Decision rules and endpoints are as above. Selection
toward the attractor is dropped, because the kick experiment already answers that question.

## Revision, 2026-10-08: reviewer-driven priorities, and memory purification

**Results that set the direction:**

- The linear steering arms are complete. They change which collapsed state a video reaches, not
  whether it collapses: the observed drift axis is a readout of the dynamics, not a control
  vector.
- The memory swap shows that the video's position along the drift is set by the content of its
  21-latent memory window. Re-encoding the opening at 60 s takes it from 1.56 to 0.17 units. From
  clean context the model then drifts again at about 1.0 units per 30 s.

**Priorities, following the cross-platform review:**

1. State transplant: A's current chunk with B's memory, and the reverse.
2. Prompt-switch susceptibility at 10, 30, 60 and 90 s, with identical future noise.
3. A second backbone: LongSANA, with and without its sink.
4. Sink sweep {0, 1, 2, 3, 4, 6, 8} at a fixed 21-latent budget. In Self Forcing the sink already
   sits inside the window.
5. A second representation (CLIP or SigLIP).
6. Interventional attention masking and denoising-step attribution.

No new steering variants.

**Memory purification (time-boxed, after 1–2).** The test is gated on the swap continuations
looking visually clean. When the attracting-regime readout passes a threshold, the memory
chunks are partially re-noised and re-denoised by the model with the clean opening as context.
The result keeps the scene layout but removes the accumulated drift, and the memory holds only
the model's own outputs, never an added vector.

- Comparisons: sink, periodic hard refresh, baseline.
- Cost: about 4–6 A100-h.
- Expected odds: about 30%.
- If it only matches a sink, it is reported as an explanation of why sinks work.

### 2026-10-08 (late): gates passed, mechanism runs launched
- Smoke 19441287–89: transplant (kv, at 10 s) and prompt switch (at 10 s) are bit-identical to the baseline through 10.5 s and diverge afterwards (max |Δ| 1.4–4.9 latent units). Both mechanisms work as intended.
- Memory-swap visual gate: re-encoding the opening into the 21-latent window at 60 s gives a clean, artifact-free return to the opening scene on all 4 checked prompts (9, 11, 13, 15). It then re-collapses within ~30 s. Purified memory goes ahead as planned (time-boxed, behind the main experiments). Frame grids: `MemorySwap_frame_grids_fullres.pdf`.
- Launched SF, held-out prompts 8–15, seed 0, 120 s: transplant kv/latest at 30 s (19441737/39); prompt switch at 10/30/60/90 s (19441741/42/44/46). LL/RF prompt switch waits on a hook check.

### 2026-10-08: free analyses (no generation)
- **End states form a continuum, not a few basins.** DINOv2 end states (last 10 s) of 336 rollouts: silhouette 0.14–0.23 for k = 2–8, rising slowly with k and never clearly peaking. Self Forcing end states sit much closer together (mean cosine distance 0.605) than Rolling Forcing (0.925) or LongLive (0.912). This is consistent with one shared attracting region for SF only.
- **VBench mostly does not track semantic retention once time is held fixed.** Pooled over windows, everything correlates because everything declines with time. Within a window:
  - imaging quality vs prompt identification: rho +0.06 (55 s) and −0.12 (110 s); vs an arm's cross-prompt diversity: −0.32 at 110 s;
  - aesthetic quality is the only dimension that tracks diversity across arms (rho 0.75–0.86, n = 7 arms, low power);
  - subject consistency correlates negatively with diversity (−0.29 to −0.54).
  - This supports reporting VBench alongside semantic metrics, never on its own. `attractors/results/free_round2*.json`.
