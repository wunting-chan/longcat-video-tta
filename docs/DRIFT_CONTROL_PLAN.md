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
