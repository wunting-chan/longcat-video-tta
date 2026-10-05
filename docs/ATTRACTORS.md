# Where do videos go to die? Attractors of autoregressive video generators

Status note, 2026-10-05. Exploratory phase; one seed pair per setting, 8 prompts.
Numbers below are the current evidence, not paper-final.

## Question

Long autoregressive video rollouts degrade. Prior work catalogues the symptoms
(drift, over-saturation, motion loss, freezing). We treat a streaming generator
as a stochastic dynamical system and ask *where* rollouts go, *how fast*, and
*which design knobs* change the destination:

- attractor type: one global end state, prompt-anchored states, or none;
- prompt half-life and relaxation time toward the end state;
- sensitivity to tiny perturbations (twins) versus semantic divergence;
- bifurcations under one design knob (attention sink, local window).

Novelty check (2026-10-03): prior work offers symptom taxonomies (memory survey
2609.28466, erank collapse 2607.27036, FreqForcing, Steady-Forcing). None uses a
dynamical-systems analysis.

## Models and protocol

- Self Forcing (SF), Rolling Forcing (RF) and LongLive (LL), all on Wan2.1-1.3B.
  Official pipelines are imported read-only, with an SDPA fallback in place of
  flash-attn.
- **Phase −1:** the existing cite-128 30 s V2V rollouts plus a real-future
  control. Features are per-frame DINOv2 [CLS] and low-level observables:
  brightness, contrast, saturation, colorfulness, sharpness, spectral slope,
  frame-difference energy and optical flow.
- **Phase 0:** 48 T2V rollouts of 180 s: 3 models × 8 MovieGen prompts ×
  seeds {0, 1}, with common noise per seed.
- **Twins:** the same run, with the noise of one block perturbed from a
  separate generator so the global RNG stream is identical. Unperturbed reruns
  are bit-deterministic.
- **Bifurcation probe:** SF weights with sink ∈ {0, 3} × local window ∈ {21, 12}.
- GPU used so far: about 8.1 H200-hours.

## Findings so far

| Finding | Evidence |
|---|---|
| Generators contract and drift to model-specific end states. Real video does not. | Phase −1: DINOv2 ensemble spread for SF falls 0.95 → 0.85; real video stays flat. SF gets darker, saturation rises 0.31 → 0.58, sharpness rises 2.4×. RF gets brighter and more contrasty, plateauing at about 20 s. |
| Fast motion collapse, slow appearance drift. | Motion collapses in about 2 s; appearance drifts over tens of seconds. |
| **SF has a global attractor.** | Every prompt ends in purple-blue streaks. Distance between different prompts falls 0.97 → 0.66. Prompt retrieval falls from 100% to 49%. This is not synchronization by common noise. |
| RF is prompt-anchored and degrades slowly (glare, whitening). LL stays stationary and alive. No run freezes. | Phase 0, 3 min rollouts. |
| SF erases the prompt completely. | Prompt half-life 58 s. The asymptote (0.57) equals the noise floor of a same-prompt pair. |
| SF relaxation is exponential. | R² 0.91, τ ≈ 360 s (`results/long/relax.json`). |
| Small perturbations grow at a model-specific rate, but only SF diverges semantically. | Growth rate does not depend on perturbation size: SF 0.31/s, RF 0.22/s, LL 0.41/s. LL is the most sensitive yet the most stable (`twins*.json`). |
| **One knob flips the attractor type.** | Sink = 3 on SF weights switches the global attractor to prompt-anchored. Window 12 alone changes how the collapse looks (`bif.json`, `bif_strip.png`). |
| Rollouts carry recurrent structure. | Recurrence quantification beats phase-randomized surrogates in 100% of runs. |
| **Negative:** no early warning. | Critical-slowing-down indicators (rising variance and autocorrelation) do not precede collapse. Collapse is a slow ramp (10% → 90% over about 90 s), not a tipping point (`deep.json`, `deep2.json`). |
| Search slows the drift. | Best-of-N search delays the slide (p ≈ 1e-6) but does not stop it. |

## Open next steps (need user approval for GPU)

1. Scale-up: 32 prompts × 2 seeds for the three-model comparison, twins and
   the sink/window sweep, plus sink ∈ {1, 2} and RF without its sink. About
   8–10 H200-hours.
2. Second-seed replication on the 128-clip set.
3. Paper outline: attractor-led, with the weight-adaptation null and the
   intervention atlas (`docs/INTERVENTION_ATLAS.md`) as supporting chapters.

## Files

- `attractors/long_rollout.py`: 180 s rollouts (`--local-attn`, `--sink`,
  `--perturb-block`, `--perturb-eps`). `run_long.sbatch` submits them.
- `attractors/extract_observables.py`, `extract_long.py`: per-frame features.
  `--real` runs the real-video control.
- `attractors/analyze*.py`, `deep*.py`, `relax.py`, `twins*.py`, `bif*.py`,
  `portrait.py`: analyses.
- `attractors/results/phase_m1/`, `attractors/results/long/`: result JSON and
  figures.
- Cluster: `/scratch/wc3013/video-attractors-runs` holds the rollouts (`long/`),
  features (`lobs/`, `obs/`) and slurm logs. The scripts still use the old path
  `/scratch/wc3013/ai-attractors`, which is a symlink to it.
