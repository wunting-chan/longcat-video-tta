# video-attractors

Where do videos go to die? We study the attractors of autoregressive video
diffusion models (Self Forcing, Rolling Forcing and LongLive on Wan2.1-1.3B)
as stochastic dynamical systems. The target venue is CVPR 2027.

The project was renamed from `longcat-video-tta` on 2026-10-05. The earlier
test-time-adaptation work (LongCat-Video, AdaSteer, LoRA, TinyLoRA, Wan
interventions) stays in this repo as supporting evidence.

Start here:

- `AGENTS.md`: rules and an index for agents.
- `docs/ATTRACTORS.md`: the live study and its results so far.
- `docs/INTERVENTION_ATLAS.md`: every test-time intervention we tried, on one table.
- `attractors/`: code and results of the attractor study.
- `paper/`, `sweep_experiment/reports/paper_draft.md`: the earlier TTA paper
  draft, now source material for the weight-adaptation chapter.
