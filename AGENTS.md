# AGENTS.md — persistent index & workflow rules

**Purpose:** This file is the canonical entry point for any AI agent (Claude,
Cursor, etc.) picking up work on this project. Read it FIRST before any
substantive task. Update it whenever a new persistent artifact is created.

**Project renamed 2026-10-05: `video-attractors`** (formerly
`longcat-video-tta`). **Live direction (user decision 2026-10-05):** define
how autoregressive video diffusion models decay toward attractors. Treat
Self Forcing, Rolling Forcing and LongLive (Wan2.1-1.3B) as stochastic
dynamical systems: attractor type, prompt half-life, relaxation time, twin
divergence, and bifurcations under design knobs (sink, window). Target:
CVPR 2027, deadline 2026-11-16. Read `docs/ATTRACTORS.md` first.

- The Wan / Self-Forcing GPU line is **reopened for this study only**.
  Ask the user before spending beyond an approved H200 budget.
- The test-time-adaptation record (§3) is now supporting evidence: the
  weight-adaptation null at scale and `docs/INTERVENTION_ATLAS.md`. Do not
  revive TTA as the title method.
- Code: `attractors/`. Cluster runs: `/scratch/wc3013/video-attractors-runs`
  (old path `/scratch/wc3013/ai-attractors` is a symlink to it).
- Artificial Individuality lives in its own repo and cluster tree
  (`/scratch/wc3013/artificial-individuality`); §7 below is historical.

---

## 1. Persistent files & where to find them

| What | Path | Notes |
|---|---|---|
| **This index file** | `AGENTS.md` | Updated as artifacts are added |
| **Attractor study (LIVE)** | `docs/ATTRACTORS.md`, `attractors/` | Results note, code, result JSON/figures. Cluster runs in `/scratch/wc3013/video-attractors-runs`. |
| **Intervention atlas** | `docs/INTERVENTION_ATLAS.md` | 33 test-time intervention families, edit vs select, on one table. Supporting chapter. |
| **Artificial Individuality (LIVE)** | `/scratch/wc3013/artificial-individuality` | DINO / ImageNet-100 history experiment. Protocol is §7 of this file. 1× H200, ≤48 h. First milestone only until Exp. 1–2 gates pass. |
| **Cluster & sbatch onboarding guide** | `docs/CLUSTER_SBATCH_GUIDE.md` | Self-contained guide for a brand-new agent: cluster quirks (account flag, /scratch, conda/PYTHONHOME), how to write sbatch jobs, and ready-to-use fine-tune + long-horizon continuation recipes. |
| **Master experiment index** | `sweep_experiment/reports/INDEX.md` | **Single source of truth** for what experiments exist + cluster paths. Read this first when picking up work. |
| **Analysis log (decisions/findings)** | `sweep_experiment/reports/ANALYSIS_LOG.md` | Append-only log of paper-relevant findings and decisions. NEVER edit past entries. |
| **Paper-ready tables** | `sweep_experiment/reports/paper_tables/YYYY-MM-DD_<name>.md` | One Markdown file per table set, dated. Reproducible via `scripts/build_paper_tables.py`. |
| **Pseudo-future Search note** | `sweep_experiment/reports/paper_tables/2026-08-25_pseudo_future_search.md` | **DROPPED as paper title 2026-09-04.** Code `sf_pseudo` stays for the cite-128 ablation. |
| **Drop Pseudo + next territories** | `sweep_experiment/reports/paper_tables/2026-09-04_drop_pseudo_next_territories.md` | Outcome atlas. Territories A (new student) / B (analysis) / C (new control). No GPU until the user picks. |
| **Failure modes (plain language)** | `sweep_experiment/reports/paper_tables/2026-09-04_failure_modes_plain.md` | What each failed family did to the videos. Read this before quoting IQ / twitch / identity. |
| **Method hypotheses + motivation** | `sweep_experiment/reports/paper_tables/2026-09-04_method_hypotheses_motivation.md` | Why each live idea follows from the appendix. Four hypotheses. Failed runs stay appendix. |
| **Train/eval same-metric literature** | `sweep_experiment/reports/paper_tables/2026-09-05_train_eval_same_metric.md` | Related-family motion rewards + VBench eval are accepted. Official RAFT-in-the-loss is DOLLAR / not a title. |
| **Go-with-the-Flow read** | `sweep_experiment/reports/paper_tables/2026-09-05_go_with_the_flow.md` | Image warped-noise is FT-free. Video needs paired FT. A mid-step circular roll is not their method. |
| **Mid-step warp holes** | `sweep_experiment/reports/paper_tables/2026-09-05_midstep_warp_holes.md` | Gaussian wrap is a no-op. Late extra has no energy. Do not roll pred against the KV. |
| **Mid-step warp fixes** | `sweep_experiment/reports/paper_tables/2026-09-05_midstep_warp_fixes.md` | Persist HIWYN on every extra; carry field across strips; leftover mean flow; γ ≈ 0.5. No GPU. |
| **Caption nwarp N=8 spec** | `sweep_experiment/reports/paper_tables/2026-09-06_wan_v2v_caption_nwarp_spec.md` | Extra-only HIWYN extras. `sf_nwarp` + `sf_nwarp_live`. **DONE / NO.** Do not remake cite-128. |
| **Caption nwarp harvest** | `sweep_experiment/reports/paper_tables/2026-09-06_wan_v2v_caption_nwarp_harvest.md` | 17028867–876 COMPLETED. Always IQ 49.18 Dyn 0/8. Live IQ 54.42 Dyn 2/8. Both **NO**. |
| **Why nwarp IQ died** | `sweep_experiment/reports/paper_tables/2026-09-06_nwarp_vs_gwf_why_iq_died.md` | Extra-only locked a noise stencil. Not GwF. dy=0 still IQ 45. |
| **GwF / SAVi run?** | `sweep_experiment/reports/paper_tables/2026-09-06_gwf_savi_should_we_run.md` | Retrain GwF **no**. SAVi = DNO increment; leak-easy writeup. No GPU. |
| **Caption pwarp N=8 spec** | `sweep_experiment/reports/paper_tables/2026-09-06_wan_v2v_caption_pwarp_spec.md` | Slide `pred` after pass 1. `sf_pwarp` + `sf_pwarp_live`. **DONE / NO.** Do not remake cite-128. |
| **Caption pwarp harvest** | `sweep_experiment/reports/paper_tables/2026-09-06_wan_v2v_caption_pwarp_harvest.md` | 17058386–393 COMPLETED. Always IQ 66.81 Dyn 3/8. Live IQ 66.81 Dyn 3/8. Both **NO**. Extra Dyn = 0007 twitch. |
| **Pwarp eye-inspect pack** | `sweep_experiment/reports/paper_tables/2026-09-06_pwarp_eye_inspect.md` | Matched SF vs slide mp4s. `export_pwarp_examples.py` then scp. Watch 0007 first. |
| **Pwarp eye notes** | `sweep_experiment/reports/paper_tables/2026-09-06_pwarp_eye_notes.md` | 0007/0004 host flicker, slide worse. 0002 rewrites a still room. 0006 sailing: quality ok, no extra motion. |
| **Prompt rewrite literature** | `sweep_experiment/reports/paper_tables/2026-09-06_prompt_rewrite_lit.md` | Movie Gen / Hunyuan / CogVideoX / Wan Qwen / MovieGen-128. Rewrite = long-text match, not a flow. 0006 is zoom. |
| **Three eval tracks** | `sweep_experiment/reports/paper_tables/2026-09-06_three_eval_tracks.md` | A pan-filter V2V. B Wan-extend V2V **DONE / NO**. C MovieGen T2V + first-chunk warp. Do not mix. |
| **Track C + first-chunk warp** | `sweep_experiment/reports/paper_tables/2026-09-07_t2v_moviegen_warp_spec.md` | MovieGen T2V 30 s. Chunk-0 flow → nwarp/pwarp. Smoke **HARVESTED 17121785–792**. Do not letter n=2. Do not launch 128. |
| **Track C smoke harvest** | `sweep_experiment/reports/paper_tables/2026-09-07_t2v_moviegen_warp_smoke.md` | Protocol PASS. notta IQ 70.93 Dyn 1/2. nwarp IQ **47.60**. pwarp IQ 69.35. Live==always. |
| **Noise-method baselines** | `sweep_experiment/reports/paper_tables/2026-09-07_noise_method_baselines.md` | Baseline = attached host. SF only if the machine is SF. GwF cites CogVideoX. FIFO cites VideoCrafter2. |
| **Clean host split** | `sweep_experiment/reports/paper_tables/2026-09-07_clean_host_split.md` | Portable ideas on official Wan teacher. Cite `wan_notta`. Forcing-only stay on SF/RF tables. Smoke leftover + MovieGen n=2. |
| **Wan-teacher smoke harvest** | `sweep_experiment/reports/paper_tables/2026-09-07_wan_teacher_smoke_harvest.md` | 17135846–861 COMPLETED 0:0. Protocol PASS. nwarp IQ **54.25 / 51.60** **NO**. pwarp ≈ notta, no Dyn. Do not letter n=2. Do not launch 128. |
| **Pwarp amplify?** | `sweep_experiment/reports/paper_tables/2026-09-07_pwarp_amplify.md` | Slide was a crop, not a pan. Ramp \(t \cdot v\) tried. |
| **Pwarp amp leftover n=2** | `sweep_experiment/reports/paper_tables/2026-09-07_pwarp_amp_spec.md` | **DONE / NO.** 17172470–483. A dxL=−15 still Dyn 1/2. B persist IQ 38.94. C/D/E ≈ notta. |
| **Pwarp amp harvest** | `sweep_experiment/reports/paper_tables/2026-09-08_pwarp_amp_harvest.md` | Protocol PASS. Cite `wan_notta` 75.60 / 1/2. All five **NO**. Do not letter n=2. Do not launch 128. |
| **Territory A sentence + kill** | `sweep_experiment/reports/paper_tables/2026-09-08_territory_a_sentence_kill.md` | **DRAFT.** Honest: SF protocol ablation. Not a title. |
| **Why DMD vs frozen noise** | `sweep_experiment/reports/paper_tables/2026-09-08_why_dmd_frozen_noise.md` | DMD papers say train=infer, not “don’t warp frozen noise.” GwF: video warp needs FT. FIFO claims training-free schedule; our FIFO was NO. |
| **Idea 2 occupied** | `sweep_experiment/reports/paper_tables/2026-09-08_idea2_occupied_rolling.md` | Trajectory sink **is** Rolling’s first-chunk sink. Real leftover vs self-chunk is not a title. |
| **Distill ideas from atlas** | `sweep_experiment/reports/paper_tables/2026-09-08_distill_ideas_from_atlas.md` | Idea 2 **OCCUPIED**. Rank-1 still search-mode distill. No GPU until user picks. |
| **Search-mode distill neighbors** | `sweep_experiment/reports/paper_tables/2026-09-08_search_mode_distill_neighbors.md` | Idea 1 class is published (BOND / DanceGRPO / Reward Forcing / **Alice v1** [2605.08115](https://arxiv.org/abs/2605.08115)). Not an empty field. |
| **BOND + DMD on Best-of-N** | `sweep_experiment/reports/paper_tables/2026-09-08_bond_dmd_bon_open.md` | Vanilla winner-only DMD is the control, not the title. BOND’s open issues: judge, Jeffreys, iterative N=2. |
| **Winners / losers / energy** | `sweep_experiment/reports/paper_tables/2026-09-08_winner_loser_energy.md` | Negatives yes. Do not teacher-match losers. Low-energy well must be in-scene and living. |
| **Streaming / CL novel methods** | `sweep_experiment/reports/paper_tables/2026-09-18_streaming_cl_novel.md` | Fills the briefing blank. Rank-1: reality-ranked amortize (world as judge). Rank-2: scene-well energy. No GPU until user picks. |
| **Fast/slow streaming reviewer pass** | `sweep_experiment/reports/paper_tables/2026-09-20_fast_slow_streaming_review.md` | (b) delta-memory + evict-after-slow is not a title yet. ARL² / TTT-Video / EMA-sink occupy. Slow step cannot sit in the 17–23 FPS loop. |
| **Streaming × CL field buckets** | `sweep_experiment/reports/paper_tables/2026-09-20_streaming_cl_field_buckets.md` | Challenges / failures / approaches mapped to fights. Keep the evict-blocked-until-sleep policy, not the name stack. |
| **Slow object: DMD sleep vs scene well** | `sweep_experiment/reports/paper_tables/2026-09-20_slow_object_dmd_vs_well.md` | Same eviction chassis. Train-time teacher-DMD = student paper. Tiny well bank = frozen generator. Do not stack. |
| **Surprise replay + fast-weight read** | `sweep_experiment/reports/paper_tables/2026-09-20_surprise_replay_and_fastweight_read.md` | Surprise-weighted replay is Titans/SuRe. Forcing SOTA reads KV+sink, not W_fast. ARL²/TTT-Video already read a fast state at emit. |
| **GT stream vs no-GT assessment** | `sweep_experiment/reports/paper_tables/2026-09-20_gt_stream_vs_nagt_assessment.md` | Pursue leftover-growth hybrid. Pure no-GT CL is a motivation swap. GT of the generated horizon is prediction. |
| **Field gaps + settings** | `sweep_experiment/reports/paper_tables/2026-09-20_field_gaps_and_settings.md` | Four real gaps from 2026 streaming-gen + video-CL. Default first: Gap 1 on caption V2V leftover → 30 s. Do not invent a GT delay. |
| **Leftover scene well method** | `sweep_experiment/reports/paper_tables/2026-09-20_leftover_scene_well_method.md` | One method for Gaps 1+3+4. Blocked KV evict until a leftover well writes. Gap 2 is a later student. No GPU. |
| **Named evict vs OOD store-cap** | `sweep_experiment/reports/paper_tables/2026-09-20_named_evict_and_ood_store.md` | Name is real only vs blend-at-cut. OOD cap: yes on generated, no on leftover (that opens a new well). |
| **Problem setting LOCKED** | `sweep_experiment/reports/paper_tables/2026-09-20_problem_setting_locked.md` | Caption V2V leftover → 30 s, first-32, leftover-only well, cite `wan_notta`. Do not re-ask task / N / store-self. |
| **T2V compare vs success odds** | `sweep_experiment/reports/paper_tables/2026-09-20_t2v_compare_and_success.md` | T2V is easier to cite and hollows the well. Odds: low VBench title, moderate cut-hygiene if leftover grows. Setting stays V2V. |
| **V2V vs T2V not principal** | `sweep_experiment/reports/paper_tables/2026-09-20_t2v_v2v_not_principal.md` | User analogy accepted: first generated chunk = clean prefix. T2V still not the run: that instance is Rolling/AdaState sink. |
| **Banded fast-weight write** | `sweep_experiment/reports/paper_tables/2026-09-20_banded_fastweight_write.md` | Drawing board. Promote mid→mid-high OOD into \(W_{\text{fast}}\); refuse high. Representation paper dropped. |
| **Banded FW overlaps** | `sweep_experiment/reports/paper_tables/2026-09-20_banded_fw_overlaps.md` | Five shared concepts vs Titans / TTT / ARL² / SuRe / Alice. Remaining sentence: shoulder write + tail refuse. |
| **Beyond OOD for artifacts** | `sweep_experiment/reports/paper_tables/2026-09-20_beyond_ood_artifacts.md` | Promote = Titans residual. Refuse = IQ+flicker/subject vs this opening, not dataset divergence. |
| **Opening-conditional refuse** | `sweep_experiment/reports/paper_tables/2026-09-20_opening_conditional_refuse.md` | General refuse: teacher residual \(p(\text{chunk}\mid\text{opening})\), quantile-calibrated on the opening. Checklist is diagnostic only. |
| **Opening spread match** | `sweep_experiment/reports/paper_tables/2026-09-20_opening_spread_match.md` | Two-sided support + scale. **T2V:** first chunks are the high-spread reference (freeze is the tail death). I2V-still is another protocol. |
| **Slow stays frozen** | `sweep_experiment/reports/paper_tables/2026-09-20_slow_stays_frozen.md` | User pipeline: KV then gated \(W_{\text{fast}}\) write. 1.3B not updated at test. Fast weights are session-local. |
| **KV cache vs fast weights** | `sweep_experiment/reports/paper_tables/2026-09-20_kv_vs_fast_weights.md` | KV = every recent token for attention. \(W_{\text{fast}}\) = gated compression that can outlive the window. |
| **Spread-as-motion novelty** | `sweep_experiment/reports/paper_tables/2026-09-20_spread_motion_novelty.md` | Gauge is old (FlowMo / AdaIN / \(\|\Delta\mathrm{frame}\|\)). Paper must stay the gated \(W_{\text{fast}}\) write. |
| **Prefix-protected fast weights** | `sweep_experiment/reports/paper_tables/2026-09-20_prefix_protected_fast_weights.md` | Proposed method: protect / update / fork on \(W_{\text{fast}}\). Titans inverted on freeze; EMA inverted on smear. |
| **KV sink vs FW gate** | `sweep_experiment/reports/paper_tables/2026-09-20_kv_sink_vs_fw_gate.md` | Permanent first-chunk KV sink kills motion (RF; Rolling Dyn%). Gated \(W_{\text{fast}}\) only if that sink is removed. |
| **Prefix stats as gate** | `sweep_experiment/reports/paper_tables/2026-09-20_prefix_stats_as_gate.md` | Tokens evict; frozen \((\mu,\mathrm{scale})\) only labels the \(W_{\text{fast}}\) write set. Medium novelty vs prototype CL / EMA / Titans. |
| **Prefix-protect first-8 spec** | `sweep_experiment/reports/paper_tables/2026-09-20_t2v_pprot8_spec.md` | MovieGen T2V 30 s. `notta` / `sf_window` / `sf_pprot`. No first-chunk KV sink. **SUBMIT-READY, not launched.** Do not letter n=2. Do not launch 128. |
| **Sponsor summer report** | `sweep_experiment/reports/paper_tables/2026-09-20_sponsor_summer_report.md` | External partner note. Classes + outcomes only; no unpublished recipe. Figures: `paper_tables/sponsor_summer_2026_figures/`. Late-Sep class veto added 2026-09-28. Writing handoff: `2026-09-28_sponsor_report_handoff.md`. |
| **Temporal threshold → FW write** | `sweep_experiment/reports/paper_tables/2026-09-21_temporal_threshold_fast_weights.md` | Coincidence window + dynamic spike threshold as the \(W_{\text{fast}}\) write. Not STDP. Not a spiking backbone. |
| **Coincidence-gated FW update** | `sweep_experiment/reports/paper_tables/2026-09-21_coincidence_fastweight_update.md` | Detailed write: event tape, high-pass, spatial \(C\), Azouz \(\theta\), DeltaNet on \(S\) only, no decay. Not a submit. |
| **Coincidence write publishability** | `sweep_experiment/reports/paper_tables/2026-09-21_coincidence_publishability.md` | Medium novelty. Top venue only if isolation holds on 128. Not on the sentence alone. |
| **Temporal logic, other uses** | `sweep_experiment/reports/paper_tables/2026-09-21_temporal_logic_other_uses.md` | Same timing rule on freeze / KV split / eviction / flicker / fork / CL. Write stays first. |
| **Coincidence first-8 spec** | `sweep_experiment/reports/paper_tables/2026-09-21_t2v_coinc8_spec.md` | MovieGen T2V 30 s. **DONE / NO.** Quality: `2026-09-22_t2v_coinc8_quality.md`. After-NO: `2026-09-22_coinc8_after_no_options.md`. Match ARL² (156 H100-h): `2026-09-22_competitor_fw_settings.md`. |
| **Search vs KV open challenges** | `sweep_experiment/reports/paper_tables/2026-09-22_search_kv_open_challenges.md` | Coinc / KV-admission **dropped**. Remaining TTA holes after 2024–26. No GPU until user picks a row. |
| **Leftover / well / memory glossary** | `sweep_experiment/reports/paper_tables/2026-09-20_leftover_well_memory_glossary.md` | **SAY:** context frames, KV cache, context-frame representation. Well ≠ fast weights. Old slang stays in old files only. |
| **Pwarp failure points** | `sweep_experiment/reports/paper_tables/2026-09-06_pwarp_failure_points.md` | Harvest named F3 (1 cell/strip on 0007) + F6 (dust pan on 0002). |
| **Gate neighbors + publishability** | `sweep_experiment/reports/paper_tables/2026-09-01_gate_neighbors_publishability.md` | EFD / SDVG / Video-T1 / CachedSearch / LatSearch. 13% vs Always is not a quality paper. |
| **RF schedule neighbors** | `sweep_experiment/reports/paper_tables/2026-09-01_rf_noise_schedule_neighbors.md` | Deep / Relax / Ms. / Stream / Reward / FIFO. Most RF follow-ons are memory. TTA cousins: lookahead, shallower / local-steep diagonal. |
| **RF non-linear timestep list** | `sweep_experiment/reports/paper_tables/2026-09-01_rf_nonlinear_schedule.md` | **DONE / NO.** Linger / dump Imaging Quality died. Harvest: `2026-09-04_wan_v2v_caption_schedule8_harvest.md`. Do not start 8-GPU DMD. |
| **Schedule8 harvest** | `sweep_experiment/reports/paper_tables/2026-09-04_wan_v2v_caption_schedule8_harvest.md` | 16855778–780 COMPLETED. Native floor 556. Linger −10% IQ 66.34. Dump +39% IQ 68.14. Both **NO**. |
| **SF / RF shared experiment machine** | `sweep_experiment/reports/paper_tables/2026-09-04_sf_rf_common_impl.md` | Both papers: unroll inference + holistic DMD. Mix + context noise are diagnostic, not the paper idea. |
| **SF / RF KV + compute audit** | `sweep_experiment/reports/paper_tables/2026-09-04_sf_rf_kv_opt_audit.md` | Quality KV / sink / RoPE / window already on. Do not retune `enlarge_kv` / `local_attn` / extra sink on cite hosts. |
| **Mix + context-noise spec** | `sweep_experiment/reports/paper_tables/2026-09-04_wan_v2v_caption_mixctx_spec.md` | Caption N=8. `rf_mix` / `sf_mix` + always + `*_ctx`. Do not remake cite-128. |
| **FIFO + lock-score spec** | `sweep_experiment/reports/paper_tables/2026-09-04_wan_v2v_caption_fifo_tscore_spec.md` | Caption N=8. `rolling_fifo` + `rf_tscore` / `sf_tscore`. 1.3B freeze-score, not Wan-14B. |
| **Mix+ctx harvest** | `sweep_experiment/reports/paper_tables/2026-09-04_wan_v2v_caption_mixctx_harvest.md` | 16931124–130 COMPLETED. All six **NO**. |
| **FIFO+tscore harvest** | `sweep_experiment/reports/paper_tables/2026-09-04_wan_v2v_caption_fifo_tscore_harvest.md` | 16931441–447 COMPLETED. All six **NO**. Gated tscore = host identity. |
| **LPIPS + aligned FVD spec** | `sweep_experiment/reports/paper_tables/2026-09-01_wan_v2v_lpips_fvd_spec.md` | Fill existing pixel jsons. I3D on 30 s tails. **DONE 16738784.** Harvest: `2026-09-04_wan_v2v_cite128_lpips_fvd.md`. |
| **Cite-128 LPIPS/FVD** | `sweep_experiment/reports/paper_tables/2026-09-04_wan_v2v_cite128_lpips_fvd.md` | SF 0.745 / 410. Pseudo 0.753 / **405**. Always 0.751 / 425. RF 0.762 / 436; last16 RF **1108**. |
| **Cite-128 full grid (2026-09-04)** | `sweep_experiment/reports/paper_tables/2026-09-04_wan_v2v_cite128_all_metrics.md` | VBench + pixels + LPIPS + FVD. Supersedes open cells in the 2026-08-31 grid. |
| **Denoise-hooks spec** | `sweep_experiment/reports/paper_tables/2026-08-28_wan_v2v_denoise_hooks_spec.md` | lastmix / bpseudo / restep. Caption N=8. `WAVE=lastmix` first. |
| **Pseudo-next N=8 spec** | `sweep_experiment/reports/paper_tables/2026-08-31_wan_v2v_pseudo_next8_spec.md` | Cheapen + re-gate. Caption N=8. **NO.** Harvest: `2026-08-31_wan_v2v_pseudo_next8_harvest.md`. |
| **Caption leftover ρ spec** | `sweep_experiment/reports/paper_tables/2026-09-01_wan_v2v_caption_leftovers_spec.md` | Stem leftovers are panda-infected. Caption N=8 ρ / look only. Do not remake cite-128. |
| **Caption leftover harvest** | `sweep_experiment/reports/paper_tables/2026-09-01_wan_v2v_caption_leftovers_harvest.md` | 16734909–913 COMPLETED. All four **NO**. Real captions did not save Imaging Quality. |
| **Cite-128 wall (job/96)** | `sweep_experiment/reports/paper_tables/2026-09-01_wan_v2v_cite128_wall.md` | 108 / 47 / 294 / 354. Not n=32 sidecar 196 / 45 / 304 / 348. |
| **Weekly recap (current week)** | `weekly_recap_YYYY-MM-DD.md` | One per Monday meeting. Latest: `weekly_recap_2026-09-08.md` |
| **Daily experimental-output log** | `sweep_experiment/reports/experiment_outputs/YYYY-MM-DD.md` | Append every pasted output (raw + interpretation) |
| **Canonical results memory (legacy)** | `sweep_experiment/reports/experiment_metrics_log.md` | Long-form running log. Superseded by INDEX.md + ANALYSIS_LOG.md as of 2026-06-08, but kept for history. |
| **Paper draft** | `sweep_experiment/reports/paper_draft.md` | LaTeX-aligned narrative + result placeholders. Often dehydrated locally. |
| **Paper LaTeX** | `paper/main.tex`, `paper/sections/*.tex`, `paper/refs.bib` | Real submission source |
| **Run registry** | `experiment_tracker/run_registry.yaml` | Job-ID ↔ result-dir mapping |
| **Laptop → cluster SSH/SCP** | **`wc3013@torch`** | **LOCKED. From this Mac, the host is `torch` (SSH config alias). Never invent `torch-login-a-*.hpc.nyu.edu` or `torch-login-b-*` as the scp/ssh target. Prompt `[wc3013@torch-login-a-1]` is the node after login, not the host you type.** |
| **Cluster repo root (this repo)** | `/scratch/wc3013/longcat-video-tta/` | Video-generation results and this operating manual. Local repo is mostly views. |
| **Artificial Individuality root** | `/scratch/wc3013/artificial-individuality` | Live project. Code, manifests, schedules, checkpoints, and logs go here. Read cluster procedure from this repo; do not write the new experiment into it. |
| **Wan 1.3B / Self-Forcing setup** | `wan_experiment/README.md` | I2V-32 is **discovery only**. Official VBench **DONE** (full-clip tie). **Do not scale I2V-32.** Current next: V2V Panda bake-off (`2026-08-20_wan_v2v_sampling_bakeoff_spec.md`). T2V 128 is optional. Do **not** add TTC. |

## 2. CRITICAL workflow rules

### 2a-pre. Laptop SSH/SCP host is `wc3013@torch` (LOCKED 2026-09-09)

From the user's local machine, every `ssh` / `scp` / `rsync` to the
cluster is:

```bash
wc3013@torch
```

**Never** write any of these as the host the user types:

- `wc3013@torch-login-a-1.hpc.nyu.edu`
- `wc3013@torch-login-a-0.hpc.nyu.edu`
- `wc3013@torch-login-b-2.hpc.nyu.edu`
- `torch-login-*` of any letter

`torch` is the SSH config alias on this Mac. After login, the prompt
may say `[wc3013@torch-login-a-1]` — that is the node you landed on,
not the hostname for the next SCP command. Agents have invented the
FQDN many times; the user has had to correct it every time. Do not
do it again.

Example:

```bash
scp wc3013@torch:/scratch/wc3013/longcat-video-tta/<remote> <local>
```

### 2a. iCloud / `UF_DATALESS` gotcha

The local repo at `/Users/macrohard/Desktop/longcat-video-tta/` lives on iCloud
Drive. Files routinely dehydrate (`UF_DATALESS`) and `.git` ops time out
(`ETIMEDOUT`). **Never** run `git add / commit / push` directly from the local
working tree. Use the subagent pattern:

1. Write/edit files in `/Users/macrohard/Desktop/longcat-video-tta/` with the
   normal file tools — this works because writing materializes the file.
2. Dispatch a `shell` subagent that does `git clone --depth=1` into `/tmp`,
   `cp` the local files into the clone, then commits and pushes from `/tmp`.
3. Subagent prompt template is in §5 of this file.

After pushing, the user pulls on the cluster:
```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
```

### 2b. Save EVERY pasted experimental output

When the user pastes terminal/cluster output:

1. **Append the raw output** verbatim to today's
   `sweep_experiment/reports/experiment_outputs/YYYY-MM-DD.md` with a
   timestamped section header.
2. **Add a 1-3 line interpretation** below the raw block.
3. If the output contains paper-grade metrics, also update the relevant
   weekly recap table AND the master `INDEX.md`.
4. If a new fact emerges that future agents must know (a path, a bug, a
   workflow change), add it to AGENTS.md.

If the date file doesn't exist yet, create it with the standard header
template in §4.

### 2b-quater. Jobs that left `squeue` (ADDED 2026-09-01)

`squeue` only shows what is still queued or running. A finished
job vanishes. If a known JobID is missing from the next
`squeue` paste, **immediately** `sacct -j <ID>` and harvest
disk. Do not wait for the user to notice. Caption leftover
16734909–911 left the queue at 15:20 and were not harvested
until the user called it out.

### 2b-ter. Same-wave ablations (ADDED 2026-08-24)

When submitting a **gated** method, the same paste must also
launch the obvious twins: **always-on** (no gate) and the
**other host** if the claim is host-specific. Do not wait for
harvest to invent the ablation. Harvest still decides the call.
k stays locked to the family width (k=4) unless a dated spec
says otherwise. CachedSearch headline N=8 is a later width
sweep, not a harvest retune.

### 2b-bis. Record-keeping commandments (ADDED 2026-06-08 after the user
called out repeated record-keeping failures)

These are NON-NEGOTIABLE. Every agent that touches this repo must follow
them. The user has explicitly cited "bad record keeping" as a paper-blocking
issue. Past failures included: stale `merged_summary.json` numbers leaking
into recap tables; results not saved locally; no clear mapping from cluster
paths to paper tables; no audit trail when narrative pivots happened.

1. **Whenever you produce a table that the user might cite in the paper or
   meeting,** save it as a dated Markdown file under
   `sweep_experiment/reports/paper_tables/YYYY-MM-DD_<short_name>.md` AND
   push it to GitHub the same turn. Do NOT just emit it inline in chat
   and move on.

2. **Whenever a new experiment series finishes,** add a row to
   `sweep_experiment/reports/INDEX.md`. Include cluster path, methods,
   N, frames, status, and key finding. Update existing rows when a series
   is re-merged or backfilled.

3. **Whenever you reach a methodology decision or paper-narrative finding,**
   append an entry to `sweep_experiment/reports/ANALYSIS_LOG.md` with date,
   tags, refs, and 5–15 line body. Past entries are immutable; rebut with
   a new entry.

4. **Whenever the user pastes raw cluster output,** the rule from §2b
   applies (append to `experiment_outputs/YYYY-MM-DD.md`). DO NOT skip
   this even if the output seems "uninteresting" — context that looks
   trivial today is what unblocks debugging next week.

5. **Every paper-grade table must be regenerable from cluster data.**
   Use `scripts/build_paper_tables.py --regime <regime>` to produce
   tables from `merged_summary.json` files. If you produce a table by
   any other means (manual edit, ad-hoc calculation), document it in
   `ANALYSIS_LOG.md` so future agents know not to overwrite it.

6. **Push every record-keeping update to GitHub the same turn.** The
   local iCloud workspace is unreliable; GitHub is the persistence layer.
   If you wrote to `INDEX.md` / `ANALYSIS_LOG.md` / `paper_tables/*.md`,
   the next thing you do is dispatch the subagent push (§5).

7. **If you find a stale or wrong number in a published table,** add a
   new dated table file (don't edit the old one — keep the audit trail)
   AND add an entry to `ANALYSIS_LOG.md` explaining what was stale and why.

8. **NEVER delete generated videos without a manifest + metric capture.**
   The `*.mp4` files are the ONLY source from which frame-based metrics
   (VBench++, FID, FVD, future perceptual metrics) can ever be recomputed;
   once deleted a run is frozen at whatever is already in `merged_summary.json`.
   Before deleting any generated videos:
   (a) run `scripts/build_run_manifest.py` (records per-run provenance +
       metrics + a sha1 pool fingerprint proving which runs share a test set);
   (b) confirm each run's needed metrics — ESPECIALLY VBench — are captured,
       and backfill on the saved frames first if not;
   (c) curate a few matched examples into `figure_bank/` via
       `scripts/curate_figure_bank.py` (keys on the normalized video index so
       NOTTA/ADA/LoRA filename schemes align);
   (d) delete only via `scripts/cleanup_generated_videos.sh` — use PURGE
       allowlist mode (delete only named series), which refuses to run without
       a manifest and hard-protects `datasets/`, `**/figure_bank/`,
       `baseline_experiment/results/gt_clips_*`, and `LongCat-Video/`.
   Comparison-method / T2V baselines (PVDM, DFoT, OpenSora, LongCat-T2V) have
   NO stored metrics as of 2026-07-17 — do not delete their frames until their
   FVD/FID/etc. are computed and saved.

### 2c. Local repo is mostly dehydrated

Don't waste tokens trying to read files like
`sweep_experiment/reports/experiment_metrics_log.md` from the local filesystem
— they're almost certainly dehydrated. To read them, do one of:

- Ask the user to `cat` it on the cluster and paste the output.
- Dispatch a subagent that clones from GitHub into `/tmp` and reads from there.
- Read the raw blob via `gh api` (slower).

For writing: writing **creates** the file fresh on disk (no dehydration
issue), so writing to `experiment_outputs/2026-06-01.md` is fine even if the
parent directory is dehydrated — the directory rehydrates as the file appears.

### 2d. Cluster series-name conventions (CONFIRMED 2026-06-01)

The cluster's `sweep_experiment/results/` directory naming convention for
recent paper-grade work:

| Series dir | What it is | Methods present |
|---|---|---|
| `panda_1000v_standard/` | Panda-70M N=999 std horizon | NOTTA, ADA, LORA_R8_TTA |
| `panda_longctx_1000v/` | Panda-70M N=999 LONG context | NOTTA, ADA_S10, LORA_R8 |
| `ucf101_932v_standard/` | UCF-101 N=932 std horizon | NOTTA, ADA, LORA_R8_TTA |
| `ucf101_683v_longhorizon/` | UCF-101 N=683 long horizon | NOTTA, ADA, LORA_R8_TTA (7 chunks — partial?) |
| `ucf101_932v_retrieval/` | UCF retrieval sweep (this week) | K5/K10 × SIM/RAND, NOT YET MERGED |

TinyLoRA lives separately under `delta_experiment/results/`:
- `tinylora_panda_1000v_standard/` (TL_BARE_R2, TL_TIED_R2 — merged)
- `tinylora_ucf101_932v_standard/` (TL_BARE_R2, TL_TIED_R2 — NOT merged)
- `tinylora_longctx_1000v/` (PANDA_TL_LAST24 — merged, used in §1.1 of recap)

### 2d-bis. Comparison baselines (ADDED 2026-07-17)

Three TTA comparison baselines for AdaSteer, all leakage-free and pinned to the
paper pools + shared metric code:

| Baseline | Horizon | Entry point | Notes |
|---|---|---|---|
| **SAVi-DNO** (noise opt) | short (14+14@48) | `comparison_methods/scripts/savi_dno_longcat.py` + `sbatch/run_savi_dno_longcat.sbatch` | DEFAULT is now the **fair leakage-free** protocol (adapt noise on observed history, predict unseen future). `--oracle-leak` (env `SAVI_ORACLE_LEAK=1`) reproduces the OLD behaviour = optimizes noise against the scored future = ORACLE upper bound only. The previous default leaked GT — do NOT cite pre-2026-07-17 SAVi-DNO numbers. |
| **SlowFast-VGen** (Temp-LoRA) | short | `lora_experiment/scripts/run_temp_lora_tta.py`, `METHOD=temp_lora` in run_sweep | Config `sweep_experiment/configs/panda_1000v_temp_lora.yaml`. Temp-LoRA fast-learns sequentially (warm-start streams over observed context, then per-rollout-chunk updates on self-generated frames). Uses the standard summary/FVD/VBench plumbing. |
| **TTC** (pathwise correction) | **LONG only** (14+79@14) | `comparison_methods/scripts/ttc_longcat.py` + `sbatch/run_ttc_longcat.sbatch` | Training-free; re-anchors appearance to first frame at low-noise steps. Do NOT report short-horizon. Knobs (sigma-threshold/cadence/weight) are a LongCat adaptation of the T2V TTC paper — tune/validate before citing. `--no-correction` gives the matched baseline arm. |

All three were syntax/dry-run validated 2026-07-17 but NOT yet cluster-run;
smoke-test before full 1000v.

### 2e. `merged_summary.json` schema (CONFIRMED 2026-06-01)

Top-level keys (no nested `metrics` dict):

```
psnr, psnr_std, ssim, ssim_std, lpips, lpips_std
fvd, fvd_num_chunks, fvd_num_videos, fvd_num_ref_videos, fvd_per_chunk
fid, fid_num_frames_gen, fid_num_frames_ref, fid_per_chunk
num_chunks, num_successful, num_videos
vbench, vbench_num_chunks
avg_train_time, avg_gen_time, avg_total_time
```

Per-method `merged_summary.json` lives at:
`{sweep,delta}_experiment/results/<series>/<METHOD>/merged_summary.json`

## 3. Active project state (snapshot — keep current)

**Date:** Updated 2026-09-26.

**Live task:** Artificial Individuality (§7), rooted at
`/scratch/wc3013/artificial-individuality`. This checkout is context
for how Torch works. Video bullets below are the frozen record. Do
not submit Wan, Self-Forcing, nwarp, pwarp, coincidence, or DMD jobs
from this section.

- **Paper target:** CVPR 2027.
- **Paper method (2026-09-04):** Pseudo-future Search is **dropped**
  as the title. A 13% gate on Always-search is not a CVPR idea.
  Next fork: territories A (new student) / B (analysis paper) /
  C (new frozen-weight control). No GPU until the user picks.
  `paper_tables/2026-09-04_drop_pseudo_next_territories.md`.
- **Method stack (current):** Wan2.1-T2V-1.3B + Self-Forcing causal DMD.
  I2V-32 30 s is a **discovery / stress** run, not the field
  long-horizon table. **Do not scale I2V-32 or I2V-200.** **Do not
  add TTC / LoRA-at-test-time.** LongCat 13.6B stays the
  saturated-large-model audit. Do not launch more LongCat TTC.
- **Protocol stop (2026-08-18):** recent long-horizon papers on this
  model family are **T2V** self-continuation, **128 MovieGen**
  (Qwen-refined), **VBench-Long**, 30 s / 60 s. Ours was I2V-from-still,
  N=32, `custom_input`. Length 30 s and Wan 1.3B are fine; task / N /
  suite are not. Stop:
  `paper_tables/2026-08-18_wan_protocol_stop.md`.
- **Task lock (2026-08-18, user correction):** T2V was **not** agreed.
  I2V-from-still scale-up stays closed. **V2V prefix-continuation is
  allowed** and is the closer match to the claim (visual history →
  long AR). T2V 128 MovieGen is only an optional comparison to Relax
  Forcing–style tables. Note:
  `paper_tables/2026-08-18_v2v_continuation_allowed.md`.
- **Optional T2V compare (SUBMIT-READY, not launched):** T2V 30 s,
  128 MovieGen, do-nothing | always-BoN | gated-BoN. New runner
  `wan_experiment/scripts/run_t2v_chunked.py`. Submit:
  `SMOKE=1 bash wan_experiment/sbatch/submit_t2v_bon128.sh` then
  `bash wan_experiment/sbatch/submit_t2v_bon128.sh`. Spec:
  `paper_tables/2026-08-18_wan_t2v_vbenchlong_128_spec.md`.
- **V2V caption bug (2026-08-24):** Panda pool had
  `metadata.csv` (1000/1000 list captions). The runner only
  loaded JSON, so every finished V2V arm used filename stems
  (`panda 0013`). Real 0013 caption is a bathroom stain.
  Tail→panda is T5 takeover. Same-prompt deltas still hold.
  Runner now reads `metadata.csv`. Caption WAVE=1
  **protocol PASS** (`prompt_source=metadata_csv`, 0 stem).
  WAVE=1 generate **32/32** all arms. Always **16310324**
  COMPLETED tail **+39%** 30/2 (Pseudo +28% 23/0/9). VBench
  **16310330** still R; notta/rolling/rewind written (SF
  0.700/71.54/0/0.989; RF IQ −1.32 vs SF). AdaSteer N=8
  **NO** (16326033–036 COMPLETED): `|δ|`≈0.84, IQ 43/51/18.
  Do not scale AdaSteer. Do not mix stem-prompt numbers into
  caption tables.
  Outcomes: `paper_tables/2026-08-24_wan_v2v_caption_wave1_outcomes.md`.
  Spec: `paper_tables/2026-08-24_wan_v2v_caption_rerun_spec.md`.
- **Language (2026-09-20):** To the user say
  **context frames**, **KV cache**, **fast weights**.
  Do not say leftover / well / memory. Rule:
  `.cursor/rules/field-language.mdc`.
- **Current next (2026-09-22):** Coincidence
  first-8 and whole-latent KV admission
  are **dropped**. Search vs KV
  remaining-challenge review is the
  live note:
  `paper_tables/2026-09-22_search_kv_open_challenges.md`.
  Still-open after 2024–26: honest
  temporal judge; mid-horizon /
  occlusion memory; 4-step search
  cheapen; training-free 30 s that
  holds IQ and Dyn (Deep Forcing /
  Relax claim, unreproduced).
  AdaState occupies evolve-the-sink.
  No GPU until the user picks a
  row. Do not letter n=2. Do not
  launch 128. No I2V. No TTC. Do
  not re-impl coinc-8 or KV-admission.
  coinc-8 harvest stays **DONE / NO**
  (IQ 56.29 / Dyn 8/8 twitch vs
  notta 72.96 / 2/8). Quality:
  `paper_tables/2026-09-22_t2v_coinc8_quality.md`.
  Prior Wan-teacher leftover still stands:
- **Current next (2026-09-07):** Clean Wan-teacher
  host. Portable ideas cite `wan_notta` (official
  Wan2.1-T2V-1.3B, no `self_forcing_dmd.pt`).
  Pwarp amplify leftover n=2 **DONE / NO**
  **17172470–483 COMPLETED 0:0.** Protocol PASS.
  Cite `wan_notta` IQ 75.60 / Dyn 1/2. A ramp
  `dxL=−15` still Dyn 1/2. B persist IQ **38.94**.
  C/D/E ≈ notta. E skipped 0001. All five **NO**.
  Do not letter n=2. Do not launch 128. Do not
  scale pwarp. No GPU until the user picks.
  Wan-teacher leftover + MovieGen smoke
  **HARVESTED** 17135846–861 COMPLETED 0:0.
  Protocol PASS. Cite `wan_notta`. Leftover IQ
  75.60 / Dyn 1/2. MovieGen IQ 69.97 / Dyn 2/2.
  nwarp IQ **54.25 / 51.60** Dyn 0/2 both
  datasets — **NO** even isolated from SF.
  pwarp ≈ notta, no Dyn lift. Do not letter n=2.
  Do not launch 128. Do not scale nwarp or pwarp.
  Native clip 81 frames. nwarp = HIWYN on \(x_T\)
  once. pwarp = mid-timestep pred slide. Do not
  launch 128. Do not remake cite-128. Forcing-only
  (Rolling / mix / FIFO / leftover ρ) stay on the
  existing SF/RF tables. Caption Wan-extend N=8
  **DONE / NO** (17095709–711). IQ 69.22 / subject
  0.576 / Dyn 4/8 vs SF first-8 70.62 / 0.658 / 2/8.
  Extra Dyn = invented pans. Do not scale. Do not
  pwarp that dest. Caption nwarp N=8 **DONE / NO**.
  Caption pwarp N=8 **DONE / NO**
  (`sf_pwarp` / `sf_pwarp_live`, 17058386–393). Slide
  fired; IQ **66.81** both arms; extra Dyn is 0007
  twitch (flicker 0.878). Harvest:
  `2026-09-06_wan_v2v_caption_pwarp_harvest.md`. Do
  not retune extra γ. Do not stack nwarp. Still no
  8-GPU DMD. Eyes: 0007/0004 flicker is in Self
  Forcing and worse after the slide. 0002 rewrites
  a still room. Only 0006’s caption asks for motion
  and it did not get more. A prompt-gate on this
  eight is one clip. Prompt rewrite in the field is
  train/test text-length match (Wan Qwen extend,
  MovieGen-128 Qwen). Not a leftover flow. 0006 is
  likely zoom/expansion; our slide only pans. Three
  tracks (2026-09-06): A login pan-filter
  (`filter_pwarp_pan_shortlist.py`, **Self Forcing
  python** — login `base` imageio has no ffmpeg.
  First-128 keep 3/128 were word accidents; do not
  `--write-dir`. User started track A 2026-09-07:
  retag json then `--n 1000` if thin. Table
  `2026-09-07_pwarp_pan_filter_128.md`). B Wan-extend
  N=8 **DONE / NO** 17095709–711. C MovieGen T2V +
  first-chunk nwarp/pwarp **SMOKE HARVESTED**
  17121785–792 COMPLETED 0:0. Protocol PASS
  (`t2v_chunk0` 8/8). notta IQ 70.93 / Dyn 1/2.
  nwarp IQ **47.60** (same death as leftover).
  pwarp IQ 69.35, Dyn still 1/2. That was the
  **wrong host** (SF extras). Isolated kind-A table
  is the Wan-teacher smoke. Do not letter n=2. Do
  not launch 128. Do not scale nwarp or pwarp on SF.
  Panda is not their 30 s table. Do not cheapen
  Pseudo.
  Do not scale mix / FIFO / tscore / ρ / list /
  nwarp / pwarp.
  Caption official (historical):
  N=32 **DONE**. Cite Dyn as **percent of clips** (VBench official),
  not median. SF 21.9% (7/32), Pseudo **40.6%** (13/32), Always
  **43.8%** (14/32). `rf_sink` 0.709 / 70.15 / 15.6% / 0.980.
  Prefix-match NO. AdaSteer N=8 **NO**. Table:
  `paper_tables/2026-08-25_wan_v2v_caption_dyn_percent.md`.
  Method note (historical name only):
  `paper_tables/2026-08-25_pseudo_future_search.md`.
  Code `sf_pseudo` stays as a cite-128 ablation. **Not the title.**
  In-chunk: scored arms **NO** (lastmix / sf_bpseudo / rf_restep
  identity). SF intra + SF restep + RF bpseudo still **FAILED**.
  Caption-128 hosts **DONE**: SF 0.666 / 72.07 / **Dyn 32.8%
  (42/128)**; rolling +33% tail / 0.685 / 71.52 / Dyn **28.9%**.
  Cite 128 **COMPLETE**: Pseudo tail **0.0157 ≈ RF 0.0158**;
  official **Dyn 47.7% (61/128)** / 0.660 / 72.38. Always
  **50.8% (65/128)** / 0.661 / 72.19. Gate **90 fire / 38 skip**
  costs 4 Dyn clips. SF 32.8% (42). RF 28.9% (37).   Pixel PSNR/SSIM **DONE** (**16702323**): SF **9.25** /
  RF 7.98 / Pseudo **9.22** / Always **9.21**. Search ≈ SF
  on reconstruction; Rolling −1.28 dB. Headline stays
  VBench + Dyn%. Mid-chunk rewrite **CLOSED**. Pseudo-next
  N=8 **NO** (CachedSearch slower; re-gate no lift). Tables:
  `paper_tables/2026-08-31_wan_v2v_cite128_complete.md`,
  `paper_tables/2026-09-04_wan_v2v_cite128_all_metrics.md`
  (VBench 7/7; pixel suite **DONE**),
  `paper_tables/2026-09-04_wan_v2v_cite128_lpips_fvd.md`,
  `paper_tables/2026-09-01_wan_v2v_cite128_pixel.md`,
  `paper_tables/2026-08-31_wan_v2v_pseudo_next8_harvest.md`,
  `paper_tables/2026-08-31_wan_v2v_keep_intra_closed.md`.
  GPU: `paper_tables/2026-08-23_wan_gpu_batch_policy.md`.
  Success bar + neighbors (2026-08-30): RF quality, cost <<
  always-search; mid-chunk rewrite closed.
  `paper_tables/2026-08-30_wan_success_and_neighbors.md`.
  Beat-RF path (2026-08-30): not seed search. Intervene at
  window-exit (context noise / next-block noise / softer sink).
  `paper_tables/2026-08-30_wan_rf_intervene.md`.
  RF descendants (2026-09-01): Deep / Relax / Reward /
  Forcing-KV rewrite **memory**. Stream Forcing and
  Ms. Forcing are the schedule papers (need a student).
  TTA-legal cousins: FIFO lookahead, shallower diagonal,
  local next-block bump, context noise. Note:
  `paper_tables/2026-09-01_rf_noise_schedule_neighbors.md`.
  Next on Pseudo (2026-08-31): CachedSearch / re-gate **NO**.
  Gate is almost free; cheapen is still the paper move, but
  **not** this CPU KV snap. Search-early or prune k. Or RF
  window-exit. Pixel skip-existing **landed**. LPIPS/FVD
  **DONE 16738784**: SF 0.745 / 410; Pseudo 0.753 / **405**;
  Always 0.751 / 425; RF 0.762 / 436 (last16 **1108**).
  Caption leftover **NO**. Schedule8 linger/dump **NO**.
  Mix+ctx **NO**. FIFO **NO**. Gated lock-score = host identity.
  Caption nwarp **NO**. Caption pwarp **NO**.
  Pseudo-future Search **dropped as title**. Distill is
  territory A if the user picks it — do not start 8-GPU DMD
  tonight. Harvests:
  `paper_tables/2026-09-04_wan_v2v_caption_mixctx_harvest.md`,
  `paper_tables/2026-09-04_wan_v2v_caption_fifo_tscore_harvest.md`,
  `paper_tables/2026-09-04_drop_pseudo_next_territories.md`.
- **N=32 leftover (closed):** `appear_bon` NO. `rolling_notta` YES
  on locked tail+quality bars (Dyn 0). Host, not our controller.
  Verdict: `paper_tables/2026-08-22_wan_v2v_forward32_verdict.md`.
- **Next methods (no weights):** motion verifier + `{shift,cfg}` probe
  + prefix backtrack now live on V2V. CachedSearch / sink / HG-f wait.
  Memo: `paper_tables/2026-08-18_wan_nonweight_next.md`.
- **Week briefing (2026-08-18):** model + dataset switch with citations,
  plus long-horizon concepts. Setup talk, not the method talk.
  `paper_tables/2026-08-18_week_switch_briefing.md`.
- **Methods-since-switch talk (2026-08-24):** every widget, question,
  hypothesis, papers, gates, and real N=32 numbers. Anyone-readable.
  `paper_tables/2026-08-24_wan_methods_since_switch.md`. Canvas:
  `wan-methods-since-switch`.
- **Wan drift (2026-08-17, N=16):** 5 s median sharp +11% / motion −14%
  (mild). **30 s median sharp +167% / motion −60%** (15/16 each).
  Signature = sharpen + freeze. Table:
  `paper_tables/2026-08-17_wan_i2v_notta16_drift.md`.
- **Wan 16v three-way (2026-08-17):** last-chunk NOTTA 4.43 / always
  3.23 / gated 3.38. Search works. Gated vs always is **not** a
  quality win (mean +0.152, 6/16 better-or-tie); median slightly
  favors gated; always-on hurt 2/16. Honest line: efficiency
  controller that keeps most of the search gain. No TTC yet.
- **Wan hybrid 32v (2026-08-17):** cite medians, not means (video 26
  = 85.6). Last-chunk median NOTTA 3.68 / always 2.97 / gated 3.04.
  gated−always −0.041 / 0, 19/32 better-or-tie, **33% cheaper**.
  First-16 hybrid flipped T=2.0 +0.15 → −0.12. Efficiency on the
  **handcrafted score only**. Official VBench (full clip) is a tie —
  see the Official VBench bullet. Table:
  `paper_tables/2026-08-17_wan_i2v_bon32_hybrid.md`.
- **Wan sticky 32v (2026-08-18):** 03/24 caught (exact ties with
  always-search). 21/32 exact ties overall. Wall 256 vs 258 s —
  spent the hybrid 33% saving. Erased hybrid wins on 11 and 16.
  Not a quality win. Hybrid remains the efficiency method. 11/16
  diagnosis: hybrid slept after recovery; stay-on rebuilt
  always-search while the pick-score lied about the tail. Next
  lever: search-while-sick (turn off on recovery). Table:
  `paper_tables/2026-08-18_wan_i2v_11_16_diagnosis.md`.
- **Wan search-while-sick (2026-08-18):** Job **15959146 DONE**.
  Checklist pass on the handcrafted score. Median 2.764 vs always
  2.966 / hybrid 3.036. 11/16 recovered, 24 exact always, wall 204 s.
  9/14/9 — not a strict quality win. Table:
  `paper_tables/2026-08-18_wan_i2v_bon32_sick.md`. No TTC.
- **VBench protocol (locked 2026-08-18):** always score the **full
  generated clip**. That is the comparable number. last5 is optional
  diagnostic only — never the paper’s “VBench++” table. Defaults:
  `CLIPS=full last5`, analyze `--clip full`.
- **Official VBench (2026-08-18, DONE, hybrid 32):** full-clip is a
  tie (Aes 0.587/0.593/0.591, IQ 71.24/71.28/71.19, dynamic median
  0). last5 IQ drop is diagnostic only. Verifier anti-aligned with
  IQ on last5 (ρ +0.23 to +0.33). Read:
  `paper_tables/2026-08-18_wan_i2v_bon32_vbench_read.md`. No PSNR.
  No TTC.
- **LongCat audit (closed):** short-horizon in-domain 14→14 saturated;
  native AR long-horizon drifts; AdaSteer delta + routing closed;
  BoN k=4 N=8 passed credibility gate as always-on search, not a hard
  incoming-context gate.
- **In-flight cluster jobs** (as of 2026-09-22 20:55):
  Coincidence first-8 **DONE / NO**.
  18258206 COMPLETED 0:0. Titans rerun
  and six-arm VBench left `squeue`
  (disk 8/8 + VBench on all six). Series
  `t2v_moviegen_coinc_8v`. No GPU until
  the user picks. Do not letter n=2.
  Do not launch 128. Prior closed:
  Pwarp amp leftover n=2 **HARVESTED**
  17172470–483 COMPLETED 0:0. All five **NO**.
  Do not letter n=2. Do not launch 128.
  Track C MovieGen smoke **HARVESTED** 17121785–792
  COMPLETED 0:0. nwarp IQ 47.60 **NO** even as
  smoke (SF host). pwarp no Dyn lift. Do not launch
  128. Wan-extend N=8 **DONE / NO** 17095709–711.
  Caption pwarp **DONE / NO** 17058386–393.
  Caption nwarp **DONE / NO** 17028867–876.
  Mix+ctx / FIFO+tscore / leftover / LPIPS+FVD /
  schedule8 **DONE / NO**. Do not remake cite-128.
  Do not start 8-GPU DMD. **No I2V. No TTC.**
- **VBench 5 s windows (DONE 16009916):** hybrid 32. Aes 0.651→0.538,
  IQ 72.9→68.1 (do-nothing). Search does not reverse it. Dynamic
  median 0 every window. Full clip stays official.
  `paper_tables/2026-08-19_wan_i2v_bon32_vbench_trend.md`
  All 7 dims in one grid:
  `paper_tables/2026-08-19_wan_i2v_bon32_vbench_alldims.md`
- **VBench 16v 5 s vs 30 s + first16/last16 (DONE 16010032):**
  **Cite entire clips:** 5 s full vs 30 s full, subject 0.932→0.842.
  `paper_tables/2026-08-19_wan_i2v_notta16_vbench_fullclip.md`
  16-frame VBench does **not** copy handpicked sharp/motion. Only
  aesthetic Δrel matches “30 s worse” (−11.5% vs +1.8%).
  `paper_tables/2026-08-19_wan_i2v_notta16_vbench_headtail.md`
  Read: `paper_tables/2026-08-19_wan_i2v_vbench_windows_read.md`.

## 4. Daily-log template

When creating a new `experiment_outputs/YYYY-MM-DD.md`, use this header:

```markdown
# Experimental Outputs — YYYY-MM-DD

This file accumulates every cluster output the user pasted on this date,
plus 1-3 line interpretation of each. Raw blocks are preserved verbatim
so future agents can re-analyze.

---

## HH:MM — <short title>

**Source:** <user paste / cluster command that produced it>
**Run:** <jobID or series_name if known>

```
<RAW OUTPUT HERE>
```

**Interpretation:** <1-3 lines>
**Action taken:** <what we did with this data>

---
```

## 5. Subagent push-template (copy-paste-ready)

When pushing local changes via a `shell` subagent, use this prompt skeleton
(fill in the file list and commit message):

```
You are running shell commands on macOS to push <N> files to the GitHub repo
https://github.com/FifthEpoch/longcat-video-tta.git on the `main` branch. The
user's local clone at /Users/macrohard/Desktop/longcat-video-tta is on iCloud
Drive and chronically hits UF_DATALESS / ETIMEDOUT errors during git ops, so
you MUST do everything inside /tmp.

Files to push:
  1. <relative/path/from/repo/root>
  2. ...

Steps:
1. WORK=$(mktemp -d -t longcat-push-XXXXX) && cd "$WORK" && \
   git clone --depth=1 https://github.com/FifthEpoch/longcat-video-tta.git repo && \
   cd repo
2. mkdir -p <dirs needed> && \
   cp /Users/macrohard/Desktop/longcat-video-tta/<file1> <file1> && \
   cp ...
3. git status (confirm only expected files appear)
4. git add <file list>
5. Write commit message to /tmp/msg.txt with cat <<'MSGEOF' (do NOT inline
   heredoc into `git commit -m "$(cat <<...)"` — the wrapper shell breaks it)
6. git commit -F /tmp/msg.txt
7. git push origin main
8. git log -3 --pretty=format:"%h %s"
9. Cleanup
```

## 6. Where the user is in the meeting cycle

- **Mondays:** weekly recap meeting with PhD partner (next: today, 2026-06-01).
  Each Monday a fresh `weekly_recap_YYYY-MM-DD.md` is generated.
- **Thursdays/Fridays:** PI updates as needed.
- **Paper deadline (target):** CVPR 2027 submission window (~Nov 2026).

## 7. Artificial Individuality — experiment implementation (LIVE)

**Version:** 2026-09-23. Incorporated 2026-09-26. Path corrected the same day.

**Two directories, one cluster.**

| Role | Path |
|---|---|
| Cluster context (this repo) | `/scratch/wc3013/longcat-video-tta` |
| Live project | `/scratch/wc3013/artificial-individuality` |

Agents that already know this cluster keep using that knowledge:
`wc3013@torch`, `docs/CLUSTER_SBATCH_GUIDE.md` (account flag, `/scratch`,
conda/`PYTHONHOME`), `sbatch` only, no training on a login node. New
code, manifests, graphs, schedules, checkpoints, Slurm scripts, and
result logs are written only under
`/scratch/wc3013/artificial-individuality`. Do not create
`individuality_experiment/` inside this repo.

**Worker.** On a Torch login node, after `agent login`, start one
worker that can see both trees:

```bash
agent worker start --name torch \
  --worker-dir /scratch/wc3013/longcat-video-tta \
  --worker-dir /scratch/wc3013/artificial-individuality
```

Read this file before editing. Implement §7 in the artificial-individuality
tree. Leave the video-generation line frozen.

**Hardware:** 1× NVIDIA H200 per job, ≤48 h wall time. Request
≤47:30. Checkpoint at least once per epoch and support resume.
**SSH from the laptop:** `wc3013@torch` only.

### Core causal chain

`Visual history → learned representation → associative structure → perception under ambiguity → imagination → creative behavior`

Test one arrow at a time. Competence is measured separately from
individuality/creativity. Matched competence is not assumed.

### Scientific invariants

- All designed observers receive the same underlying image multiset. The current pilot does **not** use partial experience.
- The manipulated variable is temporal organization / relational adjacency of experience.
- Architecture, initialization, exposures, optimizer family, LR schedule, augmentation policy, and update count remain fixed.
- First pass: one run per trajectory. Replicate only if a measurable effect appears.
- Track competence separately and test for competence–individuality / competence–creativity trade-offs.

### Experiment 1 — visual history → learned representation

**Hypothesis.** Different temporal organizations of the same visual
experiences cause initially identical models to develop systematically
different internal representations.

**Observers.**

| Observer | Trajectory | Relation emphasized |
|---|---|---|
| I_IID | globally shuffled | none deliberately imposed |
| I_Tax | taxonomy/commonality | same kind / category / superclass |
| I_Rel | thematic/relational | co-occurs / participates in same scene, event, function |
| I_Struct | structural/analogical | similar visual structure despite semantic distance |

**Default learner.**

- DINO, ViT-S/16, trained from a shared initialization checkpoint θ0.
- 100 epochs, AdamW.
- Base LR 5e-4 at batch 256, scaled linearly with actual batch.
- Warmup 10 epochs; cosine LR to ~1e-6.
- Weight decay 0.04 → 0.4; teacher momentum starts ~0.996 and moves toward 1.
- Standard DINO 2 global + 8 local crops.
- BF16 on H200 if stable; otherwise official AMP/FP16.
- Save epochs 0, 10, 25, 50, 75, 100 and a resumable latest checkpoint.

**Dataset.**

- Fixed ImageNet-100 or a fixed 100-class ImageNet-derived subset.
- Freeze class and image manifests before training.
- All observers see identical image identities and exposure counts.
- Common held-out natural validation set X for every observer.

**Experience graphs.** All observers share node set V. Designed
histories differ in edges.

`V_IID = V_Tax = V_Rel = V_Struct`

`E_Tax ≠ E_Rel ≠ E_Struct`

- **Taxonomy graph:** derive from WordNet/BREEDS hierarchy. Example weight: `exp(-hierarchy_distance/τ)`.
- **Relational graph:** derive from Visual Genome or comparable scene-graph/co-occurrence data; map labels to ImageNet concepts. Default weighted association is PPMI: `w_Rel(a,b) = max(0, log(P(a,b)/(P(a)P(b))))`.
- **Structural graph:** use a frozen external vision encoder for class centroids and independent semantic similarity. Normalize both and score `S_V(i,j) = cos(μ_i, μ_j)`, `A(i,j) = S_V(i,j) - λ S_S(i,j)`. Start `λ=1` after normalization. Prefer high visual similarity and low semantic similarity.

**Schedule generation.** Each macro-epoch contains every training image
exactly once. Only order and local batch composition change.

1. Build deterministic image lists per concept.
2. Traverse each designed graph along high-weight edges to create a concept path.
3. Group the path into sustained episodes spanning multiple minibatches.
4. Shuffle images inside each episode deterministically.
5. Rotate the starting point across epochs so concepts do not always occur at the same LR position.
6. IID observer globally shuffles all images each epoch.
7. Save schedule JSONL files or reproducible hashes.

`multiset(D_IID^e) = multiset(D_Tax^e) = multiset(D_Rel^e) = multiset(D_Struct^e)`

**H200 protocol.**

- Benchmark 1,000 steps at batch 64/128/256 first.
- Select the largest stable batch and scale LR linearly.
- Log software versions, git commit, dataset/graph/schedule hashes, and the full resolved config.

**Evaluation.** Extract features at several layers (ViT blocks 3/6/9/12)
and checkpoints.

- Linear CKA; `d_CKA = 1 - CKA`.
- Representational similarity matrices; compare with Spearman correlation.
- Top-k nearest-neighbor overlap (default k=20 at image level).
- Layerwise and training-time divergence curves.

**Competence is a separate axis.**

- Frozen-backbone k-NN accuracy.
- Frozen-backbone linear probe.
- Optional generalization/transfer set.
- Plot competence versus representational/relational divergence. Do not assume they are equal.

**Replication plan.**

- Stage A: four exploratory runs (IID, Tax, Rel, Struct).
- Stage B: if an effect exists, rerun IID plus the two most divergent designed trajectories with new seed(s).
- Primary confirmatory quantity: `Δ_history = E[d | different trajectory] - E[d | same trajectory, different seed]`.

### Experiment 2 — learned representation → associative structure

- Build concept centroids from held-out images for each observer.
- Retrieve top-k concept neighbors and form learned graph G_hat_i.
- Compare G_hat_i to each designed graph using edge recovery, adjacency correlation, Jaccard, and label-permutation nulls.
- Key prediction: `sim(G_hat_i, G_i) > sim(G_hat_i, G_j)` for `i ≠ j`.

### Experiment 3 — associative structure → perception under ambiguity

- Select concept pairs on which observer graphs disagree.
- Build morph / cue-conflict / validated ambiguous stimuli.
- Read out interpretation with nearest-centroid or the same frozen probe.
- Estimate each observer’s switch point `α_i*` along the ambiguity continuum.
- Test whether boundary shifts follow learned association differences.

### Experiment 4 — perception under ambiguity → imagination

Recommended first implementation: freeze observer backbones and attach
matched inpainting/reconstruction decoders. Do not train large generators
from scratch.

- Same decoder architecture, init, and training budget for every observer.
- Use underdetermined masks compatible with multiple completions.
- Sample multiple completions per stimulus.
- Measure completion semantics, diversity, between-observer distribution shift, and separate reconstruction competence.

### Experiment 5 — imagination → creative behavior

- Same constrained concept-combination tasks for all observers.
- Multiple samples under a matched sampling budget.
- Measure competence/validity, novelty, diversity, and observer-specificity.
- Plot creativity/diversity against competence. Test for a Pareto frontier. Do not assume matched competence.

### Minimum first milestone

Deliver these, in order, before any full 100-epoch training:

1. Pinned environment.
2. Frozen ImageNet-100 manifests.
3. Tax / Rel / Struct graph JSON plus diagnostics.
4. Deterministic schedule generator plus equality tests (`multiset` equality across the four trajectories).
5. Shared θ0 checkpoint.
6. DINO training script with H200 Slurm resume support.
7. Throughput report (1,000 steps at batch 64/128/256).
8. CKA / RSA / neighbor analysis.
9. k-NN / linear-probe competence evaluation.
10. Tiny end-to-end dry run for all four trajectories.

Do **not** add generative or creativity stages until Experiment 1 and
then Experiment 2 pass their decision gates. One run per trajectory until a measurable
effect appears. Record hashes, the git commit, and the resolved config
with every job. Write those records under
`/scratch/wc3013/artificial-individuality`, not into
`sweep_experiment/` in this repo.
