# Intervention atlas: frozen-model and test-time families (2026-10-05)

Zero-GPU compilation. No new runs. Every number is copied from the source
file in the last column. Where a source compares an N=8 arm against an N=32
host, this file re-states the delta against the **paired** 8-video host where
one exists, and says so.

## How to read this

**Class**

- **Edit**: changes the trajectory the model would have produced: weights or
  activations (AdaSteer, LoRA), noise, schedule, sampler, KV write, pixels or
  latents inside a chunk.
- **Select**: draws several of the model's own unmodified futures and keeps
  one. Gated variants that never fire produce the do-nothing pixels.
- **Other**: explained in the row.

**Baselines (do-nothing)**

| Host | Protocol | Do-nothing | Source |
|---|---|---|---|
| LongCat 13.6B, std | Panda V2V, 14 cond to 28 gen, N=999 | FVD 154.7, PSNR 17.93, IQ 64.9, Subj 0.907, Dyn 0.565, Flick 0.976 | `sweep_experiment/reports/paper_tables/2026-06-08_headline_1000v.md` |
| LongCat 13.6B, long | Panda V2V, 14 cond to 93 frames, N=999 | FVD 278.7, PSNR 12.769 (no IQ) | `weekly_recap_2026-06-01.md` §1.1 |
| LongCat, OOD preview | segment pool, N≈1000 | FVD 81.22 | `paper_tables/2026-07-31_fvd_bootstrap_ci_1000v.md` |
| Wan SF, cite-128 | caption V2V, 30 s, n=128 | IQ 72.07, Subj 0.666, Dyn 32.8%, PSNR 9.25, FVD 410.3 | `paper_tables/2026-09-04_drop_pseudo_next_territories.md`, `..._cite128_lpips_fvd.md` |
| Wan SF, caption N=32 | caption V2V, 30 s | IQ 71.54, Subj 0.700, Flick 0.989 | `paper_tables/2026-08-25_wan_v2v_caption_always_adasteer.md` |
| **Wan SF, paired first-8** | same 8 clips the N=8 arms use | **IQ 70.62, Subj 0.658** (identity row `sf_tscore`) | `paper_tables/2026-09-04_wan_v2v_caption_fifo_tscore_harvest.md` |
| **Wan RF, paired first-8** | same 8 clips | **IQ 66.70, Subj 0.658** (identity row `rf_tscore`) | same |
| Wan SF, stem N=32 | stem-prompt V2V (old protocol) | IQ 69.65, Subj 0.665 | `paper_tables/2026-08-24_wan_v2v_sf_family32_verdict.md` |
| Wan-teacher / T2V smokes | n=2 | per row | per row |

IQ = VBench imaging quality on a 0–100 scale. LongCat tables store it as 0–1, so
0.649 is written as 64.9 here. Dyn = VBench dynamic degree (share of clips, or
mean for LongCat). "—" means the metric was not computed.

**Protocol flags** in the last-but-one column:
**A** = at-scale (N≥128). **B** = N=32. **C** = N=8, compared against the
paired host. **C\*** = N=8, and the source compared it against the unpaired
N=32 host; this file restates the delta against the paired host. **D** = n=2
smoke; the source says do not draw a conclusion. **X** = no valid number.

## Atlas

### LongCat-Video 13.6B (continuation; Panda-70M; PSNR/FVD available)

| # | Family | Class | Protocol, n | Δ vs do-nothing | Flag | Source |
|---|---|---|---|---|---|---|
| 1 | AdaSteer (timestep-embedding δ, S10 LR5e-3) | Edit | std 28f, 999 | FVD −1.3 (154.7→153.4), PSNR +0.01, IQ 0.0, Subj 0.000, Dyn +0.003, Flick 0 | A | `paper_tables/2026-06-08_headline_1000v.md` |
| 1b | AdaSteer, long horizon | Edit | 93f, 999 | FVD +5.4, PSNR +0.018, FID −0.4; IQ — | A | `weekly_recap_2026-06-01.md` |
| 2 | LoRA-R8 TTA | Edit | std, 999 | FVD +3.2, PSNR −0.08, **IQ −3.4**, Aes +0.047, Dyn +0.031, Subj −0.005 | A | `2026-06-08_headline_1000v.md` |
| 2b | LoRA-R8, long horizon | Edit | 93f, 999 | FVD +3.7, PSNR −0.035 | A | `weekly_recap_2026-06-01.md` |
| 3 | TinyLoRA (SVD-scalar) BARE-R2 / TIED-R2 | Edit | std, 999 | FVD −0.5 / +6.4, PSNR +0.01 / 0.00, IQ 0.0 / 0.0 | A | `2026-06-08_headline_1000v.md` |
| 3b | TinyLoRA LAST24, long horizon | Edit | 93f, 999 | FVD −0.1, PSNR +0.004 | A | `weekly_recap_2026-06-01.md` |
| 4 | Retrieval-augmented batch TTA (AdaSteer + K neighbours) | Edit | std, 999 | FVD +1.0 to +7.4, PSNR −0.03 to −0.06, **IQ −3.4**, Aes +0.046 | A | `paper_tables/2026-07-05_panda_1000v_retrieval.md` |
| 5 | AdaSteer budget grid (12 fixed step×LR configs) | Edit | OOD preview, ~1000 | PSNR spread 0.11 dB; fixed S10_LR5e3 FVD +3.15 [−5.79, +12.19] (null); IQ — | A (mixed pre/post-fix code, see Task 1 note) | `paper_tables/2026-07-19_budget_grid_1000v_preview.md`, `2026-07-31_fvd_bootstrap_ci_1000v.md` |
| 6 | Per-video config router (12 configs + skip) | Other: selects among *edited* outputs | OOD preview, 898–1000 | best router FVD −0.51; others +0.97 to +1.45; GT oracle −8.94 (not deployable) | A | `paper_tables/2026-07-28_router_fvd_1000v.md` |
| 7 | AdaSteer variants and add-ons (Delta-B/C, FiLM, norm-tune, full-model TTA, early stopping, augmentation, CLIP gate, anchor gate/reg) | Edit | discovery N≈100 | not compiled on a shared table | X (discovery only) | `sweep_experiment/reports/INDEX.md` (discovery rows) |
| 8 | Streaming δ re-fit, native ~60 s rollout (two recipes) | Edit | 8 videos × 12 chunks | GT-free drift: NULL (p≥0.26; clean target p≥0.53); IQ — | C (different metric) | `INDEX.md` long-horizon rows |
| 9 | Best-of-4 with GT-free drift verifier | Select | 8 × 12 chunks | +0.833 dB PSNR vs random pick on 11 GT chunks; drift vs NOTTA not significant | C (different metric) | `INDEX.md` long-horizon rows |
| 10 | Pathwise TTC (noise/trajectory gradient) | Edit | smoke | first run decoded noise (Euler sign bug); no valid result | X | `ANALYSIS_LOG.md` 2026-08-14 |
| 11 | SAVi-DNO (noise optimisation) | Edit | 10-video sanity | optimised ≈ not optimised, both PSNR ~7.2 (sampler bug); no valid result | X | `ANALYSIS_LOG.md` 2026-07-19 / 07-20 |

### Wan2.1-1.3B Self Forcing (SF) / Rolling Forcing (RF) (30 s; no ground truth past the opening; VBench only, plus PSNR/FVD on cite-128)

| # | Family | Class | Protocol, n | Δ vs do-nothing | Flag | Source |
|---|---|---|---|---|---|---|
| 12 | AdaSteer on Wan (fixed / stream / resid) | Edit | caption, 8 | **IQ −27.95 / −19.14 / −52.87**; Subj −0.026 / −0.040 / +0.017 | C\* | `paper_tables/2026-08-25_wan_v2v_caption_always_adasteer.md` |
| 13 | Always-search, best-of-4 every chunk | Select | cite-128 | IQ +0.12, Subj −0.005, Dyn +18.0 pp, PSNR −0.04, LPIPS +0.006, FVD +14.9 | A | `2026-09-04_drop_pseudo_next_territories.md`, `2026-09-04_wan_v2v_cite128_lpips_fvd.md` |
| 14 | Pseudo-future search (prefix hold-out gate + best-of-4) | Select | cite-128 | IQ +0.31, Subj −0.006, Dyn +14.9 pp, PSNR −0.03, LPIPS +0.008, FVD −4.9 | A | same |
| 15 | Rewind (re-roll a frozen chunk, keep if it moved more) | Select | caption 32; stem 32 | IQ −0.65, Subj −0.002 (caption); IQ −0.21 (stem) | B | `2026-08-25_..._always_adasteer.md`; `2026-08-24_wan_v2v_sf_family32_verdict.md` |
| 16 | Sick-search (k=4 only after a freeze) | Select | stem 32 | IQ −0.52, Subj +0.003, tail −1% | B (stem protocol) | `2026-08-24_wan_v2v_sf_family32_verdict.md` |
| 17 | Prefix-match / appearance pick | Select | stem 32 | IQ −0.77, Subj +0.040, tail −18%, Dyn 0 | B (stem protocol) | `2026-08-24_wan_methods_since_switch.md` §5.0c; `2026-09-04_failure_modes_plain.md` §2 |
| 18 | Freeze-score self-grade (1.3B lock score); gated / always-on | Select | caption 8 | gated = identity on both hosts; always: RF IQ −0.80, SF IQ +0.02; Subj RF +0.007, SF −0.005 | C | `2026-09-04_wan_v2v_caption_fifo_tscore_harvest.md` |
| 19 | CachedSearch / re-gate (cheaper search) | Select | caption 8 | same pixels as the uncached twin; IQ −0.79 (always) / −0.85 (re-gate) vs paired SF-8; wall time up | C\* | `2026-08-31_wan_v2v_pseudo_next8_harvest.md` |
| 20 | I2V best-of-4 with handcrafted verifier (always / gated) | Select | I2V stills, 32 | full clip IQ +0.04 / −0.05 (tie); last 5 s IQ −1.8 | B (I2V, discovery only) | `2026-08-24_wan_methods_since_switch.md` §4; `failure_modes_plain.md` §9 |
| 21 | Best-of-4 on T2V and on the Wan teacher (always / gated) | Select | n=2 each | T2V SF: IQ +0.59 / −1.00. Teacher V2V: −0.71 / 0.00. Teacher MovieGen: **−2.10** / 0.00 (always lost a Dyn clip) | D | `2026-09-07_t2v_moviegen_warp_smoke.md`; `2026-09-07_wan_teacher_smoke_harvest.md` |
| 22 | Mid-chunk rewrite (intra redraw, lastmix, nudge, next-seed, wiggle, latent-motion pick; gated / always) | Edit | caption 8 | SF vs paired 70.62: intra −2.43, lastmix −0.99, nudge −0.96/+0.25, nextseed −0.51/−0.81, wiggle +0.03/−2.51, latmot −0.59/−0.87; Subj −0.012 to −0.031. RF vs paired 66.70: intra −0.37, lastmix −1.17, nudge +0.40/−0.09, wiggle +0.03/+0.47, latmot +0.15/−0.92; Subj −0.013 to +0.001 | C\* | `2026-08-31_wan_v2v_keep_intra_closed.md`; `2026-08-29_wan_v2v_lastmix8_harvest.md` |
| 23 | Crossed host / sampler mix (gated / always) | Edit | caption 8 (stem 32 for crossed host) | rf_mix +1.60 / +1.01 vs RF-8; sf_mix −0.11 / +0.12 vs SF-8; Subj RF −0.009 / −0.026, SF +0.002 / +0.003; always-on RF mix flicker 0.978, Dyn 8/8. Stem crossed host: flicker 0.972, Dyn 32/32 (jitter) | C\* (mix); B stem (cross) | `2026-09-04_wan_v2v_caption_mixctx_harvest.md`; `failure_modes_plain.md` §5 |
| 24 | Noise-schedule list (linger-high / dump-early) | Edit | caption 8, RF | IQ −0.36 / +1.44 vs RF-8; Subj +0.010 / 0.000 | C\* | `2026-09-04_wan_v2v_caption_schedule8_harvest.md` |
| 25 | Noise re-injection: leftover ρ (lo / hi / adapt) and `look` | Edit | caption 8, RF | IQ +1.39 / −2.26 / +0.77; look +2.81 vs RF-8; Subj −0.005 / −0.028 / −0.028 / +0.008; flicker to 0.976 on hi | C\* | `2026-09-01_wan_v2v_caption_leftovers_harvest.md` |
| 26 | Context noise in the KV write (t=50) | Edit | caption 8 | RF ≈ +0.9 vs RF-8; SF ≈ −1.4 vs SF-8; SF clip 0004 tail 0.190 (explosion) | C\* | `failure_modes_plain.md` §6; `..._mixctx_harvest.md` |
| 27 | FIFO lookahead (all / sick-only) | Edit | caption 8, RF | IQ +1.53 / +0.05 vs RF-8; Subj +0.001 / 0.000 | C\* | `2026-09-04_wan_v2v_caption_fifo_tscore_harvest.md` |
| 28 | Extra attention sink / KV pin (incl. replay-sink, LongLive sink) | Edit | stem 32 (SF) | IQ +0.33, Subj −0.020, flicker 0.977, tail +72%. Replay without re-index = identity. LongLive sink 9: IQ failed (no number in source) | B (stem) | `2026-08-24_wan_v2v_sf_family32_verdict.md`; `failure_modes_plain.md` §7 |
| 29 | Noise warp, extras only (nwarp, always / live) | Edit | caption 8; T2V n=2; teacher n=2 | caption **−21.44 / −16.20** vs SF-8, Subj −0.064 / −0.030; T2V −23.33; teacher V2V −21.35, MovieGen −18.37 | C\* / D | `2026-09-06_wan_v2v_caption_nwarp_harvest.md`; T2V and teacher smoke files |
| 30 | Pred-slide (pwarp) and amplified variants | Edit | caption 8; T2V n=2; teacher n=2 | caption **−3.81** vs SF-8 (both arms), Subj −0.030 / −0.007; T2V −1.58; teacher −0.08 / −0.20; amplified ramp/crop/early/mag −0.17 to +0.05; **persist −36.66** | C / D | `2026-09-06_wan_v2v_caption_pwarp_harvest.md`; `2026-09-08_pwarp_amp_harvest.md` |
| 31 | Fast-weight write (coincidence gate, write-every, Titans, mean-delta) and KV window | Edit | T2V MovieGen, 8 | FW arms **IQ −16.66 to −16.92**, Subj −0.33, flicker −0.08, Dyn 8/8 (jitter); KV window IQ −0.49, Subj −0.010 | C | `2026-09-22_t2v_coinc8_quality.md` |
| 32 | Guidance / shift tweaks (CFG, timestep shift) | Edit (nominal) | V2V probe, n=2 | all 9 (shift, CFG) cells gave identical pixels: no-op on the DMD student | D | `ANALYSIS_LOG.md` 2026-08-20 |
| 33 | Prefix-protected fast weights (protect / update / fork) | Edit | — | spec marked submit-ready, never launched | X | `2026-09-20_t2v_pprot8_spec.md` |

Count: **36 rows, 33 numbered families** (1b/2b/3b are a second horizon for
the same family). Edit 22 families, Select 10, Other 1 (the router). Four
families (7, 10, 11, 33) have no usable number.

## Does "editing costs image quality in every family; selection keeps it" hold?

**Not as stated. A weaker version does hold.**

What holds:

- Every IQ collapse larger than about 3 points comes from an edit family: AdaSteer
  on Wan (−19 to −53), nwarp (−16 to −23), fast-weight writes (−17), pwarp persist
  (−37), caption pwarp (−3.8), and LoRA and retrieval on LongCat (−3.4 each).
- No selection family at n≥8 loses more than 0.85 IQ. At scale (cite-128),
  always-search and pseudo-search *raise* IQ (+0.12, +0.31).

Counterexamples on the edit side (edits that cost no IQ):

1. **AdaSteer on LongCat, 999 videos:** IQ 64.9 → 64.9. TinyLoRA BARE and TIED:
   IQ unchanged. These are the only at-scale edits with IQ, and they are
   neutral. On this host, editing does nothing measurable either way.
2. **Extra sink on SF, N=32:** IQ +0.33. The cost appeared in subject and flicker,
   not IQ.
3. **Pwarp on the Wan teacher** and its amplified crops: within ±0.2 IQ (n=2).
4. **Several Rolling-host edits are positive against the paired 8-video host**
   (66.70): `look` +2.81, rf_mix +1.60, FIFO +1.53, dump +1.44, ρ-lo +1.39. Their
   source files reported −0.7 to −3.9 because they compared N=8 arms with the
   **N=32** Rolling host (70.22). The 8 clips used are harder: RF-8 = 66.70 and
   SF-8 = 70.62, against 70.22 and 71.54 at N=32. Several SF-side mid-chunk
   rewrites are also near zero against SF-8 (nudge-always +0.25, wiggle +0.03).
   So `failure_modes_plain.md` overstates "the photo dies" for noise-list,
   mix, FIFO and mid-chunk families. What those families reliably cost is
   **subject consistency**: most SF-side rewrites land at 0.627–0.646 against
   0.658.

Counterexamples on the selection side (selection that costs IQ):

1. **Always-on freeze-score on RF:** −0.80 against the paired host, and it loses 5 of 8 clips on tail.
2. **CachedSearch / re-gate:** −0.79 / −0.85 against SF-8.
3. **Prefix-match:** −0.77 (stem N=32). **Rewind:** −0.65 (caption N=32).
4. **Best-of-4 on the Wan teacher, MovieGen:** −2.10, and it lost a Dyn clip (n=2). **Gated best-of-4 on T2V:** −1.00 (n=2).
5. **I2V always-search:** −1.8 on the last 5 s. The full clip is a tie.

**Restated claim the table supports:** large image-quality damage appears only
when the trajectory is edited, and selection never produces it. But editing
does not always cost IQ (LongCat AdaSteer, TinyLoRA, sink, pwarp-teacher, and
several paired-8 Rolling edits). Selection also has a small IQ cost of up to
about 1 point at n≥8. Most edit and selection rows below 128 videos are N=8
or n=2. The only at-scale selection evidence is cite-128, two families.

## Comparability warnings

- LongCat rows (FVD/PSNR on ground-truth continuation) and Wan rows (VBench
  on 30 s with no ground truth) are different tasks, models and metrics. Do not
  rank across the two blocks.
- Stem-prompt rows (15–17, 28, crossed host) are an older protocol than caption
  rows. The SF baseline moved from 69.65 to 71.54 when captions replaced stems.
- C\* rows: the deltas here are against the paired first-8 host, taken from the
  identity rows `sf_tscore` / `rf_tscore` (pixels identical to do-nothing on all
  8 clips). The `keep_intra`, `lastmix` and `pseudo_next8` files label the N=32
  host "same 8". It is not the same 8 clips.
- LongCat AdaSteer, LoRA and retrieval at standard horizon ran on code that
  adapted on about 75% of the target latents (hold-out bug, see the Task 1
  audit). TinyLoRA still does. Their deltas are for that code.
- Row 5's grid on the OOD preview pool mixes configs run at different times
  relative to the 2026-07-14 fix (`ANALYSIS_LOG.md` 2026-07-14, 2026-07-19).
  Treat it as indicative.
