# Related-work and novelty scan: "Where do videos go to die?"

Scan date: 2026-10-05. Target venue: CVPR 2027 (deadline 2026-11-16).
Method: about 50 web searches (standard and extended), plus abstract or
full-text reads of the closest hits (arXiv abs/html, CVF, OpenReview,
PMC). Coverage: arXiv through 2026-10-03, CVPR/ICCV/ECCV/NeurIPS/ICLR/ICML
2024-2026 as indexed, and workshop papers where search surfaced them. Search
engines index brand-new arXiv postings unevenly, so do one more arXiv
listing sweep (cs.CV, cs.LG) in the week before submission.

Confidence labels: **[read]** means the abstract or text was fetched and
checked. **[snippet]** means the claim rests on search-result text only.

---

## 1. Verdict

**No paper found does what we do.** No paper found treats streaming /
autoregressive video diffusion generators (Self Forcing, Rolling Forcing,
LongLive on Wan2.1-1.3B) as stochastic dynamical systems and measures their
attractor type, prompt-memory half-life, twin-run perturbation growth,
bifurcation under a design knob, RQA against surrogates, or
critical-slowing-down signals. The dynamical-systems vocabulary does appear
in nearby work, but each use has a narrower scope:

- LLMs: attractor cycles under iterated paraphrasing (ACL 2025).
- Iterated text-to-image-to-text loops: all prompts converge to about 12
  "visual attractors" (Patterns 2025).
- Neural PDE and weather emulators: FTLE, Jacobian eigenanalysis, failure
  regimes of year-long rollouts.
- Denoising itself (Equilibrium Forcing).

Nobody has applied it to long-horizon streaming video.

**What is not novel:** the phenomena themselves. Drift, color/saturation
drift, over-exposure, motion stagnation, prompt non-adherence, and "sink
helps stability" are reported in dozens of 2025-2026 method papers. Three
papers come close to individual findings:

- **LoL (2601.16914)** shows sink-collapse at the *same frame indices
  regardless of prompt and noise* in LongLive and Self-Forcing++.
- **2607.27036** ties drift onset to an *abrupt* effective-rank collapse.
- **Meta-ARVDM (2503.10704)** formalizes "history forgetting" and "temporal
  degradation".

A reviewer will say "drift is well known." They should not be able to say
"this analysis was already done," as long as the paper presents the
dynamical characterization, not the existence of drift, as the
contribution.

**Main risks to manage:**
1. LoL's sink-collapse in **LongLive**. We call LongLive "stationary /
   alive". LoL reports periodic reversion to the sink frame at fixed latent
   indices (for example 132 and 201, roughly 33 s and 50 s at Wan's 4x
   temporal compression and 16 fps), which falls inside our 180 s rollouts.
   We must address it explicitly; our RQA should show these recurrences if
   they occur.
2. The **bifurcation** claim (item 4) overlaps heavily with existing sink /
   window ablations (LongLive, Rolling Forcing, Deep Forcing, FreqForcing,
   Rolling Sink). The *fact* that a sink prevents SF drift is known. Only
   the *qualitative attractor-type switch* (global to prompt-anchored),
   measured with our observables, is new.
3. The negative CSD result (item 5) has a partial theoretical precedent
   ("Entropy Collapse", 2512.12381, argues early-warning signals fail for
   generative-model collapse). That paper is about training loops, not
   rollouts, so our result remains new for rollouts.

---

## 2. Closest works, ranked

Overlap columns refer to our items:

1. Attractor type
2. Prompt half-life
3. Twin / Lyapunov
4. Bifurcation
5. RQA / CSD
6. Real-video control

Key: S = substantial, P = partial, - = none.

| Rank | Paper | Venue / date | 1 | 2 | 3 | 4 | 5 | 6 | Why it matters |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **LoL: Longer than Longer, Scaling Video Generation to Hour** (Cui, Wu, Li, Yang, Li, Wang, Bai, Ban, Hsieh) [read] | arXiv 2601.16914, Jan 2026 | P | - | - | P | P | - | Sink-collapse in LongLive and Self-Forcing++: content reverts to sink frame at identical latent indices regardless of prompt or noise. Measured as a drop in L2 distance to initial frames. Attributed to RoPE phase alignment across heads. Framed as structural, not as an attractor. |
| 2 | **Mitigating Compounding Error via Video Representation Regularization** (T. Chen, Q. Zhang, Y. Wang) [read] | arXiv 2607.27036, Jul 2026 | P | - | - | - | P | - | Drift onset coincides with a *sharp* effective-rank collapse of DiT hidden states. Diffusion Forcing / world models, not SF/RF/LongLive. Gives an internal early indicator, which contrasts with our "slow ramp, no early warning". |
| 3 | **Error Analyses of Auto-Regressive Video Diffusion Models: A Unified Framework (Meta-ARVDM)** (J. Wang, Zhang, Li, Tan, Pang, Du, Sun, Yang) [read] | arXiv 2503.10704, Mar 2025 (rev. Dec 2025) | - | P | - | - | - | - | Theory of "history forgetting" (conditional MI with past frames) and "temporal degradation" (cumulative per-step error). Needle-in-haystack tests in DMLab / Minecraft. No prompt half-life, no attractor. |
| 4 | **FreqForcing: AR Long Video Generation via Spectral Self-Anchoring** (Li, Liang, Kong, Zhang) [read] | arXiv 2607.27110, Jul 2026 | P | - | - | P | - | - | Drift appears as low-frequency / DC spectral energy drift on Self Forcing. A larger sink *slows* the drift but does not stop it. Overlaps our spectral-slope observable and the sink knob. |
| 5 | **Deep Forcing** (2512.05081) and **Rolling Sink** (2602.07775) [snippet] | arXiv Dec 2025 / Feb 2026 | - | - | - | P | - | - | Sink-size sweeps on Self Forcing: degradation falls as the sink grows. Deep Forcing reports an optimum at 10-15 frames. This is the main prior art against item 4's "knob" claim. |
| 6 | **LongLive** (2509.22622), **Rolling Forcing** (2509.25161, ICLR 2026), **Self-Forcing++** (2510.02283) [read] | 2025-2026 | P | P | - | P | - | - | They report qualitative failure (SF degrades past 5 s; over-exposure / darkening), a ΔQuality-Drift metric (RF 0.01 vs SF 1.66), and sink / window ablations. LongLive reports new-prompt non-adherence with a KV cache. All descriptive and segment-based. |
| 7 | **Can AI Weather Models Predict Beyond Two Weeks? A Quantitative Benchmark and Analysis of Long Rollouts** (Lehmann et al., ETH) [read] | arXiv 2605.30184, May 2026 | P (method) | - | P (method) | P (method) | - | P (method) | The closest *methodological* analogue. It sorts year-long rollouts of 9 models into blow-up, drift, and loss of seasonality. Noise-injection test: stable models act as denoisers. Architecture ablations. Weather domain only. |
| 8 | **Autonomous language-image generation loops converge to generic visual motifs** (Hintze, Proschinger Åström, Schossau) [read] | *Patterns* 2025 | S (concept) | - | - | - | - | - | SDXL and LLaVA run in an iterated loop. 700 trajectories collapse to about 12 "visual attractors" independent of prompt and temperature, in explicit attractor language. Iterated T2I captioning, not video rollout. It is the conceptual twin of our SF "global end state". |
| 9 | **Unveiling Attractor Cycles in LLMs: A Dynamical Systems View of Successive Paraphrasing** (Z. Wang, Li, Yan, Cheng, Y. Zhang) [read] | ACL 2025 (long), arXiv 2502.15208 | P (concept) | - | - | - | - | - | Iterated paraphrasing settles into 2-period limit cycles that are robust to prompt, temperature, and perturbation. Text domain. |
| 10 | **Eigenanalysis framework for AR neural emulators of multi-scale chaotic dynamics** (Ainslie, Hassanzadeh, Mahoney, Chattopadhyay) [snippet] and **Comparative study of accuracy and rollout stability of temporal surrogate models** (Biswas) [read] | arXiv 2608.16084; 2605.24868 | - | - | P (method) | P (method) | - | - | Jacobian spectral radius and finite-time Lyapunov exponents for AR emulators (Kuramoto-Sivashinsky, Kolmogorov flow). These supply the Lyapunov / twin machinery for our item 3, in PDE surrogates only. |
| 11 | **Selz & Craig, AI weather models and the butterfly effect** (GRL 2023; EMS 2024) [snippet] | 2023-2024 | - | - | P (method) | - | - | - | Twin experiments on Pangu / GraphCast / FourCastNet. AI models fail to reproduce fast small-scale error growth, and perturbation growth depends on amplitude. A direct precedent for "perturbation growth vs semantic divergence". |
| 12 | **Equilibrium Forcing** (Lillemark, Rojas, Novack, Wang, Du, Ma, Berg-Kirkpatrick, Yu) [read] | arXiv 2608.14706, Aug 2026 | - | - | - | - | - | - | Uses "attractor dynamics" for the *denoising* process (data manifold as a fixed point). Shows actual frame noise level drifts from the schedule as frames accumulate. Same words, different object. |
| 13 | **ReStraV: AI-Generated Video Detection via Perceptual Straightening** (Internò, Geirhos, Olhofer, Liu, Hammer, Klindt) [read] | NeurIPS 2025 | - | - | - | - | - | P | DINOv2 trajectory curvature and stepwise distance separate real from generated (short) videos. Supports our DINOv2 observables and the real-video control, without long horizons. |
| 14 | **DySink** (2605.21028) [read], **Head Forcing** (2605.14487) [snippet], **Infinity-RoPE** (2511.20649, CVPR 2026) [snippet] | 2025-2026 | P | - | - | P | - | - | Mechanistic sink analyses: RoPE-induced inter-head consensus causes sink collapse; local / anchor / memory heads. Infinity-RoPE: slow prompt responsiveness in long rollouts. |
| 15 | **Entropy Collapse: A Universal Failure Mode of Intelligent Systems** (Khanh, Hoa) [read] | arXiv 2512.12381 | - | - | - | - | P | - | Argues that generative-model collapse is first-order, so autocorrelation / variance early-warning signals fail. About recursive *training*, not rollouts, and mostly theoretical. Partial precedent for our negative CSD result. |
| 16 | **WorldRoamBench** (2606.31672) [snippet], **StreamAV-Bench** (2608.26336) [snippet], **SNF-Bench** (2608.28694) [read] | 2026 | P | - | - | - | - | - | Benchmarks with segment-based drift and "degradation slope" metrics up to 3600 s. WorldRoamBench notes non-monotonic mid-sequence collapse. No attractor, memory-decay, or sensitivity analysis. |

Further method papers that mention drift but contain no analysis of our
kind (cite as part of the drift landscape): Steady-Forcing (2606.14732),
Reward Forcing (2512.04678), In-Distribution Forcing (2610.03120, 2 Oct
2026), Towards Error-Free Long Video Generation (2606.22370), BAgger
(CVPR 2026, 2512.12080), Stable Video Infinity (ICLR 2026 oral,
2510.09212), FLEX / Train Short Inference Long (2602.14027), Recency
Forcing (2609.19729; measures context influence vs temporal distance,
which is adjacent to item 2), TANGO test-time adaptation (2607.15849),
Pathwise Test-Time Correction (2602.05871), Stream-T1 test-time scaling
(2605.04461), Goodbye Drift / anchored tree sampling (2605.20476), DiVid
diversity collapse (2610.01661), SEGUE prompt switching (2609.38691), Mask
Forcing (2609.09123), Rollout-Marginal Distillation (2609.37925), and the
memory survey "The Past Frames the Future" (2609.28466, Chen et al.,
HKUST, Sept 2026).

The known-to-us items, checked:

- **CachedSearch (2607.23159)**: Saini, Birkbeck, Wang, Adsumilli, Bovik.
  Test-time search on non-AR Wan2.1. No drift claim. [read]
- **GEARS (2608.29322)**: SIGGRAPH Asia / TOG 2026. Its real title is
  "Test-Time Scaling for Video Diffusion Models via Diagnosis-Guided
  Candidate Recycling". It recycles and edits rejected candidates. Short
  clips, not AR. [read]
- **Reward Forcing (2512.04678)**: EMA-Sink against copying of the initial
  frame, and Re-DMD toward motion. [read]
- **Steady-Forcing (2606.14732)**: trade-off between persistence and
  motion. [read]
- **Effective rank (2607.27036)** and the **memory survey (2609.28466)**:
  above. [read]

---

## 3. Novelty status per claim

| # | Our claim | Status | Closest prior | What to say |
|---|---|---|---|---|
| 1 | Attractor type per model: SF converges to one global, prompt-independent end state (different-prompt distance 0.97 to 0.66; retrieval 100% to 49%); RF is prompt-anchored with slow decay; LongLive stays stationary. | **Mostly novel (measurement and taxonomy).** The phenomenon "SF degrades / color drifts" is known. That *all prompts converge to the same state*, measured by cross-prompt distance and retrieval, has not been shown for video. | LoL (prompt-independent sink-collapse *timing*); Patterns 2025 (prompt-independent attractors in an iterated T2I loop); Self-Forcing++ / FreqForcing (over-exposure, spectral drift). | Claim the first *quantitative cross-prompt convergence* analysis and a three-way taxonomy of attractor types for streaming video generators. **Address LoL's LongLive sink-collapse directly** (run RQA or distance-to-sink on our LongLive rollouts). |
| 2 | Prompt half-life (about 58 s for SF) with an asymptote at the same-prompt noise floor; exponential relaxation fits (τ about 360 s). | **Novel.** No paper fits a decay curve to prompt information over time, or compares it with a noise floor. | Meta-ARVDM (history forgetting as conditional MI; theory and needle tests); Recency Forcing (context influence vs distance inside attention); LongLive / Infinity-RoPE (slow prompt response). | Safe as "first measurement of prompt-memory half-life". Cite Meta-ARVDM as the conceptual neighbor: it concerns *frame history*, not the *text condition*. |
| 3 | Twin experiments: bit-deterministic reruns with one perturbed noise block; growth rate independent of perturbation size; sensitivity and semantic divergence dissociate. | **Novel for video generators.** Methodology is standard in weather and PDE emulators. | Selz & Craig (twin experiments, amplitude-dependent growth in AI weather models); Biswas 2605.24868 and Ainslie et al. 2608.16084 (FTLE / Jacobian for emulators); seed-sensitivity papers for single-shot T2V. | Present it as importing an established predictability method into video generation. Do not claim the method is new. |
| 4 | Bifurcation: one knob (sink size, local window) switches attractor type. | **Partially covered.** That a sink stabilizes SF, and how window size affects consistency, is well documented in ablations. | LongLive, Rolling Forcing, Deep Forcing, Rolling Sink, FreqForcing sink sweeps; LoL / DySink sink mechanics. | Claim only the *qualitative regime change* (global to prompt-anchored) under our observables, with the order parameter (cross-prompt distance / retrieval) traced across knob values. Avoid "we discover the sink prevents drift". |
| 5 | RQA vs phase-randomized surrogates; CSD early-warning signals are negative (slow ramp, no early warning). | **Novel for generated video.** RQA has been applied to *real* video (scene-change detection). No CSD test on generative rollouts was found. | RQA for scene change (J. Imaging 2025); Entropy Collapse 2512.12381 (EWS fail in training-loop collapse); 2607.27036 (*abrupt* rank collapse, which tensions with our "slow ramp"). | Safe. **Reconcile with 2607.27036**: their abrupt internal collapse vs our slow observable ramp. The difference could come from models (Diffusion Forcing vs SF), observables, or horizon. |
| 6 | Real-video control: real videos do not contract. | **Novel as a control in this setting.** | ReStraV (DINOv2 trajectories differ between real and generated, but only short clips); weather "stable models produce unique trajectories". | Safe. Cite ReStraV for DINOv2 trajectory statistics. |
| Suppl. | Test-time weight adaptation null at scale; atlas of about 33 interventions; search slows drift without stopping it. | **Contested.** Positive TTA / TTC papers exist (TANGO, Pathwise TTC, ID-Forcing, Stream-T1). | Those papers. | A null result *against published positive claims* must cite them and explain differences (scale, horizon, metric). |

---

## 4. Positioning

**Claims we can make (as of 2026-10-05):**

- "We treat streaming video generators as stochastic dynamical systems and
  characterize their long-horizon *asymptotic* behavior (attractor type),
  not only their short-term quality drop."
- "To our knowledge, this is the first quantitative evidence that a
  streaming video generator (Self Forcing) forgets its prompt and converges
  to a single prompt-independent end state, measured by cross-prompt
  distance and prompt retrieval."
- "We introduce the prompt half-life, the time over which prompt
  information decays to the same-prompt noise floor, and show that models
  with similar per-frame quality differ in it."
- "We adapt twin-experiment predictability analysis from numerical weather
  prediction to video generation and show that noise sensitivity and
  semantic divergence dissociate."
- "Long-horizon collapse in these models is a slow ramp without classical
  critical-slowing-down precursors."
- "Generated rollouts contract in embedding space; real videos under the
  same observables do not."

**Claims to avoid or qualify:**

- "Drift / error accumulation / color drift has not been studied." It has
  been studied extensively.
- "We discover that attention sinks prevent drift." Known from LongLive,
  Rolling Forcing, Deep Forcing, FreqForcing.
- "LongLive does not collapse" without qualification. LoL reports
  sink-collapse in LongLive. Specify horizon and observables, and report
  any sink-reversion recurrences.
- "First dynamical-systems view of generative models." Attractor views
  exist for LLMs (ACL 2025), iterated T2I loops (Patterns 2025), model
  collapse in self-consuming training (Shumailov et al. Nature 2024;
  Alemohammad et al. ICLR 2024; Fu et al. ICLR 2025), and emulators.
  Restrict the claim to *streaming video generators / AR video
  diffusion*.
- "First to use Lyapunov / twin experiments." Restrict to video
  generators.
- "Collapse always has no early warning." 2607.27036 finds an abrupt
  internal signal. Say "in output-space observables" and test effective
  rank ourselves if cheap.
- "Test-time adaptation does not work." Say "does not work at our scale
  and horizon", and engage TANGO and TTC.

---

## 5. Must-cite list

**Models and drift fixes (direct):** Self Forcing (Huang et al., 2025);
CausVid (Yin et al., CVPR 2025); Diffusion Forcing (Chen et al., NeurIPS
2024); Rolling Forcing (2509.25161, ICLR 2026); LongLive (2509.22622);
Self-Forcing++ (2510.02283); LoL (2601.16914); Deep Forcing (2512.05081);
Rolling Sink (2602.07775); Infinity-RoPE (2511.20649, CVPR 2026); DySink
(2605.21028); Head Forcing (2605.14487); FreqForcing (2607.27110);
Steady-Forcing (2606.14732); Reward Forcing (2512.04678); Stable Video
Infinity (2510.09212, ICLR 2026); BAgger (2512.12080, CVPR 2026);
In-Distribution Forcing (2610.03120); FLEX (2602.14027); Wan2.1.

**Analysis of AR video error:** Meta-ARVDM (2503.10704);
effective-rank collapse (2607.27036); memory survey (2609.28466);
Equilibrium Forcing (2608.14706); Recency Forcing (2609.19729).

**Benchmarks:** VBench / VBench-Long; WorldRoamBench (2606.31672);
StreamAV-Bench (2608.26336); SNF-Bench (2608.28694); DiVid (2610.01661).

**Test-time interventions:** Video-T1 (2503.18942); Stream-T1
(2605.04461); TANGO (2607.15849); Pathwise TTC (2602.05871); CachedSearch
(2607.23159); GEARS (2608.29322); Goodbye Drift (2605.20476); test-time
training for one-minute video (Dalal et al., CVPR 2025; verify the
citation).

**Attention sinks:** StreamingLLM (Xiao et al., ICLR 2024); When Attention
Sink Emerges (Gu et al., ICLR 2025); The Spike, the Sparse and the Sink
(2603.05498, ICML 2026).

**Dynamical-systems analogues:**

- LLMs: Attractor Cycles in LLMs (ACL 2025); Learning to Break the Loop
  (Xu et al., NeurIPS 2022, the self-reinforcement of repetition);
  Holtzman et al. (ICLR 2020, degeneration).
- Iterated T2I: Hintze et al. (Patterns 2025).
- Model collapse: Shumailov et al. (Nature 2024); Alemohammad et al.
  (ICLR 2024, MAD); Fu et al. (ICLR 2025).
- Emulators and weather: Lehmann et al. (2605.30184); Selz & Craig (GRL
  2023); Ainslie et al. (2608.16084); Biswas (2605.24868); LUCIE / DLESyM
  (stable climate emulators).
- Recurrence and early warning: Marwan et al. (2007, RQA review); Theiler
  et al. (1992, surrogates); Scheffer et al. (Nature 2009, early warnings);
  Entropy Collapse (2512.12381).
- World models: DIAMOND (NeurIPS 2024, convergence to empty grassland
  observed by others); Oasis / Matrix-Game 2.0 (static frames after
  collapse); ReStraV (NeurIPS 2025).

---

## 6. Search log (queries run, 2026-10-05)

Extended mode unless marked (s).

1. autoregressive video diffusion long-horizon drift attractor dynamical systems analysis
2. Self Forcing long video collapse analysis prompt adherence decay over time
3. Lyapunov exponent video generation model rollout perturbation twin experiment
4. attention sink streaming video diffusion drift analysis LongLive Rolling Forcing
5. "sink-collapse" autoregressive video generation predictable frame indices
6. streaming video generation all prompts converge same end state mode collapse long rollout
7. effective rank collapse autoregressive video diffusion 2607.27036
8. memory in autoregressive video generation survey 2609.28466 (s)
9. diagnosing failure modes long video generation analysis paper error accumulation 2026
10. world model long rollout stability chaos Lyapunov divergence Genie Oasis DIAMOND
11. LLM text degeneration repetition attractor dynamical systems analysis
12. recurrence quantification analysis generated video / neural network outputs surrogate
13. "attractor" autoregressive video generation analysis
14. video diffusion color drift over-saturation spectral energy drift analysis
15. "half-life" / "memory decay" prompt conditioning streaming video generation
16. noise seed sensitivity video diffusion perturbation divergence initial noise
17. long video generation benchmark drift metric degradation curve streaming 2026
18. critical slowing down early warning signals neural network generative model collapse
19. bifurcation design parameter autoregressive generator attractor type attention window
20. video world model long rollout converges to fixed point forgetting dynamical
21. understanding / why error accumulation AR video diffusion empirical study
22. self-consuming generative models model collapse recursive training fixed point (s)
23. ML weather emulator long rollout stability spectral drift attractor Lyapunov
24. Lyapunov exponents LLMs chaos sensitivity generation trajectory
25. video generation stochastic dynamical system / ergodic / stationary distribution
26. cross-prompt diversity decreases over time long video generation
27. CachedSearch 2607.23159 (s); 28. GEARS 2608.29322 (s)
29. comparison Self Forcing Rolling Forcing LongLive minute-long failure analysis
30. chaos / butterfly effect video diffusion noise perturbation frames diverge
31. dynamical systems lens generative video rollouts attractors basins
32. real vs generated video temporal statistics DINOv2 trajectories
33. test-time scaling search streaming video generation drift beam / best-of-N
34. test-time training weight adaptation long video generation drift
35. WorldRoamBench (s); 36. Self Forcing sink size / local window ablation drift
37. iterated image-to-image diffusion converges attractor "telephone"
38. Infinity-RoPE / Rolling Sink (s); 39. exponential relaxation time fit video quality decay
40. CLIP score vs time long AR video prompt fidelity
41. twin rollouts identical seed one noise chunk divergence
42. phase-randomized surrogate generated video (s)
43. AI weather models beyond two weeks long rollouts (s)
44. anatomy of drift / phases of collapse long AR video
45. streaming video diffusion phase transition / fixed point collapse
46. Selz Craig butterfly effect AI weather (s); 47. Learning to Break the Loop (s)
48. probability-flow ODE Lyapunov stability; 49. prompt forgetting curve AR video
50. Deep Forcing sink analysis (s); 51. Head Forcing (s)
52. inter-seed divergence over time long video; 53. Matrix-Game / Oasis degradation analysis
54. OpenReview ICLR 2027 AR video drift dynamical; 55. "Where do videos go to die" (s)
56. Stable Video Infinity (s); 57. contraction / dissipative latent dynamics AR video
58. recursive LLM loops perturbation dose (s); 59. eigenanalysis AR neural emulators (s)
60. "video" diffusion "Lyapunov" 2026; 61. prompt-independent global end state streaming video
62. Entropy-guided k-Guard (s)

Pages read: 2606.22370, 2607.27110, 2610.03120, 2503.10704, 2601.16914
(abs and html), 2610.01661, 2608.28694, 2608.14706, 2608.29322,
2607.27036, 2605.24868, 2609.38691, 2512.12381, 2602.14027, 2512.04678,
2606.14732, 2607.23159, 2609.19729, 2605.21028, 2605.30184, 2510.02283,
2604.10103, 2502.15208, 2507.00583, PMC12827715.

**Gaps and caveats:**

- OpenReview submissions for ICLR 2027 (deadline around late September
  2026) are not yet public or indexed. A concurrent submission could
  appear when they are released, so recheck in mid-October and again
  before November 16.
- Deep Forcing, Rolling Sink, Head Forcing, and Infinity-RoPE details
  rest on snippets.
- Self-Forcing++'s exact metrics (it may define a "visual stability"
  score) were not verified from the full text.
