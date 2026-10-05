# Methods since the Wan switch — talk notes (2026-08-24)

Audience: anyone. Not a methods paper. Interactive walk-through:
`~/.cursor/projects/Users-macrohard-Desktop-longcat-video-tta/canvases/wan-methods-since-switch.canvas.tsx`

Cite **medians**. Official quality = **full-clip VBench**. Paper
baseline now = Self-Forcing do-nothing. Canvas expanded 2026-08-24
evening: AdaSteer LongCat-vs-Wan + per-method sampling-space
slides. Caption WAVE=1 and AdaSteer 8v are **in flight** — Wan
AdaSteer is not called dead. Stem-prompt V2V tables stay an audit.

---

## 0. What we are trying to do

A video model that writes the future **one chunk at a time**, using
its own past as memory, slowly **freezes and over-sharpens**. That
is published. We are asking: can a **cheap test-time controller**
(no new training, no LoRA) keep motion alive for ~30 seconds
without wrecking identity?

We switched off LongCat 13.6B. We now run **Wan2.1-T2V-1.3B +
Self-Forcing causal DMD** — the student the 2025 streaming papers
already use.

---

## 1. The setup, in one picture

Think of a storyteller who has already seen **~2 seconds** of a
real video (the prefix) and must invent the next **~30 seconds**.
Every few seconds it writes a new paragraph. It cannot go back and
edit the earlier paragraphs. Those stay in its memory (a **KV
cache**). If paragraph 3 is a freeze, paragraph 4 is written *as
if the freeze were real*.

| Word | Meaning |
|---|---|
| **Self-Forcing (SF)** | The student we treat as the paper baseline. Writes in 21-latent chunks. |
| **Rolling Forcing (RF)** | A different student + a rolling sampler. Caption official: tail +22%, IQ **70.22 vs SF 71.54** (fails −1). Someone else’s host. |
| **Prefix** | First 9 latents (~2.1 s) from a real Panda clip. Never searched. |
| **k** | How many alternate futures we try before picking one. k=4 is the family width and CachedSearch’s cheap best-of-4. |
| **Gate** | A cheap check that decides whether to spend those k tries. |

Videos: first 32 of `datasets/panda_1000_480p/` (sorted). N=8 ⊂ 32 ⊂ 128.

---

## 2. How we watch a video (gates and scores)

We never look at the “true” future. There is no 30 s ground-truth
for invented motion. We watch **what the model just wrote**.

**Per-chunk signals (on decoded pixels, 0–1 RGB):**

| Signal | Plain English | How |
|---|---|---|
| **Temporal motion** | How much the picture changes frame to frame | Mean absolute pixel difference |
| **Sharpness** | How crispy / over-etched | Laplacian variance |
| **Appearance / hinge** | Did we wander from the look of the prefix? | Distance to a prefix reference window |
| **Sick** | Did this chunk freeze vs the last one? | motion < 0.8 × previous chunk (or prefix) |
| **Prefix hold-out (pseudo)** | Can a different seed better predict 3 prefix latents we hid? | MAE on held-out B |

**A gate** is: if sick / if hold-out wins / if prefix is “live,”
then spend k. Otherwise write the default seed (same as do-nothing).

**Why 0.8?** Pre-registered. Do not retune after seeing 32.

---

## 3. Why we evaluate this way

| What we report | Why | What it is not |
|---|---|---|
| **Tail motion (median)** | Did the 30 s stay alive? The failure we can see without a judge model. | Not “cinematic quality.” |
| **Win/loss vs SF** | Did typical videos move, or two outliers? | Not a p-value. |
| **Exact-SF rate** | If 18/32 match do-nothing, the gate never fired. | Not a quality win. |
| **VBench subject / IQ** | Did we buy motion by ruining identity or imaging? Locked bars: IQ ≥ SF−1, subject ≥ SF−0.02. | Not PSNR (no 30 s GT). |
| **VBench dynamic_degree** | Official 0/1 “is this clip dynamic?” Often stays 0 even when tail rose. Honest: tail method ≠ Dyn method. | Not the only motion number. |
| **Flicker** | ~0.972 + Dyn 1 + subject down = crossed-sampler twitch (H1). ~0.982 is a small tax. | Not “more motion.” |
| **Full clip, not last 5 s** | That is the comparable VBench++ number. last5 lied to us on I2V. | |

We cite medians because one video (I2V #26) already pulled a mean.

---

## 3b. AdaSteer — LongCat closed, Wan unconfirmed

**Question:** Does a single activation residual δ (t′ = t + δ)
help a *smaller* causal student on *30 s* AR, given it was
saturated on LongCat 14→14?

**Hypothesis:** Wan 1.3B + 30 s is more headroom than LongCat
13.6B + 14 frames. Streaming refit might matter here even though
it was NULL on LongCat native AR.

**Papers:** Our AdaSteer method tex; Huang et al. Self Forcing
limitation (quality drop past 5 s). Not LoRA. Not TTC.

**LongCat outcome (closed):** ADA ≈ NOTTA at Panda 999 / UCF 932.
Stream δ NULL (N=8). Placement adaln = residual, both lost.

**Wan confirmation (IN FLIGHT):** `ada_fixed` / `ada_stream` /
`ada_resid` N=8, jobs **16314667–669**. Cite vs caption notta.
Do not call Wan dead from the LongCat table.

---

## 4. Chapter 1 — I2V from a still (discovery only)

**Question:** If we start from one image, does best-of-k search
fix 30 s freeze?

**Hypothesis:** More seeds + a handcrafted “looks like the start”
score will unfreeze the tail. Gating will keep the gain cheaper.

**Papers:** Self-Forcing (student dies past 5 s); VBench / VBench-Long
(full-clip protocol); Early Failure Detection / CachedSearch
(spend compute only when needed).

**What we did:** 16 then 32 stills. Do-nothing vs always k=4 vs a
gate. **Do not scale this.** Task / N / suite are not the field table.

| Fact | Number | Call |
|---|---|---|
| 5 s drift | sharp +11%, motion −14% | Mild |
| 30 s drift | sharp **+167%**, motion **−60%** (15/16) | Sharpen + freeze |
| Handcrafted last-chunk (32) | notta 3.68 / always 2.97 / gated 3.04 | Search moves *our* score |
| Official VBench full clip | Aes 0.587 / 0.593 / 0.591; IQ 71.24 / 71.28 / 71.19; Dyn **0** | **Tie** |
| Verifier vs IQ (last5) | ρ +0.23 to +0.33 | Anti-aligned: we punish sharpness, MUSIQ rewards it |

**Lesson:** The score we optimized does not track official quality.
Gating is an efficiency story, not a VBench win. Closed I2V-32.

---

## 5. Chapter 2 — V2V: give it real motion, then continue

**Question:** Same claim, honest task. Visual history → long AR.

**Hypothesis:** A moving prefix is a better test than a still.
Do-nothing on SF will still freeze. Controllers should beat **SF
notta**, not ride RF’s host gap.

**Papers:** Relax / Rolling Forcing (T2V self-continuation tables);
StreamingT2V (long AR stands still); History Guidance (identity
vs dynamics).

**Protocol:** 9 real latents + 6 × 21 generated. Same first 32.

**Ours:** Prefix-match, Rewind, Sick-search, Pseudo, Always-search
(ablation of Pseudo).

**Not ours:** Self-Forcing do-nothing (paper baseline); Rolling
Forcing (comparison host); Sink (LongLive); Crossed host (SF×RF
swap).

| Method | Whose | What we actually do |
|---|---|---|
| Prefix-match | Ours | Four futures; keep the one closest to the real 2 s opening |
| Rewind | Ours | If the chunk froze, try one other seed; keep it only if it moved more |
| Sick-search | Ours | One try until a freeze, then four tries |
| Pseudo | Ours | Hide the last bit of the opening; search the tail only if another seed predicts it better |
| Always-search | Ours (ablation) | Four tries every chunk; no “should we search?” check |
| Self-Forcing do-nothing | Baseline (Huang et al.) | Default seed every chunk |
| Rolling Forcing | Other host | Published student + overlapping window |
| Sink | LongLive | Keep opening tokens in attention; no search |
| Crossed host | Their pieces | SF student + RF window (or reverse) |

### 5.0b Experimental settings (sampling-space N=32)

| Item | Setting |
|---|---|
| Model | Wan2.1-T2V-1.3B + Self-Forcing causal DMD. RF is comparison only. |
| Dataset | First 32 of `datasets/panda_1000_480p` (Panda-70M 480p), sorted |
| Task | V2V prefix continuation, not I2V-from-still |
| Geometry | 480×832, 16 fps. Prefix 9 latents / 33 frames / 2.06 s. Generate 6×21 latents (~30 s). Full clip 537 frames. |
| Sampler | shift=8, CFG=1.0, seed=0. Search k=4. Sink 3 / local attn 12. |
| Evaluator | VBench / VBench++ **full clip** (Huang et al. 2023) |

These 32 used stem prompts. Caption WAVE=1 will replace the table.

### 5.0c Outcomes — official VBench only

Subject consistency, imaging quality, dynamic degree, temporal
flickering. Not our pixel-tail diagnostic.

| Method | Subject | IQ | Dyn | Flicker |
|---|---:|---:|---:|---:|
| SF do-nothing | 0.665 | 69.65 | 0 | 0.986 |
| RF (other host) | 0.702 | 70.44 | 0 | 0.983 |
| Prefix-match | 0.705 | 68.88 | 0 | 0.988 |
| rewind | 0.680 | 69.44 | 0 | 0.985 |
| sick | 0.669 | 69.13 | 0 | 0.986 |
| pseudo | 0.691 | 69.83 | **0.50** | 0.982 |
| sink | 0.646 | 69.98 | 0 | 0.977 |

Only pseudo moves official dynamic degree. Prefix-match raises
identity and stays at Dyn 0. Sink pays subject / flicker.

### 5.1 Do-nothing SF (`notta`) — the baseline

Just write the default seed every chunk. Tail **0.0135**. Subject
0.665, IQ 69.65, Dyn **0**, flicker 0.986. This is the paper zero.

### 5.2 Best-of-4, pick “closest to the prefix” (`seed_bon`)

**Q:** Does always-on search with a prefix-match pick help?  
**H:** N=8 said +35% / Dyn 0→0.5.  
**Papers:** Best-of-N; CachedSearch’s BoN-4 budget.

**How:** Every chunk, 4 seeds. Pick lowest two-sided prefix
deviation (look + motion both close to the start).

**Data (N=32):** tail **0.01235 (−8.8%)**, 12/32 beat notta. Hot
prefixes **0/7** wins — the score *damps* already-moving clips.
VBench subject **+0.039**, IQ −0.77, Dyn **0**. Identity damper.
**NO.** N=8 was lucky.

### 5.3 Live gate (`live_bon`)

**Q:** Search only if the prefix is already moving (motion ≥ 0.012).  
**H:** Skip stills, keep the N=8 gain on live clips.  
**Data:** Searches **bit-match seed_bon**. 4/6 live clips on the
second half lost. **NO.**

### 5.4 Appearance pick (`appear_bon`)

**Q:** Drop motion from the pick; only match look.  
**H:** Stops damping hot prefixes.  
**Data:** +3% median, **mean −2%**, 15/17, 12/32 = seed_bon.
Subject +0.065. **NO.**

---

## 6. Chapter 3 — The host is not our method

### 6.1 RF do-nothing (`rolling_notta`)

**Q:** If we swap the student/sampler and do nothing else, what happens?  
**H:** Rolling Forcing’s native roll should keep more motion.  
**Papers:** Rolling / Relax Forcing.

**How:** Same prefix. Their checkpoint + rolling windows. k=1.

| N | Tail vs SF | Subject | IQ | Dyn | Flicker | Call |
|---|---|---|---|---|---|---|
| 32 | **+31%** (21/11), 0.0178 vs 0.0135 | 0.702 | 70.44 | 0 | 0.983 | YES on locked bars — **host** |
| 128 | **+31%** (88/40), 0.0177 vs 0.0136 | 0.687 | 70.91 | **1** | 0.982 | Same. Dyn first-32 still 0 |

This is why later work must sit **on SF**. “Method + RF vs SF”
is mostly RF.

### 6.2 Crossed sampler (`sf_roll` / `rf_chunk`) — H1

**Q:** Is the win the weights or the sampler?  
**Data:** Tail 0.028, Dyn 1, flicker **0.972**, subject fail.
**Twitch. NO. Do not use `sf_roll`.**

---

## 7. Chapter 4 — Widgets on RF (then we moved them to SF)

Same four ideas. First on RF (wrong host for the SF-baseline
claim), then on SF.

### 7.1 Rewind

**Q:** If a chunk just froze, try one other seed and keep it only
if motion recovered.  
**H:** Temporal backtracking: the freeze is local; a resample
saves the trajectory.  
**Papers:** Temporal Backtracking Search; StreamingT2V freeze.

**How:** After each chunk, if motion < 0.8× previous, generate
one extra seed. Accept iff new motion ≥ old.

| Host | Tail vs that host | Notes |
|---|---|---|
| RF | +8% vs RF (23/2/7). Recovers **0027** 0.018→0.033 | HOLD N=32 |
| SF | **+6% vs SF** (19/5/8). 0027 0.035→0.042. **12/24 later re-freeze** | HOLD. Next = stay-on |

### 7.2 Sick-search

**Q:** After a freeze, spend k=4 and pick the liveliest feasible
candidate (motion ≥ 0.8× default).  
**H:** More tries after sickness beats one rewind.  
**Papers:** Search-while-sick (our I2V diagnosis); CachedSearch
(spend when failing).

| Host | Result | Call |
|---|---|---|
| RF | +6.5% vs RF, 8/32 exact host | HOLD, small |
| SF | **−1% vs SF**, 20/5/7, fire 29/32 | **NO.** Same pick as pseudo, applied *after* collapse |

### 7.3 Pseudo (prefix hold-out)

**Q:** Before writing the future, hide the last 3 prefix latents.
If another seed predicts them better, search the tail.  
**H:** A seed that “understands” the recent past is a better
author of the future.  
**Papers:** Prefix / teacher-forcing hold-out; not TTC.

| Host | Result | Call |
|---|---|---|
| RF | +1.3%, **18/32 exact RF** | **NO.** Dead gate |
| SF | **+37% vs SF** (25/2/5), fire 27/32, Dyn **0.5**, IQ/subject hold, tail 0.0186 vs RF 0.0178 | **HOLD — lead.** Gate is loose. Always-search will split gate vs pick |

### 7.4 Sink

**Q:** Keep a few “anchor” tokens from the start in attention
(LongLive sink).  
**H:** Identity stays; maybe motion does too.  
**Papers:** LongLive; History Guidance (identity vs dynamics).
**Not HG-f.**

| Host | Result | Call |
|---|---|---|
| RF | +24% vs RF, subject −0.016, flicker 0.977 | HOLD / no-scale |
| SF | **+72% vs SF** (30/2), subject **−0.0195** (on the line), flicker **0.977**. Wakes 0004; damps 0027 | HOLD / no-scale. Pixel-move probe |

---

## 8. Scoreboard (SF-hosted family, the claim)

Official numbers are the VBench full-clip table in §5.0c. The
tail / W/L columns below are an internal diagnostic only — not
what we put on the outcome slide.

| Method | Subject | IQ | Dyn | Flicker |
|---|---:|---:|---:|---:|
| SF notta | 0.665 | 69.65 | 0 | 0.986 |
| RF rolling | 0.702 | 70.44 | 0 | 0.983 |
| seed_bon | 0.705 | 68.88 | 0 | 0.988 |
| sf_rewind | 0.680 | 69.44 | 0 | 0.985 |
| sf_sick | 0.669 | 69.13 | 0 | 0.986 |
| sf_pseudo | 0.691 | 69.83 | **0.50** | 0.982 |
| sf_sink | 0.646 | 69.98 | 0 | 0.977 |
| always-search | — | — | — | — | — | — | — | **in flight** |

---

## 9. In flight (24 Aug 14:16)

| Job | What | Last seen |
|---|---|---|
| **16288113** | SF always k=4 | R ~2h14 on gh129 |
| **16288114** | RF always k=4 | Left the queue — **sacct** |
| **16288115** | VBench afterok both | PD Dependency |

Print this if RF’s fate is unknown:

```bash
sacct -j 16288113,16288114,16288115 --format=JobID,JobName,State,Elapsed,ExitCode -n -P
ROOT=/scratch/wc3013/longcat-video-tta/wan_experiment/results
for d in v2v_panda_sf_always_32v/sf_always_search_h30s_shard0 \
         v2v_panda_rf_always_32v/rf_always_search_h30s_shard0; do
  echo -n "$d  mp4="; find $ROOT/$d -name '*.mp4' 2>/dev/null | wc -l
done
```

---

## 10. What we will not do

No TTC. No LoRA-at-test-time. No I2V-32 scale-up. No `sf_roll`.
No scaling family to 128 tonight. No retuning the 0.8 drop after
seeing these 32.
