# Surprise replay + how generation reads fast weights (2026-09-20)

**Not a submit. No GPU.** Two user questions after the
slow-object fork.

---

## 1. Surprise-weighted replay / upweight — not a title

The **class** is published.

| Work | What “surprise” is | What they do with it |
|---|---|---|
| Prioritized Experience Replay (Schaul 2016) | TD error | Replay surprising transitions more |
| MIR (Aljundi et al.) | Samples the current update would most damage | Retrieve those from the buffer |
| **Titans** (Behrouz et al. 2025) | \(\|\nabla \mathcal{M}\|\) of the associative-memory loss | Write *more* of a surprising token into neural memory; decay uses surprise × capacity |
| **SuRe** ([2511.22367](https://arxiv.org/abs/2511.22367)) | High NLL | Store / replay the most surprising sequences for LLM CL; + fast/slow LoRA EMA |
| Reward Forcing | VideoAlign motion, not surprise | Upweight high-motion *teacher* samples in DMD |
| Our AdaSteer OOD | Frozen uncond FM loss \(t=100/500/900\) | Router feature. \(\|\rho\|\approx -0.16\); Q3 cancels. Not a deployable skip |

So: “pick the sequences the model finds most surprising, replay them or upweight the update” **is Titans’ write rule and SuRe’s buffer rule.** A referee will say that sentence.

A remaining *use* (not an empty field): surprise as the
**sleep trigger** on our chassis — which write is
ineligible to evict until digested, or which leaving
chunk gets more DMD / well weight. That is SuRe’s
selection term pointed at eviction, plus Titans’
surprise definition pointed at leftover chunks. Still
cite both.

**Atlas caution.** Surprising clips on this stack are
often the *failures*: twitch, paint, new room. Alice
already said equal weight on failures teaches the
student to imitate them. Official Dyn flips on twitch.
Our OOD quintiles cancelled. If surprise = “does not
fit \(W_{\text{fast}}\) yet” (Titans), that is a legal
write signal. If surprise = “high reconstruction /
RAFT / leftover ρ,” we have already watched it kill
Imaging Quality.

A well-posed version: surprise ranks **what to keep
in the fast matrix** and **what sleep must see
before delete**, not “upweight official Dyn.” Hard
negatives stay near-misses of one opening, not the
global worst 30%.

---

## 2. Do fast weights already augment generation? Yes.

Two different “augment”:

**Read in the forward pass (the FWP / TTT meaning).**
The next hidden state is a function of
\(W_{\text{fast}}\): \(y = W_{\text{fast}}\,\phi(q)\)
or a gated residual from a TTT-MLP. Generation
*is* the memory read. This is how Titans, DeltaNet,
TTT-Video, and ARL² work.

**Residual on DiT weights (the LoRA meaning).**
\(W_{\text{slow}}+\Delta W\) in every matmul.
Temp-LoRA / AdaSteer. Not a query-key memory. Our
atlas: this paints.

If \(W_{\text{fast}}\) is only a log of writes so
sleep knows what to evict, and the DiT still attends
only to KV + sink, then fast weights are
**bookkeeping**, not an augment. That is allowed,
but do not call it FWP.

---

## 3. How streaming generation actually uses them

**Self Forcing / Rolling / Reward Forcing / LongLive
(the 17–23 FPS 1.3B students).** They do **not**
insert a Titans/TTT matrix. The emit path is:

1. Denoise the current chunk with the frozen student.
2. **Read** past context from the **KV cache**
   (softmax attention on stored \(K,V\)).
3. Optional **sink**: first-chunk tokens stay
   (Rolling), or evicted KV is EMA-fused into a
   fixed sink (Reward Forcing / MemRoPE).
4. LongLive **recaches** \(K,V\) when the *text*
   changes.

That memory is activations. Linear attention’s
identity says a fast-weight matrix *is* compressed
KV, which is why a referee can call EMA-sink the
activation twin. They never do a test-time SGD
step on a memory MLP while emitting.

**Papers that do read a fast weight while emitting
video:**

| Work | Read | Write | When |
|---|---|---|---|
| **ARL²** | All tokens in the current frame query the **same** pre-update state \(S\in\mathbb{R}^{H\times D\times D}\). Output = local softmax + linear read of \(S\). | Gated delta update | **After** the clean denoise pass (not on noisy intermediates) |
| **TTT-Video** | TTT-MLP output, **gated residual** on CogVideo-X features; local softmax stays on 3 s | Test-time train the TTT-MLP | Online, as the global sequence streams |
| **Titans** (LM; VideoTitans follow-on) | Memory as context / gate / layer next to attention | Surprise-weighted gradient into \(\mathcal{M}\) | Per token, with decay |

ARL²’s two rules matter for any chassis we write:
do not write noisy intermediates into \(S\); do not
let tokens inside one frame see different \(S\).

---

## 1b. Clarification (same day) — surprise of *incoming GT*

The user did not mean “the sample we just drew is
weird.” They meant: in a stream, **real frames arrive**;
when those frames are surprising given the model’s
forecast, replay them or upweight the digest.

That is **predictive surprise of observations**
(innovation), not surprise of generated content.

In **language** CL this is the default: the stream *is*
incoming tokens, and SuRe’s high-NLL is exactly “this
GT sequence surprised the model.” Titans’ gradient
surprise is also computed on the incoming token, not on
a drawn sample.

In **SOTA streaming video generation** that signal
**does not exist at test.** After the prompt (and
maybe a first image), no real next second arrives.
Self Forcing even *removed* GT context from training
on purpose (exposure bias). Our caption V2V uses GT
only for a **2 s leftover**, then 30 s of dream.

So the idea is standard CL / SuRe / prediction-error
replay, pointed at a protocol the forcing papers do
not run. It becomes available only if we set up a
**growing leftover** or a lifelong stream (Yoo et al.;
TTT on video streams). Then “digest the surprising
GT first” is the sleep trigger we already wanted, with
the world as the thing that surprises, not the sample.

Do not confuse with our AdaSteer OOD (leftover vs
*training* distribution). That is distribution
surprise of the prefix, not “I predicted chunk \(t\)
and the camera disagreed.”

---

## 4. What that means for us

- Surprise-as-replay/upweight is **not** novel. Use
  it only as a *policy on our eviction/sleep*, and
  cite Titans + SuRe. Do not surprise-upweight
  official Dyn or leftover ρ.
- Fast weights **already** augment streaming video
  generation in ARL² and TTT-Video. Forcing SOTA
  augments with KV + sink instead.
- If we want “fast weights augment generation,”
  the forward pass must **read** \(W_{\text{fast}}\)
  (ARL²-style), not only index it for sleep. That
  read is occupied; the empty clause remains
  **delete blocked until sleep writes the slow
  object**, plus a named slow object (train-time
  DMD or a well).
