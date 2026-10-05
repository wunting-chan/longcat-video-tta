# Streaming × continual learning — field buckets (2026-09-20)

**Not a submit. No GPU.** Follow-on to the hostile review
(`2026-09-20_fast_slow_streaming_review.md`). User accepted
the criticism and asked to keep developing: what the
*fields* actually fight, how they fail, which approaches
exist, and what each approach is for.

Canvas: `canvases/streaming-cl-field-buckets.canvas.tsx`.

---

## 0. Two fields that share a word

| Field | “Streaming” means | Main fight |
|---|---|---|
| **Causal video generators** (Self Forcing, Rolling, Reward Forcing, LongLive, CausVid, ARL²) | Emit the next frame now, for minutes, on one GPU | Latency + self-rollout drift. Weights frozen after distill. |
| **Continual / lifelong / online CL** (EWC, replay, CLS-ER, lifelong VDM, Nested Learning) | Learn from a non-i.i.d. stream without starting over | Stability–plasticity. The data distribution moves. |

Our leftover-growth case sits **on the join**. The SOTA
video papers do not treat \(p(\text{environment}_t)\) as
the object. The CL papers do not emit 17–23 FPS video.

---

## 1. Challenge buckets (problems people name)

| Bucket | The problem in one line | Who names it |
|---|---|---|
| **A. Stability–plasticity** | Learn the new thing without wiping the old thing. Grossberg 1982. Forgetting is not the only failure: methods that never forget often cannot learn. | Every CL survey |
| **B. Non-stationary / task-free stream** | No task ID, no i.i.d. reshuffle, boundaries are blurry. The “environment” moves on several timescales. | Online CL; lifelong VDM (Minecraft / driving) |
| **C. Bounded memory** | Cannot keep every frame, every KV, every replay item. Must evict or compress. | StreamingLLM, sinks, EMA, linear attention, replay buffers |
| **D. Exposure bias / train ≠ infer** | Trained on clean history, tested on its own dirt. | Self Forcing’s title; Teacher vs Diffusion vs Self |
| **E. Long-horizon accumulation** | Small errors compound; 5 s student on a 60–240 s rollout. | Rolling, LongLive train-long, our freeze+sharpen |
| **F. Latency vs quality** | ~43–60 ms/frame after the first frame (17–23 FPS, H100, 1.3B). Extra work is a hitch or a miss. | Every streaming-gen abstract |
| **G. Identity vs living motion** | Holding the opening damps Dyn; chasing Dyn rewrites the room or twitches. | Rolling sink tax; Reward Forcing’s copy-first-frame; our prefix-match |
| **H. Associative interference** | A finite fast-weight / linear state overwrites old keys when over capacity. | FWP / DeltaNet / Titans |
| **I. Judge / label mismatch** | The cheap score is not the paper table. Official Dyn is a RAFT bit. Teacher score is not VBench. | Our atlas; DOLLAR; Reward Forcing uses VideoAlign, not the bit |
| **J. Loss of plasticity** | After a long stream the net becomes rigid and cannot take the next shift. | Recent CL (beyond forgetting) |
| **K. Resources** | Replay stores data (privacy, RAM). Expand-the-net grows params. Teacher-at-sleep needs a 14B. | Online CL SLR; our 8-GPU lock |

A–C, J–K are the CL field. D–G, F are the streaming-gen
field. H is the fast-weight field. I is ours and the
reward-distill papers. The join (B + C + F + G) is the
only place a leftover method can sit.

---

## 2. Failure-mode buckets (what actually breaks)

### Continual learning

| Mode | What you see | Typical cause |
|---|---|---|
| **Catastrophic forgetting** | Old scene / task dies when the new one is learned | Shared weights, one SGD stream |
| **Intransigence** | New scene never enters | Regularizer / freeze too tight |
| **Recency bias** | Only the last minute is right | Replay too small or decay too fast |
| **Replay collapse** | Buffer overfits a few stills; model copies them | Tiny reservoir, no diversity |
| **Loss of plasticity** | After hours of stream, updates do nothing useful | Optimizer / saturation |

### Streaming generation (field + our atlas)

| Mode | What you see | Typical cause |
|---|---|---|
| **Freeze + sharpen** | Tail is a still, too crisp | Exposure + long unroll on a short student |
| **First-frame copy** | Camera stuck on the opening | Static sink (Reward Forcing named this) |
| **Identity tax** | Face/room held, Dyn down | Rolling first-chunk sink; prefix-match |
| **Twitch** | Official Dyn yes, flicker ~0.97 | Mixed noise slots; official RAFT in the loop; crossed host |
| **Paint / IQ death** | Imaging Quality 18–54 | Weight TTA; leftover ρ; nwarp stencil |
| **New room / invented pan** | Subject < 0.68 | Wan-extend text; teacher pick that leaves the leftover |
| **Hitch / deadline miss** | FPS claim dies | Slow step or TTT-MLP in the frame loop |

### Fast-weight / linear memory

| Mode | What you see | Typical cause |
|---|---|---|
| **Overcapacity** | Old writes vanish without a decision | Sequence ≫ \(d\) ; additive outer products |
| **Un-undoable update** | “Evict frame 7” does not restore \(W\) | Sequential SGD / one running LoRA |
| **Sleep hitch** | A stall every \(K\) frames | Consolidate \(W_{\text{slow}}\) on the emit path |

---

## 3. Approach buckets (what people build, and what they fight)

Each row is a family, not a paper. The last column is
how that family usually dies.

| Approach family | Built to fight | How it usually fails |
|---|---|---|
| **Replay / rehearsal** (buffer, generative replay, feature replay) | A — forgetting, by interleaving the past | K — buffer size / privacy; F — not 50 ms; replay collapse |
| **Regularization** (EWC, SI, LwF, distill-to-old) | A — forgetting, cheaply, no buffer | Intransigence; J |
| **Isolation / expand / MoE / PackNet** | H — parameter interference | K — grows forever; routing errors |
| **Fast / slow + sleep / CLS** (hippocampus–cortex, Nested Learning, *LMs Need Sleep*) | A + B — different timescales; consolidate before fade | Occupied sentence; sleep not in FPS loop |
| **KV window + sink / EMA-sink / MemRoPE** | C + E + G — bound memory, keep a global token | First-frame copy **or** identity tax; activation-only |
| **Linear / delta / TTT / Titans / ARL²** | C + E + H — long context without quadratic KV | 1.8–2.5× vs local attn (TTT-Video); not \(W_{\text{slow}}\); ARL² already on AR video |
| **Recache / prompt switch** (LongLive) | Changing *text* condition | Not a visual environment; weights frozen |
| **Train = infer unroll** (Self Forcing, streaming long tune) | D + E — exposure, train-short-test-long | Frozen student; does not track \(p(\text{env})\) |
| **Reward / preference in distill** (Reward Forcing, DanceGRPO, VideoDPO) | G — motion vs freeze, at *train* | I — wrong judge twitches; not test-time CL |
| **Selection among legal futures** (Video-T1, Always-search) | G without editing the path | Occupied; 7–8× wall; our gate was not a title |
| **Path edit / weight TTA** (AdaSteer, TTC, nwarp, leftover ρ) | E on a *frozen* student | Paint, twitch, or no-op — atlas **NO** |
| **Lifelong train on one video stream** (Yoo et al.) | B — non-stationary env as *training* | Not a streaming generator; replay, not delta-evict |

Read the table as: **they are not confused about the
fights.** Streaming-gen papers fight D, E, F, G with
frozen weights and a cache. CL papers fight A, B, J, K
with updates that are allowed to be slow. Fast-weight
papers fight C and H inside the sequence model. Nobody
honestly fights **B + F + G at once** on a few-step
Wan student.

---

## 4. What we keep developing (after the valid review)

The review stands: “delta memory + evict + step the
backbone” is not a title. We keep the **policy**, not
the stack of names.

**Locked**

- \(W_{\text{fast}}\) is linear / delta, writes are
  stored, a named evict is exact.
- \(W_{\text{slow}}\) does **not** sit in the 17–23 FPS
  loop. Sleep is rare and reported as a hitch.
- Delete is **blocked** until sleep has seen those
  frames. That is the only clause EMA-sink / ARL² /
  Titans do not already claim.
- Protocol is a **growing real leftover** or an
  explicit scene-shift stream. Not T2V self-rollout
  as the env.
- Slow loss is **not** official Dyn, not pixel GD on
  the live head (AdaSteer). Must be named before any
  GPU.
- Controls, not optional: fast-only, EMA-sink,
  decay-only (no named evict), evict-without-sleep,
  sleep-after-evict, latency table.

**Still open (the actual method work)**

1. What *is* the slow loss on the to-be-evicted
   frames? (Teacher DMD off-loop is Territory A.
   A tiny well / scene token is Rank-2 from
   2026-09-18. Pick one.)
2. What triggers sleep — capacity, scene cut,
   leftover disagreement — without becoming Pseudo’s
   skip bit?
3. Is \(W_{\text{slow}}\) the 1.3B, or a small slow
   bank (scene wells) so AdaSteer stays closed?

Until (1) is a sentence, this is still a diagram.
No 8-GPU. No remake cite-128. No n=2 letter.
