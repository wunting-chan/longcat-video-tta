# The slow object — teacher-DMD sleep vs a tiny scene well (2026-09-20)

**Not a submit. No GPU.** Elaboration of the fork that
decides the paper type. Chassis from the review +
buckets: linear / delta fast memory; named evict
blocked until sleep; sleep off the 17–23 FPS loop;
protocol = growing leftover or scene-shift stream.

Atlas: Territory A is leftover-locked DMD as a
*protocol* unless a later spec names another object
(`2026-09-08_territory_a_sentence_kill.md`). Scene
wells are Rank-2 from `2026-09-18_streaming_cl_novel.md`.
Review: `2026-09-20_fast_slow_streaming_review.md`.

Canvas: `canvases/slow-object-dmd-vs-well.canvas.tsx`.

---

## 1. One chassis, two slow objects

The **fast** object is the same in both forks. It is
not the paper.

A linear / delta memory \(W_{\text{fast}}\) sits beside
the generator. Each chunk writes a stored association
\((k_t, v_t)\) (or a rank-1 delta write). Generation
reads that memory. When the matrix is full, we do
**not** decay the whole state (Titans) and we do **not**
EMA the evicted KV into a sink (Reward Forcing). We
name the oldest write, run **sleep** on those frames,
then delete that write. Delete is illegal until sleep
returns. Sleep is not in the ~50 ms emit loop.

The paper is **what sleep writes into.** That is the
slow object. Two legal answers:

| | Teacher-DMD sleep | Tiny scene well |
|---|---|---|
| Slow object | The 1.3B student (or a persistent LoRA that *is* the student) | A tiny \(E(\text{video}\mid\text{opening})\) / scene-token bank |
| Generator at test | A **new** checkpoint you distilled | Official Wan / caption SF, **frozen** |
| Paper type | Territory A — a student | Territory C — a control / judge |
| Sleep does | Holistic DMD on the leaving chunk, then evict | Fit or move a well on the leaving chunk, then evict |
| Continual-learning object | The student’s weights, trained on an eviction curriculum | The well bank, with replay so scene A does not die |

Same eviction law. Different cortex.

---

## 2. A stream, written as a movie

Shared timeline. A leftover is growing, or a scene
cut arrives. Fast memory can hold \(K\) writes.

1. Chunks \(1 \ldots K\) each write into \(W_{\text{fast}}\).
   Emit as usual. No backward through the 1.3B.
2. Chunk \(K+1\) arrives. The matrix is full. The
   oldest write is chunk 1.
3. **Sleep** on chunk 1’s frames (and whatever
   context sleep needs). Off the emit loop. Report
   the hitch if this stalls the next frame.
4. Only then: delete chunk 1’s write from
   \(W_{\text{fast}}\). Write chunk \(K+1\).
5. Repeat. Fast memory is always “the last \(K\)
   scenes.” Slow memory is “everything sleep has
   already eaten.”

The two forks diverge only at step 3.

---

## 3. Teacher-DMD sleep — the student paper

### What sleep is

Self Forcing’s machine: unroll inference with a KV
cache, score the **whole** self-made piece with the
Wan teacher (holistic DMD), step the student. Rolling
and Reward Forcing are the same machine with a
different lock / sink / sample weight.

Here the unroll is not “from noise after a text
prompt.” It is **from the frames that are about to
leave \(W_{\text{fast}}\).** Those frames are the
opening. The student emits a tail (or re-emits a
continuation of that leftover). DMD matches the
teacher on that tail. Official Dynamic Degree stays
out of the gradient. Then the write is deleted.

That is leftover-locked tail DMD **plus** a reason
the leftover is *this* leftover: it is the one the
fast memory can no longer afford. The eviction
curriculum is the object. Leftover-lock alone was
already called a protocol ablation. The extra
sentence has to be: **the student is trained on the
same “hold in linear memory, then digest, then
forget the write” law it will run at test.**

### Two versions — only one is a streaming paper

**Train-time sleep (the only streaming-legal student).**
During distillation you simulate the chassis: fast
writes, capacity hit, DMD on the leaving chunk, delete,
continue. At **test** the 1.3B is frozen. Only
\(W_{\text{fast}}\) moves. Continual learning happened
in training. Test is a student that already knows how
to live with a finite linear memory and an eviction
clock. FPS can stay in the Self Forcing band plus the
cheap delta write.

This is a student paper because you must **make a
checkpoint**. Cost class: 8 GPU, leftover loader,
fast-weight module, the eviction-DMD term. Cite
`wan_notta` / caption SF as hosts until a smoke
PASSes. Do not remake cite-128 as the first run.

**Test-time sleep with the teacher (not streaming).**
Pause every evict, run teacher DMD on the live
student, update \(W_{\text{slow}}\) on the cluster
while frames are supposed to go out. That is online
CL with a teacher in the loop. Seconds, not 50 ms.
Lifelong VDM’s cousin, not Reward Forcing’s FPS
table. Do not sell this as real-time streaming. It
is also AdaSteer-shaped: the 1.3B moves on this
stream. Our IQ 43 / 51 / 18 is the prior.

If someone says “student paper,” they mean
**train-time sleep**, not a teacher at 23 FPS.

### What it is not

- Not “V2V student beats T2V student.” That is
  A-minimum and we already called it a protocol
  ablation.
- Not VideoAlign-in-DMD (Reward Forcing).
- Not official RAFT in the loss (DOLLAR / mixctx).
- Not SFT on the real pixels of the leftover
  (the exposure-bias control Self Forcing left).
- Not EMA-sink: we do not fuse KV into a token; we
  change **weights** of a student that practiced
  the delete.

### Why a referee still might say no

“Leftover SF + ARL² / Titans, and you DMD the
evicted chunk.” The kill test has to be empirical:
a matched leftover-DMD student **without** the
blocked-evict curriculum, plus an EMA-sink host
**without** a new student. If official Dyn% and IQ
do not beat both, there is no student paper. If
they do, the object was the curriculum, not the
leftover.

### Failure modes (student fork)

| Mode | How it shows up |
|---|---|
| Protocol ablation | Leftover-DMD already matches; eviction term is a no-op |
| Identity tax | Student learns to hold chunk 1 and damps Dyn (Rolling) |
| New room | Teacher pull on the leaving chunk invents a pan |
| Twitch | Evicted-chunk loss behaves like mixctx |
| Occupied | Figure is indistinguishable from ARL² + SF |

---

## 4. Tiny scene well — the frozen-generator paper

### What sleep is

The 1.3B does not move. AdaSteer stays closed.

Sleep fits a **small** object on the leaving frames:
an energy \(E(\text{video}\mid\text{opening})\), or a
scene token \(z_s\), or a well in a bank. That object
is allowed to be low-energy only inside “same scene
and living.” A new scene opens a new well. Old wells
are **replayed** so scene A does not die when B
arrives. After the well is updated, the fast write
is deleted.

At generate time the frozen student still emits.
The well is used only in ways the atlas already
called safe: **select or skip** among legal futures
(Always-search is the existence proof), or a cheap
skip when the active well says this lock is already
in-scene and living. It is not used to rewrite the
path, the list, or the pixels. It is not a 12-config
AdaSteer router.

If the leftover is truly growing, the label for the
well can be the **arriving real chunk** (same-scene-
and-living rank). That is Rank-1’s judge on a tiny
head, not on the 1.3B. If we only have self-rollout,
the judge has to be named and will be attacked
(VideoAlign, teacher score, not official Dyn).

### Why this is continual learning without a student

CL’s cortex is “slow semantic memory.” CLS-ER built
that as an EMA of the **whole net**. We cannot do
that (AdaSteer, latency). A well bank is a cortex
that is allowed to learn: a few thousand numbers per
scene, plus replay. The hippocampus is \(W_{\text{fast}}\)
(exact delete). The neocortex is the wells, not Wan.

That is closer to how the CL field uses the words
than “we distilled another SF.”

### What it is not

- Not LatSearch unless the label is mid-trajectory
  VLM on T2V. Ours must be leftover → tail, delayed
  reality if we have it, scene bank, evict-from-
  linear-memory into the well.
- Not LongLive-RAG (retrieve *self-generated*
  latents, generator frozen, no wells-from-evict).
- Not official Dyn as \(E\).
- Not Always-search as the title (occupied). The
  well is the cheap judge; k=4 every chunk is the
  control, not the product.

### Latency

Updating a tiny head on a leaving chunk can be a
small hitch or a background job. Running k tails
every chunk is 7–8× wall (cite-128 Always). So the
deployed well should **skip or pick cheaply**, not
re-run Always-search. The existence proof can still
be “well-pick matches Always-search Dyn/IQ at much
less wall.”

### Failure modes (well fork)

| Mode | How it shows up |
|---|---|
| LatSearch with a new sentence | Head is a latent reward + prune on Wan 1.3B |
| Prefix-match well | Subject up, Dyn down |
| Motion-max well | Twitch, flicker ~0.97 |
| Replay fail | Well for scene A dies when B arrives |
| Hidden Always-search | Wall is still 7–8×; no cheapen |

---

## 5. Side by side

| Question | DMD sleep (train-time) | Scene well |
|---|---|---|
| Do we train a 1.3B? | Yes. 8-GPU class. | No. |
| What learns at test? | Only \(W_{\text{fast}}\) (cheap writes) | \(W_{\text{fast}}\) + the well bank |
| What did CL happen to? | The checkpoint, offline | The wells, online |
| Teacher at test? | No | No (unless we foolishly score with Wan) |
| Atlas risk | New student can still freeze or invent | Frozen path is safe; judge can lie |
| Occupied cousins | Leftover SF, ARL², Titans, Reward Forcing EMA | LatSearch, LongLive-RAG, CLS-ER |
| Empty clause | Eviction curriculum *in* DMD, blocked delete | Evict-into-well + delayed-reality label + replay |
| First smoke | N=8 leftover, matched leftover-DMD **without** the eviction term | N=8, well vs Always-search vs do-nothing, no 1.3B train |
| Kill | No beat of leftover-DMD and of EMA-sink | Well ≡ motion-max or prefix-match, or wall ≡ Always |

They compose later: a student trained with eviction-DMD
can still wear a well bank. Do not stack them before
one of them has a spec. The first pick **is** the
paper type.

---

## 6. What “student paper” means here

Territory A: the only legal way to change the
sampler is to **train the student on that sampler.**
Train-time DMD sleep is that sentence applied to
“hold, digest, delete.” The output is a weight file.
The table is V2V caption (or scene-shift leftover)
vs official SF / leftover-DMD / EMA-sink, full-clip
VBench, Dyn = percent of clips.

It is **not** “we do DMD at 23 FPS.” That version is
a different, slower paper and reopens weight TTA.

A-minimum (leftover unroll, no eviction law) stays
a control, not the title.

---

## 7. How to pick

Pick **DMD sleep** if we are willing to spend the
student and the claim is “the eviction curriculum
belongs in distillation.” Accept the leftover-SF +
ARL² referee and design the two matched kills.

Pick **scene well** if we want the CL words
(stability–plasticity, replay, cortex) without
moving the 1.3B, and the claim is “sleep writes a
well, not a sink token.” Accept the LatSearch
referee and design the judge + cheapen kills.

Do not pick both. Do not launch 8-GPU. Do not
letter n=2. Name the slow loss in one sentence
before any spec.
