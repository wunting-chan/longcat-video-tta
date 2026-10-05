# Incoming GT vs no-GT streaming — which to pursue (2026-09-20)

**Not a submit. No GPU.** Assessment after the user
asked which scenario is worth pursuing.

---

## Verdict

Pursue **neither as a pure title.**

- Pure **no-GT streaming generation** is the field’s
  cite table (SF / RF / LongLive, 17–23 FPS). We
  already ran that wall: selection occupied, path
  edits NO, weight TTA NO. Surprise-of-GT **does not
  exist** at test. A CL / “environment” story on this
  protocol is a motivation swap.
- Pure **incoming GT of the *generated horizon*** is
  online video prediction. PSNR comes back. Lifelong
  VDM / world models occupy “learn from a real
  stream.” It is not their streaming-generation table.

**Pursue the hybrid, but do not tell a “GT is delayed
30 s” story.** That delay is not a real situation
and we should not invent it.

The real situation is **live prefix + imagined
horizon**: at time \(t\) the camera (or the file
up to \(t\)) is all the GT you have. You emit a
horizon \(H\) of *future that has not happened*.
When the next real slice arrives, leftover grows,
sleep may digest that slice, and you emit a new
horizon from the new now. \(H=30\) s is the field’s
VBench-Long yardstick, not a wait for the truth.

On disk (Panda) this is simulated the usual way:
pretend you have only frames \(<t\), generate
\(t\ldots t+H\), never train sleep on the held-out
tail. Caption-128 sources are long enough (min 55 s,
med 314 s, all ≥ 32 s) to grow \(t\).

---

## Why not pure no-GT (their streaming)

What it is: text (and maybe frame 0), then only
self-rollout. LongLive’s stream is a new **sentence**.

What we would be allowed to do: train-time DMD sleep
(eviction curriculum, test frozen); ARL²-style read of
\(W_{\text{fast}}\); a well that must be judged by
teacher / VideoAlign / a liar.

What we cannot do: world-as-judge, surprise-of-GT,
“adapt to a changing environment.” Those sentences
become false.

Worth as a **host table**, not as the method reason.
Any student we train still has to sit on VBench 30 s
no-GT continuation. That is the referee’s bar. It is
not the place the new signal lives.

---

## Why not pure incoming-GT of the future

What it is: predict chunk \(t\), then the real chunk
\(t\) arrives, forever. The label is always there.

Occupied: Yoo et al. lifelong VDM (one Minecraft /
driving stream + replay); TTT on video streams;
classical video prediction; world models.

Referee: “this is forecasting, not streaming
generation.” Official Dyn / IQ on a forecast of Panda
is a different paper. If we leak the 30 s tail’s GT
into sleep, the cite-128 continuation number is
contaminated.

Use prediction error on the **just-arrived leftover
chunk** as a *diagnostic* and as a sleep trigger.
Do not make “forecast the whole video” the headline.

---

## The hybrid (what to pursue)

Not “GT arrives, then we wait 30 s.” That is not a
scenario.

```
now ── observed GT (leftover) ──|── imagined horizon H (no GT yet)
         digest / surprise          generate; VBench on this tail
         sleep / wells
                    ── next real slice arrives, leftover grows, repeat
```

At 2 s, 4 s, 6 s of *observed* video the new slice is
incoming GT of the **past**, not a delayed future.
Predictive surprise =
\(\|\mathrm{GT}_{\text{new}}-\text{forecast}_{\text{new}}\|\)
from the previous leftover only. Mid band → sleep.
Extreme → new well or skip. Then emit horizon \(H\)
of the **unseen** future. \(H=30\) s because that is
their long table, not because truth is 30 s late.

Why this is the only honest join:

| Need | Pure no-GT | GT of the future | Leftover growth |
|---|---|---|---|
| Surprise-of-GT exists | No | Yes (leaks eval) | Yes, on the prefix |
| Field cite table (VBench continuation) | Yes | No | Yes |
| “Environment moves” is true | No | Yes | Yes (the leftover) |
| Occupied as a title | Search / ARL² / TTT-Video | Lifelong VDM / prediction | Leftover-DMD is a **control**, not the title |
| Atlas-safe | Frozen path / selection | Weight TTA if we fit the 1.3B online | Sleep off-loop; wells or train-time student |

The method object stays the one we kept: **delete
blocked until sleep writes the slow object**, slow
object = train-time student **or** a well bank.
Leftover growth is the *protocol that makes
surprise-of-GT real* without turning the paper into
forecasting.

---

## Assessment, ranked

1. **Do this:** leftover-growth protocol + no-GT 30 s
   cite table. First lock the slow object (well vs
   train-time DMD). Surprise tiers on the **arrived
   slice only**.
2. **Keep as host / ablation:** pure no-GT 30 s on
   the same checkpoint (their table).
3. **Diagnostic only:** error on the arrived slice.
   Do not headline it.
4. **Do not pursue as title:** always-on GT of the
   generated horizon; CL story on pure no-GT.

No 8-GPU. No n=2 letter. No remake cite-128 until
the hybrid spec names the slow loss and the mid-band
score (not leftover FM-OOD).
