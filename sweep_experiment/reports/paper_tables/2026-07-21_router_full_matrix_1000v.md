# Full router ablation matrix @ 1000v — blocks × {12,13 actions} × {PSNR, VBench}

**Date:** 2026-07-21
**Series:** `panda_ood_budget_1000v_preview` (N=898 paired: ≥1 config VBench/PSNR + a NO-TTA score)
**Generator:** `scripts/run_router_full_matrix.py` (5-fold OOF ridge, leakage-free, offline)
**Feature date:** `per_video_analysis/2026-07-12`
**OOD csv (unused here):** n/a

Feature blocks: **A** = video/caption (9-d), **B** = diffusion-OOD (~20-d),
**C** = VAE-profile (~130-d), plus every non-empty subset. Actions: **12** = argmax
over the 12 AdaSteer configs; **13** = argmax over {12 configs, **NO-TTA**}. `Fixed` =
best single population-mean config on the paired pool for that metric (NOT a designated
default). Oracle = **augmented** (per-video max over {12 configs, NO-TTA}); `Captured` =
(policy − fixed)/(aug-oracle − fixed). Higher is better.

> VBench-total = unweighted mean of 7 raw dims (imaging_quality/MUSIQ 0–100 dominated),
> NOT normalized VBench++ (~0.77).

## PSNR (dB) — fixed = best config `S2_LR1e2` = 19.4351 · NO-TTA ≈ 19.438

| Block | Feat | Actions | Δ vs fixed | Δ vs NOTTA | Apply% | Captured |
|---|---:|---:|---:|---:|---:|---:|
| A     |   9 | 12 | −0.0120 | −0.0148 | 100% | −3.3% |
| A     |   9 | 13 | −0.0112 | −0.0140 | 41.9% | −3.1% |
| B     |  20 | 12 | −0.0085 | −0.0113 | 100% | −2.4% |
| B     |  20 | 13 | −0.0040 | −0.0068 | 42.1% | −1.1% |
| C     | 130 | 12 | −0.0145 | −0.0173 | 100% | −4.0% |
| C     | 130 | 13 | −0.0111 | −0.0139 | 42.1% | −3.1% |
| A+B   |  29 | 12 | −0.0139 | −0.0167 | 100% | −3.9% |
| A+B   |  29 | 13 | −0.0082 | −0.0110 | 41.9% | −2.3% |
| A+C   | 139 | 12 | −0.0145 | −0.0173 | 100% | −4.0% |
| A+C   | 139 | 13 | −0.0110 | −0.0138 | 42.3% | −3.1% |
| B+C   | 150 | 12 | −0.0144 | −0.0172 | 100% | −4.0% |
| B+C   | 150 | 13 | −0.0110 | −0.0138 | 42.2% | −3.0% |
| A+B+C | 159 | 12 | −0.0148 | −0.0176 | 100% | −4.1% |
| A+B+C | 159 | 13 | −0.0110 | −0.0138 | 42.3% | −3.0% |

**Read:** every block × action is negative vs both the best fixed config and no-TTA. The
skip option (13) helps marginally (best = B/13 at −0.004 vs fixed) but never clears zero.
The best population PSNR config is `S2_LR1e2` — the *least*-adaptive budget — i.e. no-TTA ≈
minimal-adaptation is PSNR-optimal. PSNR is un-routable across all 7 feature sets.

## VBench (raw-total) — fixed = best config `S10_LR5e3` = 9.5702 · NO-TTA = 9.5699

Config-oracle (max/12) = **9.6680** (+0.098 over fixed) · Augmented-oracle (max/13 incl NO-TTA) = **10.6005** (+1.03 over NO-TTA).

| Block | Feat | Actions | Δ vs fixed | Δ vs NOTTA | Apply% | Captured |
|---|---:|---:|---:|---:|---:|---:|
| A     |   9 | 12 | −0.0041 | −0.0038 | 100% | −0.4% |
| A     |   9 | 13 | −0.1280 | −0.1276 | 21.0% | −12.4% |
| B     |  20 | 12 | −0.0029 | −0.0026 | 100% | −0.3% |
| B     |  20 | 13 | −0.1269 | −0.1266 | 40.9% | −12.3% |
| C     | 130 | 12 | −0.0069 | −0.0065 | 100% | −0.7% |
| C     | 130 | 13 | −0.1259 | −0.1255 | 41.2% | −12.2% |
| A+B   |  29 | 12 | −0.0068 | −0.0064 | 100% | −0.7% |
| A+B   |  29 | 13 | −0.1343 | −0.1340 | 41.1% | −13.0% |
| A+C   | 139 | 12 | −0.0071 | −0.0068 | 100% | −0.7% |
| A+C   | 139 | 13 | −0.1271 | −0.1267 | 40.8% | −12.3% |
| B+C   | 150 | 12 | −0.0070 | −0.0066 | 100% | −0.7% |
| B+C   | 150 | 13 | −0.1290 | −0.1286 | 41.1% | −12.5% |
| A+B+C | 159 | 12 | −0.0069 | −0.0066 | 100% | −0.7% |
| A+B+C | 159 | 13 | −0.1276 | −0.1273 | 40.6% | −12.4% |

**Read:** 12-action VBench routers are ~flat (≤ −0.007, cap ≈ −0.5%): no per-video config
routing signal (config-oracle headroom itself is only +0.098 ≈ +1%). **13-action routers
uniformly collapse to ≈ −0.13 across all 7 blocks** — adding NO-TTA as an action is
structurally harmful for VBench regardless of features/model.

## Why the 13-action VBench collapse + the oracle anomaly

The augmented-oracle (10.6005) is **+1.03 over NO-TTA** while the config-oracle is only
**+0.098 over fixed**. Adding one option (NO-TTA, mean 9.57) raising the per-video max by
~0.93 implies **NO-TTA's per-video VBench has much fatter tails than the tightly-clustered
adapted configs**. Mechanism:
- 12-action routers land on the tight config cluster → stable ≈ 9.57, flat.
- 13-action routers pick NO-TTA ~59% on *noisy* predictions → eat NO-TTA's **downside tail**
  → −0.13. The oracle banks NO-TTA's **upside tail** (+1.03) because it sees the truth.
- Hence large "oracle headroom" that is **max-of-a-fat-tailed noisy variable**, not routable
  signal — consistent with un-routability across 13 model/feature variants and even the
  observed-probe upper bound. This is a signal/variance-structure ceiling, NOT hyperparameters.

## RESOLVED (routability diagnostic, `diagnose_routability.py`, N=898)

The apparent oracle headroom is **max-over-noise, not routable signal**, for BOTH metrics.
Coverage is complete (per-config VBench = 998; NOTTA VBench = 898). NOTTA vs CONFIG marginals
are identical (mean 9.570/9.570, std 1.860/1.848), so the fat tail is neither a variance
story nor a coverage artifact — it is independence.

| quantity | PSNR | VBench | reading |
|---|---:|---:|---|
| within-video config σ | 0.2515 dB | 0.0579 | config choice barely moves the metric per video |
| mean pairwise config corr | 0.992 | 0.998 | the 12 configs are ~identical per video |
| corr(NO-TTA, config-mean) | **0.998** | **0.051** | VBench: no-TTA is an INDEPENDENT draw ⇒ aug-oracle = max-of-noise |
| oracle gain / fixed | 0.3575 dB | 0.0978 | ≈ pure-noise floor σ·E[max/12] (PSNR 0.25·1.63≈0.41) |
| **OOF R² predict oracle gain (features)** | **−0.092** | **−0.082** | negative ⇒ NO learnable routing signal |
| OOF R² (+ probe outcomes) | −0.092 | −0.082 | even observing probes (GT-needing) adds nothing to the *gain* |

**Conclusion:** there is no deployable per-video routing win to be had — not a feature/model/
hyperparameter gap, but because the per-video oracle headroom is a statistical artifact of
taking a max over noisy per-config measurements. For VBench specifically, per-video VBench-total
(MUSIQ, no-reference) is scoring noise (no-TTA↔config corr = 0.05 vs PSNR's 0.998). This is a
clean negative result that supports shipping a single fixed config (≈ no-TTA) rather than a
per-video router.
