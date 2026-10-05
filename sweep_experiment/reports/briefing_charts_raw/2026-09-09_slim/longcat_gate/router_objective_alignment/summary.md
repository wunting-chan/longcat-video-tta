# Router objective alignment — VBench vs PSNR (9-d Block A) @ N=200

Same features, different ridge labels. OOF config picks compared per video.

## Pick overlap

| Metric | Value |
|--------|------:|
| Videos (labeled) | 998 |
| **Pick agreement** (same config) | **39.0%** (389/998) |
| Oracle agreement (VB oracle = PSNR oracle) | 12.8% |
| When oracles agree → routers agree | 34.4% |
| VB router matches VB oracle | 7.3% |
| PSNR router matches PSNR oracle | 5.3% |
| Config set Jaccard (configs ever picked) | 0.818 |

## When routers **disagree** (609 videos)

- VB router pick has **≥** PSNR-router pick on **realized VBench**: 309/609 (50.7%)
- PSNR router pick has **≥** VB-router pick on **realized PSNR**: 323/609 (53.0%)

## Rank correlation of realized metrics (per-video)

- ρ(VBench from VB pick, VBench from PSNR pick): 0.999
- ρ(PSNR from PSNR pick, PSNR from VB pick): 0.998

## Top (VB pick → PSNR pick) pairs

| VB pick | PSNR pick | Count |
|---------|-----------|------:|
| `S5_LR1e3` | `S5_LR1e3` | 387 |
| `S5_LR1e3` | `S10_LR1e3` | 383 |
| `S10_LR1e2` | `S2_LR1e2` | 193 |
| `S5_LR1e2` | `S20_LR1e2` | 5 |
| `S2_LR1e2` | `S20_LR5e3` | 3 |
| `S20_LR1e2` | `S5_LR1e2` | 2 |
| `S2_LR1e3` | `S10_LR1e3` | 2 |
| `S20_LR1e2` | `S10_LR1e2` | 2 |
| `S10_LR5e3` | `S20_LR5e3` | 2 |
| `S2_LR1e3` | `S20_LR5e3` | 2 |
| `S5_LR1e2` | `S5_LR1e2` | 2 |
| `S5_LR5e3` | `S2_LR1e2` | 2 |

## Configs used

- **VB router:** `S10_LR1e2`, `S10_LR1e3`, `S10_LR5e3`, `S20_LR1e2`, `S20_LR5e3`, `S2_LR1e2`, `S2_LR1e3`, `S5_LR1e2`, `S5_LR1e3`, `S5_LR5e3`
- **PSNR router:** `S10_LR1e2`, `S10_LR1e3`, `S20_LR1e2`, `S20_LR5e3`, `S2_LR1e2`, `S2_LR1e3`, `S2_LR5e3`, `S5_LR1e2`, `S5_LR1e3`, `S5_LR5e3`

## Interpretation

- **Low pick agreement** + **low oracle agreement** → objectives pull to different configs.
- **High ρ on cross-realized metrics** → disagreements are small in metric space.
- **When disagree:** each router should win on its own metric (sanity check).
