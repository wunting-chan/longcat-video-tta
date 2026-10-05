# Deploy PSNR router @ N=998 — Block A (9-d) → predict PSNR per config

**Features:** same 9-d ``video_caption_only`` (cuts, CLIP, DINO, texture).
**Target:** PSNR (not VBench). **Deploy:** argmax predicted PSNR → one AdaSteer.

## PSNR objective (primary)

- **N:** 998
- **OOF oracle-config match rate (PSNR oracle):** 5.3%
- **Mean PSNR (policy):** 19.4743 dB
- **Mean PSNR (fixed S10):** 19.4615 dB
- **Mean PSNR (oracle):** 19.8439 dB
- **Δ vs fixed:** +0.0128 dB
- **PSNR oracle headroom captured:** 3.3%

## VBench side effect (not optimized)

- **Mean VBench total (same picks):** 9.5667
- **Fixed S10 VBench:** 9.5730
- **VBench headroom captured (side effect):** -6.5%

## Compare to VBench-targeted router (Block A)

| Router target | PSNR Δ vs fixed | PSNR cap % | VB cap % |
|---------------|------------------:|-----------:|---------:|
| **PSNR (this run)** | +0.0128 dB | 3.3 | -6.5 |
| VBench (prior `video_caption_only`) | +0.009 dB | 1.2 | **20.8** |

## Interpretation

If PSNR cap % ≫ 1.2% while VB cap % drops, the **objective** was the bottleneck,
not the 9-d input format. If PSNR cap % stays low, features lack PSNR signal.
