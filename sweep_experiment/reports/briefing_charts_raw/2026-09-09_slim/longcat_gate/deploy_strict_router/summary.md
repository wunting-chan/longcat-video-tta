# Deploy router @ N=200 — structured feature blocks

**Block A** `video_caption` · **Block B** `diffusion_ood` · **Block C** `vae_inference`

No Tier-3 / probe / TTA eval metrics. Offline labels = pilot 12-config VBench only.

Feature dir: `/scratch/wc3013/longcat-video-tta/sweep_experiment/reports/per_video_analysis/2026-07-12`

### Feature blocks (concatenated in this order)

| Block | # dims | Source |
|-------|-------:|--------|
| `video_caption` | 9 | `video_features.csv` — cuts, CLIP, DINO, Laplacian, RGB |
| `diffusion_ood` | 20 | `diffusion_ood_scores.csv` — frozen base DiT @ t∈{100,500,900} |
| `vae_inference` | 130 | `vae_latent_profile_features.csv` — LongCat `encode_video` pools |

| Experiment | Blocks | # feat | Captured % | Match % | Δ vs fixed S10 |
|---|---|---:|---:|---:|---:|
| `video_caption_only` | A | 9 | -4.0 | 7.3 | -0.0038 |
| `diffusion_ood_only` | B | 20 | -1.6 | 7.5 | -0.0015 |
| `video_caption_ood` | A+B | 29 | -3.0 | 7.4 | -0.0029 |
| `vae_inference_embedding` | C | 130 | -7.1 | 10.3 | -0.0069 |
| `video_caption_ood_vae` | A+B+C | 159 | -7.6 | 10.2 | -0.0073 |

**Best in suite:** `diffusion_ood_only` @ **-1.6%** captured.
